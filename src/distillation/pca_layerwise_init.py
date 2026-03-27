"""
PCA Layer-wise Initialization for Knowledge Distillation.

Pipeline:
1. Extract teacher hidden states for all layers (reuses UMAP cache)
2. PCA-reduce each layer's states (teacher_dim -> student_dim)
   - No Procrustes alignment needed: PCA bases are deterministic and
     scale-preserving, so consecutive layers are already geometrically coherent
3. PCA-reduce embedding matrix
4. Train all student layers jointly: each layer maps PCA(h_{l-1}) -> PCA(h_l)
5. Initialize student embeddings from PCA-reduced teacher embeddings

Motivated by TDA diagnostic showing PCA preserves H0 (cluster structure)
far better than UMAP, while transformer representations are overwhelmingly
organized by linear cluster separation (H0 >> H1).
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

# Reuse shared infrastructure from UMAP module
from src.distillation.umap_layerwise_init import (
    _dataset_fingerprint,
    cache_teacher_hidden_states,
    train_all_layers,
)


# ---------------------------------------------------------------------------
# PCA reduction
# ---------------------------------------------------------------------------


def _pca_reduced_subdir(target_dim: int) -> str:
    return f"pca_reduced_dim{target_dim}"


def fit_pca_and_reduce_layer(
    hidden_states: list[torch.Tensor],
    target_dim: int,
) -> tuple[list[torch.Tensor], PCA]:
    """Fit PCA on flattened tokens, reduce to target_dim, reconstruct sequences."""
    seq_lengths = [h.shape[0] for h in hidden_states]
    all_tokens = torch.cat(hidden_states, dim=0).float().numpy()

    pca = PCA(n_components=target_dim)
    reduced_flat = torch.tensor(
        pca.fit_transform(all_tokens),
        dtype=torch.float32,
    )

    explained = pca.explained_variance_ratio_.sum()
    print(
        f"    PCA: {all_tokens.shape[0]} tokens, {all_tokens.shape[1]} -> {target_dim} dims, "
        f"explained variance: {explained:.4f}"
    )

    # Reconstruct sequences
    reduced_sequences, idx = [], 0
    for length in seq_lengths:
        reduced_sequences.append(reduced_flat[idx : idx + length])
        idx += length

    return reduced_sequences, pca


def reduce_embedding_matrix_pca(
    teacher_model_name: str,
    target_dim: int,
) -> tuple[torch.Tensor, PCA]:
    """PCA-reduce the teacher embedding weight matrix (vocab_size, teacher_dim) -> (vocab_size, target_dim)."""
    from transformers import AutoModelForCausalLM

    print(f"  Reducing embedding matrix -> {target_dim} dims")

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    embed_weight = teacher.model.embed_tokens.weight.data.float().numpy()
    del teacher
    torch.cuda.empty_cache()

    print(f"    Embedding shape: {embed_weight.shape}")

    pca = PCA(n_components=target_dim)
    reduced = torch.tensor(
        pca.fit_transform(embed_weight),
        dtype=torch.float32,
    )

    explained = pca.explained_variance_ratio_.sum()
    print(f"    Reduced: {reduced.shape}, explained variance: {explained:.4f}")
    return reduced, pca


def pca_reduce_all_layers(
    cache_path: Path,
    target_dim: int,
    teacher_model_name: str,
) -> Path:
    """PCA-reduce all cached hidden states + embeddings.

    Unlike UMAP, PCA projections are:
    - Deterministic (no random seed dependence)
    - Scale-preserving (no distance inflation)
    - Naturally aligned across layers (bases ordered by variance)
    No Procrustes alignment needed.
    """
    reduced_dir = cache_path / _pca_reduced_subdir(target_dim)
    done_marker = reduced_dir / "done.marker"

    if done_marker.exists():
        print(f"[Cache HIT] PCA-reduced states (dim={target_dim}): {reduced_dir}")
        return reduced_dir

    print(f"[Cache MISS] PCA-reducing all layers -> {reduced_dir}")
    reduced_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((cache_path / "metadata.json").read_text())
    num_layers = meta["num_layers"]

    # Reduce hidden states layer by layer
    explained_variances = []
    for l in range(num_layers + 1):
        print(f"  Layer {l}/{num_layers}:")
        hidden_states = torch.load(
            cache_path / f"hidden_states_layer_{l}.pt", weights_only=True
        )
        reduced_seqs, pca_model = fit_pca_and_reduce_layer(hidden_states, target_dim)

        torch.save(reduced_seqs, reduced_dir / f"reduced_layer_{l}.pt")
        with open(reduced_dir / f"pca_model_layer_{l}.pkl", "wb") as f:
            pickle.dump(pca_model, f)

        explained_variances.append(float(pca_model.explained_variance_ratio_.sum()))
        del hidden_states, reduced_seqs, pca_model

    # Reduce embedding matrix
    print("  Embedding matrix:")
    reduced_emb, pca_emb = reduce_embedding_matrix_pca(teacher_model_name, target_dim)
    torch.save(reduced_emb, reduced_dir / "reduced_embeddings.pt")
    with open(reduced_dir / "pca_model_embeddings.pkl", "wb") as f:
        pickle.dump(pca_emb, f)

    # Save metadata
    pca_meta = {
        "target_dim": target_dim,
        "explained_variances_per_layer": explained_variances,
        "avg_explained_variance": float(np.mean(explained_variances)),
    }
    (reduced_dir / "pca_metadata.json").write_text(json.dumps(pca_meta, indent=2))
    print(
        f"\n  Avg explained variance across layers: {pca_meta['avg_explained_variance']:.4f}"
    )

    # Mark done
    done_marker.write_text("done")
    print(f"PCA reduction complete: {reduced_dir}")
    return reduced_dir


# ---------------------------------------------------------------------------
# Embedding initialization
# ---------------------------------------------------------------------------


def initialize_non_layer_components(
    student_model, reduced_dir: Path, device: str = "cuda"
):
    """Initialize embeddings from PCA-reduced teacher embeddings."""
    reduced_emb = torch.load(reduced_dir / "reduced_embeddings.pt", weights_only=True)
    reduced_emb = reduced_emb.to(
        device=device, dtype=student_model.model.embed_tokens.weight.dtype
    )

    student_model.model.embed_tokens.weight.data.copy_(reduced_emb)

    if hasattr(student_model, "lm_head"):
        if (
            student_model.lm_head.weight.data_ptr()
            != student_model.model.embed_tokens.weight.data_ptr()
        ):
            student_model.lm_head.weight.data.copy_(
                reduced_emb.to(dtype=student_model.lm_head.weight.dtype)
            )
            print("  Initialized embed_tokens + lm_head (untied weights)")
        else:
            print("  Initialized embed_tokens (+ tied lm_head)")
    else:
        print("  Initialized embed_tokens")

    print("  Embeddings initialized from PCA-reduced teacher embeddings")
    print(f"  Final RMSNorm: left at default (ones)")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _student_cache_key(
    strategy: str,
    teacher_model_name: str,
    hidden_size: int,
    scale_factor: float,
    num_samples: int,
    max_length: int,
    init_epochs: int,
    init_batch_size: int,
    init_lr: float,
    init_weight_decay: float,
    init_cosine_loss_weight: float,
    init_max_grad_norm: float,
    init_warmup_fraction: float,
    dataset_fingerprint: str = "wikitext",
) -> str:
    """Build a deterministic cache key for an initialized student model."""
    import hashlib

    raw = (
        f"{strategy}|{teacher_model_name}|{hidden_size}|{scale_factor}|"
        f"{num_samples}|{max_length}|{init_epochs}|{init_batch_size}|"
        f"{init_lr}|{init_weight_decay}|{init_cosine_loss_weight}|"
        f"{init_max_grad_norm}|{init_warmup_fraction}|{dataset_fingerprint}"
    )
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:10]
    safe_name = teacher_model_name.replace("/", "_").replace("\\", "_")
    return f"{strategy}_student_{safe_name}_h{hidden_size}_{short_hash}"


def pca_layerwise_init(
    teacher_model_name: str,
    hidden_size: Optional[int],
    scale_factor: float,
    cache_dir: str,
    num_samples: int = 5000,
    max_length: int = 128,
    init_epochs: int = 10,
    init_batch_size: int = 16,
    init_lr: float = 1e-4,
    init_weight_decay: float = 0.01,
    init_cosine_loss_weight: float = 0.1,
    init_max_grad_norm: float = 1.0,
    init_warmup_fraction: float = 0.1,
    dataset_name: str = "wikitext",
    dataset_path: Optional[str] = None,
    dataset_hf_path: Optional[str] = None,
    dataset_hf_config: str = "",
    dataset_hf_split: str = "train",
    dataset_streaming: bool = False,
    device: str = "cuda",
):
    """
    Full PCA layer-wise initialization pipeline.

    Returns:
        (student_model, student_config)
    """
    from src.distillation.student_factory import create_student_model
    from transformers import AutoConfig, AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("PCA Layer-wise Initialization")
    print("=" * 60)

    dataset_name = str(dataset_name or "")
    dataset_path = str(dataset_path or dataset_name)
    dataset_hf_path = str(dataset_hf_path or "")
    dataset_fp = _dataset_fingerprint(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_hf_path=dataset_hf_path,
        dataset_hf_config=dataset_hf_config,
        dataset_hf_split=dataset_hf_split,
        dataset_streaming=bool(dataset_streaming),
    )

    # Determine student hidden size early (needed for cache key)
    if hidden_size is None:
        teacher_config = AutoConfig.from_pretrained(teacher_model_name)
        hidden_size = max(64, int(teacher_config.hidden_size * scale_factor))
        hidden_size = ((hidden_size + 63) // 64) * 64

    # Check for cached initialized student
    student_key = _student_cache_key(
        "pca",
        teacher_model_name,
        hidden_size,
        scale_factor,
        num_samples,
        max_length,
        init_epochs,
        init_batch_size,
        init_lr,
        init_weight_decay,
        init_cosine_loss_weight,
        init_max_grad_norm,
        init_warmup_fraction,
        dataset_fingerprint=dataset_fp,
    )
    student_cache_path = Path(cache_dir) / student_key
    student_marker = student_cache_path / "done.marker"

    if student_marker.exists():
        print(f"[Cache HIT] Initialized student model: {student_cache_path}")
        student_config = AutoConfig.from_pretrained(str(student_cache_path))
        student_model = AutoModelForCausalLM.from_pretrained(
            str(student_cache_path),
            torch_dtype=torch.float32,
        ).to(device)
        return student_model, student_config

    print(f"[Cache MISS] Will save initialized student to: {student_cache_path}")

    # Step 1: Cache teacher hidden states (shared with UMAP pipeline)
    print("\n--- Stage 1: Extract teacher hidden states ---")
    hs_cache = cache_teacher_hidden_states(
        teacher_model_name,
        num_samples,
        max_length,
        cache_dir,
        device,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        dataset_hf_path=dataset_hf_path,
        dataset_hf_config=dataset_hf_config,
        dataset_hf_split=dataset_hf_split,
        dataset_streaming=bool(dataset_streaming),
    )

    # Step 2: PCA reduce all layers + embeddings
    print("\n--- Stage 2: PCA reduction ---")
    reduced_dir = pca_reduce_all_layers(
        hs_cache,
        hidden_size,
        teacher_model_name,
    )

    # Step 3: Create random-init student
    print("\n--- Stage 3: Create student model ---")
    student_model, student_config = create_student_model(
        teacher_model_name=teacher_model_name,
        scale_factor=scale_factor,
        init_strategy="random",
        hidden_size=hidden_size,
        device=device,
    )

    # Step 4: Train all layers jointly (reuses UMAP training code)
    print("\n--- Stage 4: Train all layers ---")
    num_layers = student_config.num_hidden_layers
    student_model = train_all_layers(
        student_model,
        reduced_dir,
        num_layers,
        epochs=init_epochs,
        batch_size=init_batch_size,
        learning_rate=init_lr,
        weight_decay=init_weight_decay,
        cosine_loss_weight=init_cosine_loss_weight,
        max_grad_norm=init_max_grad_norm,
        warmup_fraction=init_warmup_fraction,
        device=device,
    )

    # Step 5: Initialize non-layer components
    print("\n--- Stage 5: Initialize embeddings ---")
    initialize_non_layer_components(student_model, reduced_dir, device)

    # Step 6: Cache the fully initialized student
    print(f"\n--- Stage 6: Saving initialized student to cache ---")
    generation_config = getattr(student_model, "generation_config", None)
    model_config = getattr(student_model, "config", None)
    if generation_config is not None:
        pad_id = getattr(generation_config, "pad_token_id", None)
        needs_pad_fix = True
        if pad_id is not None:
            try:
                needs_pad_fix = int(pad_id) < 0
            except (TypeError, ValueError):
                needs_pad_fix = True
        if needs_pad_fix:
            fallback_pad = None
            if model_config is not None:
                cfg_pad = getattr(model_config, "pad_token_id", None)
                if cfg_pad is not None:
                    try:
                        if int(cfg_pad) >= 0:
                            fallback_pad = int(cfg_pad)
                    except (TypeError, ValueError):
                        pass
            if fallback_pad is None:
                eos_id = getattr(generation_config, "eos_token_id", None)
                if isinstance(eos_id, (list, tuple)):
                    eos_id = eos_id[0] if eos_id else None
                if eos_id is not None:
                    try:
                        if int(eos_id) >= 0:
                            fallback_pad = int(eos_id)
                    except (TypeError, ValueError):
                        pass
            generation_config.pad_token_id = 0 if fallback_pad is None else fallback_pad
    if model_config is not None and generation_config is not None:
        cfg_pad = getattr(model_config, "pad_token_id", None)
        needs_cfg_fix = True
        if cfg_pad is not None:
            try:
                needs_cfg_fix = int(cfg_pad) < 0
            except (TypeError, ValueError):
                needs_cfg_fix = True
        if needs_cfg_fix:
            model_config.pad_token_id = int(generation_config.pad_token_id)

    student_cache_path.mkdir(parents=True, exist_ok=True)
    student_model.save_pretrained(str(student_cache_path))
    student_config.save_pretrained(str(student_cache_path))
    student_marker.write_text("done")
    print(f"  Saved to {student_cache_path}")

    print("\n" + "=" * 60)
    print("PCA Layer-wise Initialization Complete")
    print("=" * 60)

    return student_model, student_config
