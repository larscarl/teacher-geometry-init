"""
UMAP Layer-wise Initialization for Knowledge Distillation.

Pipeline:
1. Extract teacher hidden states for all layers, cache to disk
2. UMAP-reduce each layer's states (teacher_dim -> student_dim)
3. Chain Procrustes alignment: align layer l to layer l-1 so consecutive
   layers share a consistent coordinate frame
4. UMAP-reduce embedding matrix
5. Train all student layers jointly: each layer maps aligned(h_{l-1}) -> aligned(h_l)
6. Initialize student embeddings from UMAP-reduced teacher embeddings
"""

import hashlib
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from umap import UMAP
from tqdm import tqdm
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------


def _cache_key(
    teacher_model_name: str,
    num_samples: int,
    max_length: int,
    dataset_name: str = "wikitext",
) -> str:
    raw = f"{teacher_model_name}|{num_samples}|{max_length}|{dataset_name}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    safe_name = teacher_model_name.replace("/", "_").replace("\\", "_")
    safe_dataset = dataset_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{safe_name}_{safe_dataset}_{num_samples}s_{max_length}l_{short_hash}"


def _reduced_subdir(target_dim: int) -> str:
    return f"reduced_dim{target_dim}"


# ---------------------------------------------------------------------------
# Procrustes alignment utilities
# ---------------------------------------------------------------------------


def _flatten_sequences(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    """Flatten list of [seq_len_i, dim] tensors into [total_tokens, dim]."""
    lengths = [s.shape[0] for s in sequences]
    return torch.cat(sequences, dim=0), lengths


def _unflatten_sequences(flat: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
    """Reconstruct list of [seq_len_i, dim] tensors from flat [total_tokens, dim]."""
    sequences, idx = [], 0
    for length in lengths:
        sequences.append(flat[idx : idx + length])
        idx += length
    return sequences


def procrustes_fit(
    X: torch.Tensor, Y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute orthogonal Procrustes alignment: find R such that Y_aligned ≈ X.

    Solves: R = argmin ||X - (Y - μ_Y) R^T - μ_X||_F  s.t. R^T R = I

    Args:
        X: [N, D] reference points (anchor layer)
        Y: [N, D] points to align (must have same N and D as X)

    Returns:
        R: [D, D] orthogonal rotation matrix
        mu_X: [D] mean of X
        mu_Y: [D] mean of Y
    """
    mu_X = X.mean(dim=0)
    mu_Y = Y.mean(dim=0)

    X_c = X - mu_X
    Y_c = Y - mu_Y

    # SVD of cross-covariance: X_c^T Y_c = U Σ V^T
    # Optimal rotation: R = V U^T
    M = X_c.T @ Y_c  # [D, D]
    U, S, Vt = torch.linalg.svd(M)
    V = Vt.T

    # Handle reflection: ensure det(R) = +1
    d = torch.det(V @ U.T)
    sign_correction = torch.ones(X.shape[1], dtype=X.dtype)
    sign_correction[-1] = torch.sign(d)
    R = V * sign_correction.unsqueeze(0) @ U.T

    return R, mu_X, mu_Y


def procrustes_transform(
    Y: torch.Tensor, R: torch.Tensor, mu_X: torch.Tensor, mu_Y: torch.Tensor
) -> torch.Tensor:
    """Apply Procrustes alignment: Y_aligned = (Y - μ_Y) R^T + μ_X."""
    return (Y - mu_Y) @ R.T + mu_X


def chain_procrustes_align(
    reduced_layers: list[list[torch.Tensor]],
) -> tuple[list[list[torch.Tensor]], list[dict]]:
    """Chain-align UMAP-reduced layers using Procrustes: layer l aligned to layer l-1.

    Layer 0 is the anchor. Layer 1 is aligned to layer 0, layer 2 to the
    (already aligned) layer 1, and so on. This ensures consecutive layers
    share a consistent coordinate frame, matching the sequential forward pass.

    Args:
        reduced_layers: list of (num_layers+1) entries, each a list of
            [seq_len_i, dim] tensors (per-sequence UMAP-reduced states)

    Returns:
        aligned_layers: same structure, with layers 1..L aligned
        alignment_params: list of dicts with R, mu_X, mu_Y per layer pair
    """
    num_layers_plus_one = len(reduced_layers)
    aligned_layers = [reduced_layers[0]]  # layer 0 = anchor, unchanged
    alignment_params = []

    for l in range(1, num_layers_plus_one):
        # Flatten anchor (already-aligned l-1) and current layer l
        anchor_flat, lengths = _flatten_sequences(aligned_layers[l - 1])
        current_flat, _ = _flatten_sequences(reduced_layers[l])

        assert (
            anchor_flat.shape == current_flat.shape
        ), f"Shape mismatch at layer {l}: {anchor_flat.shape} vs {current_flat.shape}"

        # Fit Procrustes: align current -> anchor
        R, mu_anchor, mu_current = procrustes_fit(anchor_flat, current_flat)

        # Transform
        aligned_flat = procrustes_transform(current_flat, R, mu_anchor, mu_current)
        aligned_seqs = _unflatten_sequences(aligned_flat, lengths)
        aligned_layers.append(aligned_seqs)

        # Diagnostics
        residual = torch.norm(anchor_flat - aligned_flat, dim=-1).mean().item()
        cos_before = (
            nn.functional.cosine_similarity(anchor_flat, current_flat, dim=-1)
            .mean()
            .item()
        )
        cos_after = (
            nn.functional.cosine_similarity(anchor_flat, aligned_flat, dim=-1)
            .mean()
            .item()
        )
        print(
            f"    Layer {l} -> {l-1}: residual={residual:.4f}, "
            f"cos_sim {cos_before:.4f} -> {cos_after:.4f}"
        )

        alignment_params.append(
            {
                "R": R,
                "mu_anchor": mu_anchor,
                "mu_current": mu_current,
            }
        )

    return aligned_layers, alignment_params


# ---------------------------------------------------------------------------
# Stage 1: Extract teacher hidden states
# ---------------------------------------------------------------------------


def extract_all_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    device: str = "cuda",
) -> list[list[torch.Tensor]]:
    """
    Run teacher on all texts, extract hidden states for ALL layers.

    Returns:
        List of length (num_layers + 1), where entry[l] is a list of
        tensors each shaped [seq_len_i, hidden_dim]. Index 0 = embedding
        output, index L = output of last decoder layer.
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    all_hidden = [[] for _ in range(num_layers + 1)]

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting hidden states"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            for l in range(num_layers + 1):
                h = outputs.hidden_states[l].squeeze(0).cpu()
                all_hidden[l].append(h)

    return all_hidden


def cache_teacher_hidden_states(
    teacher_model_name: str,
    num_samples: int,
    max_length: int,
    cache_dir: str,
    device: str = "cuda",
    dataset_path: str = "wikitext",
    dataset_name: str = "wikitext",
) -> Path:
    """Extract teacher hidden states and save to disk. Skip if cache exists.

    Args:
        dataset_name: Either "wikitext" (loads from HuggingFace) or a local path
            to a pre-chunked dataset (e.g., "data/thunderbird/dataset").
    """
    key = _cache_key(teacher_model_name, num_samples, max_length, dataset_name)
    cache_path = Path(cache_dir) / key
    metadata_path = cache_path / "metadata.json"

    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if (
            meta.get("teacher") == teacher_model_name
            and meta.get("num_samples") == num_samples
            and meta.get("max_length") == max_length
            and meta.get("dataset_name", "wikitext") == dataset_name
        ):
            print(f"[Cache HIT] Teacher hidden states: {cache_path}")
            return cache_path

    print(f"[Cache MISS] Extracting teacher hidden states -> {cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load teacher
    print(f"Loading teacher: {teacher_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    teacher.eval()

    # Load data — either from HuggingFace or local pre-chunked dataset
    texts = _load_texts_for_hidden_states(
        dataset_path, tokenizer, num_samples, max_length
    )
    print(f"Got {len(texts)} texts from '{dataset_name}'")

    # Extract
    all_hidden = extract_all_hidden_states(
        teacher, tokenizer, texts, max_length, device
    )
    num_layers = teacher.config.num_hidden_layers

    # Save per-layer
    for l in range(num_layers + 1):
        layer_path = cache_path / f"hidden_states_layer_{l}.pt"
        torch.save(all_hidden[l], layer_path)
        total_tokens = sum(h.shape[0] for h in all_hidden[l])
        print(
            f"  Saved layer {l}: {len(all_hidden[l])} seqs, {total_tokens} tokens -> {layer_path.name}"
        )

    # Save metadata
    metadata = {
        "teacher": teacher_model_name,
        "num_samples": num_samples,
        "max_length": max_length,
        "num_layers": num_layers,
        "hidden_dim": teacher.config.hidden_size,
        "num_sequences": len(texts),
        "dataset_name": dataset_name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Free memory
    del teacher, all_hidden
    torch.cuda.empty_cache()

    print(f"Teacher hidden states cached to {cache_path}")
    return cache_path


def _load_texts_for_hidden_states(
    dataset_path: str,
    tokenizer,
    num_samples: int,
    max_length: int,
) -> list[str]:
    """Load text samples for teacher hidden state extraction.

    Supports:
        - "wikitext": loads from HuggingFace (original behavior)
        - local path: loads pre-chunked dataset, decodes token IDs back to text
    """
    local_path = Path(dataset_path)

    # Check for local pre-chunked dataset (e.g., Thunderbird)
    if local_path.exists() and (local_path / "dataset_info.json").exists():
        return _load_texts_from_prechunked(local_path, tokenizer, num_samples)
    if local_path.exists() and (local_path / "dataset").exists():
        return _load_texts_from_prechunked(
            local_path / "dataset", tokenizer, num_samples
        )
    if local_path.is_file() and local_path.suffix == ".jsonl":
        return _load_texts_from_jsonl(local_path, tokenizer, num_samples)
    if local_path.is_dir():
        jsonl_files = sorted(local_path.glob("*.jsonl"))
        if jsonl_files:
            preferred_names = {"train.jsonl", "trainset.jsonl"}
            preferred = [p for p in jsonl_files if p.name in preferred_names]
            chosen = preferred[0] if preferred else jsonl_files[0]
            return _load_texts_from_jsonl(chosen, tokenizer, num_samples)

    # Default: HuggingFace dataset
    print(f"Loading {num_samples} samples from WikiText...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [t for t in dataset["text"] if len(t) > 50][:num_samples]
    return texts


def _load_texts_from_prechunked(
    path: Path,
    tokenizer,
    num_samples: int,
) -> list[str]:
    """Load texts from a pre-chunked dataset (already tokenized).

    Since the dataset stores input_ids, we decode them back to text
    for the hidden state extraction pipeline.
    """
    from datasets import load_from_disk

    print(f"Loading pre-chunked dataset from: {path}")
    dataset = load_from_disk(str(path))
    train_data = dataset["train"]

    n = min(num_samples, len(train_data))
    texts = []
    for i in range(n):
        input_ids = train_data[i]["input_ids"]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        texts.append(text)

    print(f"  Decoded {len(texts)} samples from pre-chunked dataset")
    return texts


def _load_texts_from_jsonl(
    path: Path,
    tokenizer,
    num_samples: int,
) -> list[str]:
    """Load texts from local JSONL prepared splits.

    Supports rows with either:
      - `text`
      - `input_ids` (decoded back to text)
    """
    from datasets import load_dataset

    print(f"Loading prepared JSONL from: {path}")
    dataset = load_dataset("json", data_files={"data": str(path)}, split="data")

    n = min(num_samples, len(dataset))
    texts = []
    for i in range(n):
        row = dataset[i]
        if "text" in row and row["text"] is not None:
            text = str(row["text"])
        elif "input_ids" in row and row["input_ids"] is not None:
            text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
        else:
            continue

        if len(text) > 0:
            texts.append(text)

    print(f"  Loaded {len(texts)} samples from JSONL")
    return texts


# ---------------------------------------------------------------------------
# Stage 2: UMAP reduction
# ---------------------------------------------------------------------------


def fit_umap_and_reduce_layer(
    hidden_states: list[torch.Tensor],
    target_dim: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> tuple[list[torch.Tensor], UMAP]:
    """Fit UMAP on flattened tokens, reduce to target_dim, reconstruct sequences."""
    seq_lengths = [h.shape[0] for h in hidden_states]
    all_tokens = torch.cat(hidden_states, dim=0).numpy()

    print(
        f"    UMAP: {all_tokens.shape[0]} tokens, {all_tokens.shape[1]} -> {target_dim} dims"
    )

    umap_model = UMAP(
        n_components=target_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        verbose=False,
    )
    reduced_flat = torch.tensor(
        umap_model.fit_transform(all_tokens),
        dtype=torch.float32,
    )

    # Reconstruct sequences
    reduced_sequences, idx = [], 0
    for length in seq_lengths:
        reduced_sequences.append(reduced_flat[idx : idx + length])
        idx += length

    return reduced_sequences, umap_model


def reduce_embedding_matrix(
    teacher_model_name: str,
    target_dim: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> tuple[torch.Tensor, UMAP]:
    """UMAP-reduce the teacher embedding weight matrix (vocab_size, teacher_dim) -> (vocab_size, target_dim)."""
    print(f"  Reducing embedding matrix -> {target_dim} dims")

    config = AutoConfig.from_pretrained(teacher_model_name)
    # Load only the embedding weights
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    embed_weight = teacher.model.embed_tokens.weight.data.numpy()
    del teacher
    torch.cuda.empty_cache()

    print(f"    Embedding shape: {embed_weight.shape}")

    umap_model = UMAP(
        n_components=target_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        verbose=False,
    )
    reduced = torch.tensor(
        umap_model.fit_transform(embed_weight),
        dtype=torch.float32,
    )

    print(f"    Reduced embedding shape: {reduced.shape}")
    return reduced, umap_model


def umap_reduce_all_layers(
    cache_path: Path,
    target_dim: int,
    teacher_model_name: str,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
) -> Path:
    """UMAP-reduce all cached hidden states + embeddings, then chain-align with Procrustes.

    After independent per-layer UMAP reduction, consecutive layers live in
    arbitrarily rotated coordinate systems. Chain Procrustes alignment fixes
    this by aligning layer l to layer l-1, ensuring the student sees
    geometrically consistent input-output pairs during training.
    """
    reduced_dir = cache_path / _reduced_subdir(target_dim)
    done_marker = reduced_dir / "done.marker"

    if done_marker.exists():
        print(f"[Cache HIT] UMAP-reduced states (dim={target_dim}): {reduced_dir}")
        return reduced_dir

    print(f"[Cache MISS] UMAP-reducing all layers -> {reduced_dir}")
    reduced_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((cache_path / "metadata.json").read_text())
    num_layers = meta["num_layers"]

    # --- Per-layer UMAP reduction ---
    all_reduced_layers = []
    for l in range(num_layers + 1):
        print(f"  Layer {l}/{num_layers}:")
        hidden_states = torch.load(
            cache_path / f"hidden_states_layer_{l}.pt", weights_only=True
        )
        reduced_seqs, umap_model = fit_umap_and_reduce_layer(
            hidden_states,
            target_dim,
            umap_n_neighbors,
            umap_min_dist,
            umap_metric,
        )
        all_reduced_layers.append(reduced_seqs)
        # Save the UMAP model (useful for projecting new data later)
        with open(reduced_dir / f"umap_model_layer_{l}.pkl", "wb") as f:
            pickle.dump(umap_model, f)
        del hidden_states, umap_model

    # --- Chain Procrustes alignment ---
    print("\n  Chain Procrustes alignment (layer l -> layer l-1):")
    aligned_layers, alignment_params = chain_procrustes_align(all_reduced_layers)
    del all_reduced_layers

    # Save aligned reduced states
    for l in range(num_layers + 1):
        torch.save(aligned_layers[l], reduced_dir / f"reduced_layer_{l}.pt")
    # Save alignment params for reproducibility
    torch.save(alignment_params, reduced_dir / "procrustes_params.pt")
    print(f"  Saved {len(alignment_params)} Procrustes alignment transforms")

    # Reduce embedding matrix
    print("  Embedding matrix:")
    reduced_emb, umap_emb = reduce_embedding_matrix(
        teacher_model_name,
        target_dim,
        umap_n_neighbors,
        umap_min_dist,
        umap_metric,
    )

    # Align embeddings to layer 0's coordinate frame using the layer 0 Procrustes
    # (embeddings are the "pre-layer-0" representation, so align them into the same
    # space as the aligned layer 0 for consistency)
    torch.save(reduced_emb, reduced_dir / "reduced_embeddings.pt")
    with open(reduced_dir / "umap_model_embeddings.pkl", "wb") as f:
        pickle.dump(umap_emb, f)

    # Mark done
    done_marker.write_text("done")
    print(f"UMAP reduction + Procrustes alignment complete: {reduced_dir}")
    return reduced_dir


# ---------------------------------------------------------------------------
# Stage 3: Layer-wise training (using full student forward pass)
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """Dataset of (input_ids_placeholder, target_hidden_states_per_layer) pairs.

    Each sample provides:
        - h_0: the UMAP-reduced embedding output [seq_len, dim] (fed as inputs_embeds)
        - targets: list of L tensors [seq_len, dim], one per layer output
    """

    def __init__(
        self,
        h0_sequences: list[torch.Tensor],
        target_sequences: list[list[torch.Tensor]],
    ):
        """
        Args:
            h0_sequences: list of [seq_len_i, dim] tensors (UMAP-reduced layer 0 = embedding output)
            target_sequences: list of num_samples entries, each a list of L tensors [seq_len_i, dim]
        """
        self.h0 = h0_sequences
        self.targets = target_sequences
        self.num_layers = len(target_sequences[0]) if target_sequences else 0

    def __len__(self):
        return len(self.h0)

    def __getitem__(self, idx):
        return self.h0[idx], self.targets[idx]


def collate_fn(batch):
    """Collate variable-length sequences with padding."""
    h0_list, targets_list = zip(*batch)
    batch_size = len(h0_list)
    num_layers = len(targets_list[0])

    lengths = torch.tensor([h.shape[0] for h in h0_list])

    # Pad h0 (inputs_embeds)
    h0_padded = pad_sequence(h0_list, batch_first=True)

    # Pad targets per layer
    targets_padded = []
    for l in range(num_layers):
        layer_seqs = [targets_list[i][l] for i in range(batch_size)]
        targets_padded.append(pad_sequence(layer_seqs, batch_first=True))

    max_len = h0_padded.shape[1]
    attention_mask = torch.arange(max_len).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)

    return h0_padded, targets_padded, attention_mask


def train_all_layers(
    student_model,
    reduced_dir: Path,
    num_layers: int,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    cosine_loss_weight: float = 0.1,
    max_grad_norm: float = 1.0,
    warmup_fraction: float = 0.1,
    device: str = "cuda",
    train_split: float = 0.9,
):
    """Train all student layers jointly using full forward passes.

    The student receives UMAP-reduced h_0 (embedding output) as inputs_embeds,
    runs through all decoder layers, and the loss is the sum of per-layer MSE
    between the student's hidden states and UMAP-reduced teacher hidden states.

    Because the forward pass is sequential, gradients flow through the full stack:
    layer N+1 learns to work with layer N's actual output, not the teacher's.

    Loss is a hybrid of MSE and cosine similarity:
        L = MSE + cosine_loss_weight * (1 - cos_sim)
    """
    print("\n" + "=" * 60)
    print("Layer-wise Initialization Training")
    print("=" * 60)

    # Load all reduced states
    print("Loading UMAP-reduced hidden states...")
    reduced_states = []
    for l in range(num_layers + 1):
        states = torch.load(reduced_dir / f"reduced_layer_{l}.pt", weights_only=True)
        reduced_states.append(states)
        if l == 0:
            total_tokens = sum(s.shape[0] for s in states)
            print(f"  {len(states)} sequences, {total_tokens} total tokens")

    # Reorganize: h0_sequences and per-sample target lists
    n_total = len(reduced_states[0])
    h0_all = reduced_states[0]
    targets_all = []
    for i in range(n_total):
        targets_all.append([reduced_states[l + 1][i] for l in range(num_layers)])

    # Train/test split
    n_train = int(train_split * n_total)

    train_loader = DataLoader(
        SequenceDataset(h0_all[:n_train], targets_all[:n_train]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SequenceDataset(h0_all[n_train:], targets_all[n_train:]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"  Train: {n_train}, Test: {n_total - n_train}")

    # We train all decoder layers + layer norms (but not embed_tokens / lm_head)
    trainable_params = list(student_model.model.layers.parameters())
    # Also include rotary_emb if it has learnable params (usually it doesn't, but safe)
    trainable_params += [
        p for p in student_model.model.rotary_emb.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params, lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler: linear warmup + cosine decay
    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_fraction * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    total_params = sum(p.numel() for p in trainable_params)
    print(f"  Total trainable params: {total_params:,}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(
        f"  Warmup steps: {warmup_steps}/{total_steps}, Cosine loss weight: {cosine_loss_weight}"
    )
    print(f"  Max grad norm: {max_grad_norm}")

    student_model.train()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0

        for h0_padded, targets_padded, mask in train_loader:
            h0_padded = h0_padded.to(device)
            mask = mask.to(device)

            # Full forward pass: feed h0 as inputs_embeds, get all hidden states
            outputs = student_model(
                inputs_embeds=h0_padded,
                attention_mask=mask,
                output_hidden_states=True,
            )

            # Compute per-layer MSE + cosine loss
            batch_loss = torch.tensor(0.0, device=device)
            for l in range(num_layers):
                pred = outputs.hidden_states[l + 1]  # output of layer l
                tgt = targets_padded[l].to(device)

                # MSE component
                mse_per_token = ((pred - tgt) ** 2).mean(dim=-1)
                layer_mse = (mse_per_token * mask.float()).sum() / mask.sum()

                # Cosine similarity component: (1 - cos_sim)
                cos_per_token = nn.functional.cosine_similarity(pred, tgt, dim=-1)
                layer_cosine = ((1.0 - cos_per_token) * mask.float()).sum() / mask.sum()

                batch_loss = batch_loss + layer_mse + cosine_loss_weight * layer_cosine

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += batch_loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / total_tokens
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.6f}, LR: {current_lr:.2e}"
        )

    # Evaluate
    eval_loss, eval_cosine = _evaluate_all_layers(
        student_model, test_loader, num_layers, device
    )
    print(f"  Test MSE: {eval_loss:.6f}, Test Cosine: {eval_cosine:.4f}")

    return student_model


def _evaluate_all_layers(student_model, loader, num_layers, device):
    """Evaluate MSE and cosine similarity across all layers."""
    student_model.eval()
    total_mse, total_cosine, total_tokens = 0.0, 0.0, 0

    with torch.no_grad():
        for h0_padded, targets_padded, mask in loader:
            h0_padded = h0_padded.to(device)
            mask = mask.to(device)

            outputs = student_model(
                inputs_embeds=h0_padded,
                attention_mask=mask,
                output_hidden_states=True,
            )

            for l in range(num_layers):
                pred = outputs.hidden_states[l + 1]
                tgt = targets_padded[l].to(device)

                mse_per_pos = ((pred - tgt) ** 2).mean(dim=-1)
                cos_per_pos = nn.functional.cosine_similarity(pred, tgt, dim=-1)

                total_mse += (mse_per_pos * mask.float()).sum().item()
                total_cosine += (cos_per_pos * mask.float()).sum().item()
                total_tokens += mask.sum().item()

    student_model.train()
    return total_mse / total_tokens, total_cosine / total_tokens


# ---------------------------------------------------------------------------
# Stage 4: Non-layer initialization
# ---------------------------------------------------------------------------


def initialize_non_layer_components(
    student_model, reduced_dir: Path, device: str = "cuda"
):
    """Initialize embeddings from UMAP-reduced teacher embeddings."""
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

    print("  Embeddings initialized from UMAP-reduced teacher embeddings")
    print(f"  Final RMSNorm: left at default (ones)")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _umap_student_cache_key(
    teacher_model_name: str,
    hidden_size: int,
    scale_factor: float,
    num_samples: int,
    max_length: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    init_epochs: int,
    init_batch_size: int,
    init_lr: float,
    init_weight_decay: float,
    init_cosine_loss_weight: float,
    init_max_grad_norm: float,
    init_warmup_fraction: float,
    dataset_name: str = "wikitext",
) -> str:
    """Build a deterministic cache key for an UMAP-initialized student model."""
    raw = (
        f"umap|{teacher_model_name}|{hidden_size}|{scale_factor}|"
        f"{num_samples}|{max_length}|"
        f"{umap_n_neighbors}|{umap_min_dist}|{umap_metric}|"
        f"{init_epochs}|{init_batch_size}|"
        f"{init_lr}|{init_weight_decay}|{init_cosine_loss_weight}|"
        f"{init_max_grad_norm}|{init_warmup_fraction}|{dataset_name}"
    )
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:10]
    safe_name = teacher_model_name.replace("/", "_").replace("\\", "_")
    return f"umap_student_{safe_name}_h{hidden_size}_{short_hash}"


def umap_layerwise_init(
    teacher_model_name: str,
    hidden_size: Optional[int],
    scale_factor: float,
    cache_dir: str,
    num_samples: int = 5000,
    max_length: int = 128,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    init_epochs: int = 10,
    init_batch_size: int = 16,
    init_lr: float = 1e-4,
    init_weight_decay: float = 0.01,
    init_cosine_loss_weight: float = 0.1,
    init_max_grad_norm: float = 1.0,
    init_warmup_fraction: float = 0.1,
    dataset_name: str = "wikitext",
    dataset_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Full UMAP layer-wise initialization pipeline.

    Returns:
        (student_model, student_config)
    """
    # Import here to avoid circular import
    from src.distillation.student_factory import create_student_model
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("UMAP Layer-wise Initialization")
    print("=" * 60)

    dataset_path = str(dataset_path or dataset_name)

    # Determine student hidden size early (needed for cache key)
    if hidden_size is None:
        teacher_config = AutoConfig.from_pretrained(teacher_model_name)
        hidden_size = max(64, int(teacher_config.hidden_size * scale_factor))
        hidden_size = ((hidden_size + 63) // 64) * 64

    # Check for cached initialized student
    student_key = _umap_student_cache_key(
        teacher_model_name,
        hidden_size,
        scale_factor,
        num_samples,
        max_length,
        umap_n_neighbors,
        umap_min_dist,
        umap_metric,
        init_epochs,
        init_batch_size,
        init_lr,
        init_weight_decay,
        init_cosine_loss_weight,
        init_max_grad_norm,
        init_warmup_fraction,
    )
    student_cache_path = Path(cache_dir) / student_key
    student_marker = student_cache_path / "done.marker"

    if student_marker.exists():
        print(f"[Cache HIT] Initialized student model: {student_cache_path}")
        student_config = AutoConfig.from_pretrained(str(student_cache_path))
        student_model = _AutoModelForCausalLM.from_pretrained(
            str(student_cache_path),
            torch_dtype=torch.float32,
        ).to(device)
        return student_model, student_config

    print(f"[Cache MISS] Will save initialized student to: {student_cache_path}")

    # Step 1: Cache teacher hidden states
    print("\n--- Stage 1: Extract teacher hidden states ---")
    hs_cache = cache_teacher_hidden_states(
        teacher_model_name,
        num_samples,
        max_length,
        cache_dir,
        device,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
    )

    # Step 2: UMAP reduce all layers + embeddings
    print("\n--- Stage 2: UMAP reduction ---")
    reduced_dir = umap_reduce_all_layers(
        hs_cache,
        hidden_size,
        teacher_model_name,
        umap_n_neighbors,
        umap_min_dist,
        umap_metric,
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

    # Step 4: Train all layers jointly
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
    print("UMAP Layer-wise Initialization Complete")
    print("=" * 60)

    return student_model, student_config
