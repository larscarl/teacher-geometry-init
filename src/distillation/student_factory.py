from transformers import AutoModelForCausalLM, AutoConfig
import torch
from typing import Optional, Literal
import copy


def _resolve_attn_impl(model_name: str) -> str:
    return "eager" if "gemma-3" in model_name.lower() else "sdpa"


def _set_config_attn_impl(config: AutoConfig, attn_implementation: str) -> None:
    # Transformers models typically read one of these config attributes.
    config.attn_implementation = attn_implementation
    config._attn_implementation = attn_implementation


def create_student_model(
    teacher_model_name: str,
    scale_factor: float = 0.25,
    init_strategy: Literal[
        "random", "copy_subset", "umap_layerwise", "pca_layerwise"
    ] = "random",
    hidden_size: Optional[int] = None,
    device: str = "cuda",
    student_init_cfg: Optional[dict] = None,
    attn_implementation: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoConfig]:
    """
    Create a smaller student model from a teacher model.

    Args:
        teacher_model_name: HuggingFace model name or path
        scale_factor: How much to scale down (0.25 = 1/4 size). Ignored if explicit dims provided.
        init_strategy: "random", "copy_subset", "umap_layerwise", or "pca_layerwise"
        hidden_size: Explicit hidden size (overrides scale_factor)
        device: Device to load models on
        student_init_cfg: Shared config dict for layerwise init strategies
        attn_implementation: Optional attention backend override ("eager" or "sdpa")

    Returns:
        student_model: The smaller model
        student_config: Its configuration
    """

    # Load teacher config
    teacher_config = AutoConfig.from_pretrained(teacher_model_name)

    # Create student config by copying teacher
    student_config = copy.deepcopy(teacher_config)
    resolved_attn_impl = attn_implementation or _resolve_attn_impl(teacher_model_name)
    _set_config_attn_impl(student_config, resolved_attn_impl)

    # Determine scaling
    num_layers = teacher_config.num_hidden_layers

    if hidden_size is None:
        hidden_size = max(64, int(teacher_config.hidden_size * scale_factor))
        # Round to nearest multiple of 64 for efficiency
        hidden_size = ((hidden_size + 63) // 64) * 64

    # Update student config
    student_config.num_hidden_layers = num_layers
    student_config.hidden_size = hidden_size

    # Scale attention heads proportionally
    # num_heads must divide hidden_size evenly
    original_heads = teacher_config.num_attention_heads
    head_dim = teacher_config.hidden_size // original_heads
    student_config.num_attention_heads = max(1, hidden_size // head_dim)

    # Handle key_value_heads for GQA models (like Llama)
    if hasattr(teacher_config, "num_key_value_heads"):
        kv_scale = student_config.num_attention_heads / original_heads
        student_config.num_key_value_heads = max(
            1, int(teacher_config.num_key_value_heads * kv_scale)
        )

    # Scale intermediate size (FFN) proportionally
    if hasattr(teacher_config, "intermediate_size"):
        teacher_ratio = teacher_config.intermediate_size / teacher_config.hidden_size
        student_config.intermediate_size = int(hidden_size * teacher_ratio)

    print(
        f"Teacher config: {teacher_config.num_hidden_layers} layers, "
        f"{teacher_config.hidden_size} hidden, {teacher_config.num_attention_heads} heads"
    )
    print(
        f"Student config: {student_config.num_hidden_layers} layers, "
        f"{student_config.hidden_size} hidden, {student_config.num_attention_heads} heads"
    )
    print(f"Student attention implementation: {resolved_attn_impl}")

    # Create student model
    if init_strategy == "random":
        # Random initialization
        student_model = AutoModelForCausalLM.from_config(
            student_config, dtype=torch.float32
        )
        print("Initialized student with random weights")

    elif init_strategy == "copy_subset":
        # Copy subset of teacher weights
        print("Loading teacher to copy weights...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float32,
            device_map=device,
            attn_implementation=resolved_attn_impl,
        )

        # Initialize student with random weights first
        student_model = AutoModelForCausalLM.from_config(
            student_config, dtype=torch.float32
        )

        # Copy weights strategically
        _copy_teacher_weights(
            teacher_model,
            student_model,
            num_layers,
            hidden_size,
            teacher_config.num_hidden_layers,
            teacher_config.hidden_size,
        )
        print("Copied subset of teacher weights to student")

        # Clean up teacher
        del teacher_model
        torch.cuda.empty_cache()

    elif init_strategy == "umap_layerwise":
        from src.distillation.umap_layerwise_init import umap_layerwise_init

        if student_init_cfg is None:
            raise ValueError(
                "student_init_cfg is required for umap_layerwise init strategy"
            )

        return umap_layerwise_init(
            teacher_model_name=teacher_model_name,
            hidden_size=hidden_size,
            scale_factor=scale_factor,
            cache_dir=student_init_cfg["cache_dir"],
            num_samples=student_init_cfg.get("num_samples", 5000),
            max_length=student_init_cfg.get("max_length", 128),
            umap_n_neighbors=student_init_cfg.get("umap_n_neighbors", 15),
            umap_min_dist=student_init_cfg.get("umap_min_dist", 0.1),
            umap_metric=student_init_cfg.get("umap_metric", "cosine"),
            init_epochs=student_init_cfg.get("init_epochs", 10),
            init_batch_size=student_init_cfg.get("init_batch_size", 16),
            init_lr=student_init_cfg.get("init_lr", 1e-4),
            init_weight_decay=student_init_cfg.get("init_weight_decay", 0.01),
            dataset_name=student_init_cfg.get("dataset_name", "wikitext"),
            dataset_path=student_init_cfg.get("dataset_path"),
            dataset_hf_path=student_init_cfg.get("dataset_hf_path"),
            dataset_hf_config=student_init_cfg.get("dataset_hf_config"),
            dataset_hf_split=student_init_cfg.get("dataset_hf_split", "train"),
            dataset_streaming=student_init_cfg.get("dataset_streaming", False),
            device=device,
        )

    elif init_strategy == "pca_layerwise":
        from src.distillation.pca_layerwise_init import pca_layerwise_init

        if student_init_cfg is None:
            raise ValueError(
                "student_init_cfg is required for pca_layerwise init strategy"
            )

        return pca_layerwise_init(
            teacher_model_name=teacher_model_name,
            hidden_size=hidden_size,
            scale_factor=scale_factor,
            cache_dir=student_init_cfg["cache_dir"],
            num_samples=student_init_cfg.get("num_samples", 5000),
            max_length=student_init_cfg.get("max_length", 128),
            init_epochs=student_init_cfg.get("init_epochs", 10),
            init_batch_size=student_init_cfg.get("init_batch_size", 16),
            init_lr=student_init_cfg.get("init_lr", 1e-4),
            init_weight_decay=student_init_cfg.get("init_weight_decay", 0.01),
            init_cosine_loss_weight=student_init_cfg.get("init_cosine_loss_weight", 0.1),
            init_max_grad_norm=student_init_cfg.get("init_max_grad_norm", 1.0),
            init_warmup_fraction=student_init_cfg.get("init_warmup_fraction", 0.1),
            dataset_name=student_init_cfg.get("dataset_name", "wikitext"),
            dataset_path=student_init_cfg.get("dataset_path"),
            dataset_hf_path=student_init_cfg.get("dataset_hf_path"),
            dataset_hf_config=student_init_cfg.get("dataset_hf_config"),
            dataset_hf_split=student_init_cfg.get("dataset_hf_split", "train"),
            dataset_streaming=student_init_cfg.get("dataset_streaming", False),
            device=device,
        )

    student_model.to(device)

    # Print parameter counts
    teacher_params = sum(
        p.numel()
        for p in AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            device_map="meta",
            attn_implementation=resolved_attn_impl,
        ).parameters()
    )
    student_params = sum(p.numel() for p in student_model.parameters())

    print(f"\nParameter counts:")
    print(f"  Teacher: {teacher_params:,} ({teacher_params / 1e6:.1f}M)")
    print(f"  Student: {student_params:,} ({student_params / 1e6:.1f}M)")
    print(f"  Ratio: {student_params / teacher_params:.2%}")

    return student_model, student_config


def _copy_teacher_weights(
    teacher_model,
    student_model,
    student_layers: int,
    student_hidden: int,
    teacher_layers: int,
    teacher_hidden: int,
):
    """
    Copy a subset of teacher weights to student.
    Strategy:
    - For depth: Select evenly spaced layers
    - For width: Select first N dimensions (preserves low-frequency components)
    """

    # Determine which layers to copy
    if student_layers == teacher_layers:
        layer_indices = list(range(student_layers))
    else:
        # Select evenly spaced layers
        layer_indices = [
            int(i * teacher_layers / student_layers) for i in range(student_layers)
        ]

    print(f"Copying layers {layer_indices} from teacher")

    # Get target dtype from student
    student_dtype = next(student_model.parameters()).dtype
    print(f"Copying weights with dtype: {student_dtype}")

    # Copy embeddings (truncate if needed)
    if hasattr(student_model, "model") and hasattr(teacher_model, "model"):
        # For models like Llama
        teacher_embed = teacher_model.model.embed_tokens.weight.data
        student_model.model.embed_tokens.weight.data = (
            teacher_embed[:, :student_hidden].clone().to(dtype=student_dtype)
        )

        # Copy layer weights
        for student_idx, teacher_idx in enumerate(layer_indices):
            _copy_layer_weights(
                teacher_model.model.layers[teacher_idx],
                student_model.model.layers[student_idx],
                student_hidden,
                teacher_hidden,
            )

        # Copy final norm
        if student_hidden == teacher_hidden:
            student_model.model.norm.weight.data = (
                teacher_model.model.norm.weight.data.clone().to(dtype=student_dtype)
            )
        else:
            student_model.model.norm.weight.data = (
                teacher_model.model.norm.weight.data[:student_hidden]
                .clone()
                .to(dtype=student_dtype)
            )

    # Copy LM head (truncate if needed)
    if hasattr(student_model, "lm_head") and hasattr(teacher_model, "lm_head"):
        teacher_lm = teacher_model.lm_head.weight.data
        student_model.lm_head.weight.data = (
            teacher_lm[:, :student_hidden].clone().to(dtype=student_dtype)
        )


def _copy_layer_weights(teacher_layer, student_layer, student_hidden, teacher_hidden):
    """Copy weights from one transformer layer to another, handling dimension mismatches."""

    # Get student dtype to ensure all copied weights match
    student_dtype = next(student_layer.parameters()).dtype

    # Self-attention
    if hasattr(teacher_layer, "self_attn"):
        for param_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(teacher_layer.self_attn, param_name):
                teacher_param = getattr(teacher_layer.self_attn, param_name).weight.data
                student_param = getattr(student_layer.self_attn, param_name).weight.data

                # Truncate and ensure dtype matches
                if param_name == "o_proj":
                    # o_proj: [hidden, hidden] - truncate both dims
                    student_param[:] = (
                        teacher_param[:student_hidden, :student_hidden]
                        .clone()
                        .to(dtype=student_dtype)
                    )
                else:
                    # q,k,v_proj: [hidden, hidden] - truncate both dims
                    student_param[:] = (
                        teacher_param[: student_param.shape[0], :student_hidden]
                        .clone()
                        .to(dtype=student_dtype)
                    )

    # MLP
    if hasattr(teacher_layer, "mlp"):
        # Up projection (truncate input dim)
        if hasattr(teacher_layer.mlp, "up_proj"):
            teacher_up = teacher_layer.mlp.up_proj.weight.data
            student_up = student_layer.mlp.up_proj.weight.data
            student_up[:] = (
                teacher_up[: student_up.shape[0], :student_hidden]
                .clone()
                .to(dtype=student_dtype)
            )

        # Gate projection (if exists, truncate input dim)
        if hasattr(teacher_layer.mlp, "gate_proj"):
            teacher_gate = teacher_layer.mlp.gate_proj.weight.data
            student_gate = student_layer.mlp.gate_proj.weight.data
            student_gate[:] = (
                teacher_gate[: student_gate.shape[0], :student_hidden]
                .clone()
                .to(dtype=student_dtype)
            )

        # Down projection (truncate output dim)
        if hasattr(teacher_layer.mlp, "down_proj"):
            teacher_down = teacher_layer.mlp.down_proj.weight.data
            student_down = student_layer.mlp.down_proj.weight.data
            student_down[:] = (
                teacher_down[:student_hidden, : student_down.shape[1]]
                .clone()
                .to(dtype=student_dtype)
            )

    # Layer norms
    for norm_name in ["input_layernorm", "post_attention_layernorm"]:
        if hasattr(teacher_layer, norm_name):
            teacher_norm = getattr(teacher_layer, norm_name).weight.data
            student_norm = getattr(student_layer, norm_name).weight.data
            student_norm[:] = (
                teacher_norm[:student_hidden].clone().to(dtype=student_dtype)
            )


# Usage example
if __name__ == "__main__":
    # Example 1: Create small student with both width and depth scaling
    student, config = create_student_model(
        teacher_model_name="meta-llama/Llama-3.2-1B",
        scale_factor=0.1,  # 1/4 the size
        init_strategy="random",
        device="cuda",
    )

    # Example 2: Explicit dimensions
    student, config = create_student_model(
        teacher_model_name="meta-llama/Llama-3.2-1B",
        hidden_size=512,
        init_strategy="copy_subset",
        device="cuda",
    )

    # Example 3: Only scale depth (fewer layers, same width)
    student, config = create_student_model(
        teacher_model_name="microsoft/phi-2",
        scale_factor=0.25,
        init_strategy="random",
        device="cuda",
    )
