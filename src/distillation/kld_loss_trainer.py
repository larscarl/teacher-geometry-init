from transformers import Trainer
import torch
import torch.nn.functional as F
import math
from typing import Optional
from torch import nn


class KLDistillationTrainer(Trainer):
    """
    Simple KL divergence distillation trainer.

    Supports separate batch sizes for teacher and student: the dataloader
    produces batches of size ``per_device_train_batch_size`` (the *student*
    batch size).  The teacher processes the same batch in sub-batches of
    size ``teacher_batch_size`` to reduce peak VRAM usage.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.9,  # Weight for KL loss
        teacher_fp16: bool = False,  # Force teacher to fp16 for speed
        teacher_batch_size: Optional[int] = None,  # Sub-batch size for teacher
        *args,
        **kwargs,
    ):
        """
        Args:
            teacher_model: Frozen teacher model
            temperature: Temperature for softening distributions
            alpha: Weight for KL loss vs CE loss (1.0 = pure distillation)
            teacher_fp16: If True, cast teacher to fp16 for faster inference
            teacher_batch_size: If set, the teacher processes the student
                batch in chunks of this size.  None means same as student.
        """
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.temperature = temperature
        self.alpha = alpha
        self.teacher_fp16 = teacher_fp16
        self.teacher_batch_size = teacher_batch_size

        # Track loss components for logging — accumulate over forward passes
        self._reset_accumulators()

        # Move teacher to same device as student
        self.teacher_model.to(self.model.device)

        # Optionally cast teacher to fp16 for faster inference
        if self.teacher_fp16:
            print("  → Casting teacher to FP16 for faster inference")
            self.teacher_model = self.teacher_model.half()

        print(f"KL Distillation Trainer initialized:")
        print(f"  Temperature: {temperature}")
        print(f"  Alpha (KL weight): {alpha}")
        print(f"  Teacher FP16: {teacher_fp16}")
        print(f"  Teacher batch size: {teacher_batch_size or 'same as student'}")

    def _reset_accumulators(self):
        """Reset all loss accumulators."""
        self.accumulated_kl_loss_scaled = 0.0
        self.accumulated_kl_loss_raw = 0.0
        self.accumulated_ce_loss = 0.0
        self.forward_passes_since_log = 0

    def _teacher_forward(self, inputs):
        """Run teacher inference, optionally in sub-batches to save VRAM."""
        batch_size = inputs["input_ids"].size(0)
        chunk = self.teacher_batch_size

        autocast_enabled = self.teacher_fp16 and torch.cuda.is_available()
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda",
                enabled=autocast_enabled,
                dtype=torch.float16,
            ):
                if chunk is None or chunk >= batch_size:
                    teacher_logits = self.teacher_model(**inputs).logits
                else:
                    logit_chunks = []
                    for start in range(0, batch_size, chunk):
                        end = start + chunk
                        sub_inputs = {k: v[start:end] for k, v in inputs.items()}
                        logit_chunks.append(self.teacher_model(**sub_inputs).logits)
                    teacher_logits = torch.cat(logit_chunks, dim=0)

        return teacher_logits

    @staticmethod
    def _raise_if_non_finite(name: str, value, *, global_step: int, epoch: object) -> None:
        if torch.is_tensor(value):
            is_finite = torch.isfinite(value).all().item()
        else:
            try:
                is_finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                is_finite = False
        if not is_finite:
            raise RuntimeError(
                f"Non-finite `{name}` detected at step={global_step}, epoch={epoch}."
            )

    def _compute_kl_and_ce(
        self, student_logits, teacher_logits, student_ce_loss, attention_mask=None
    ):
        """
        Compute KL divergence (raw and T²-scaled) and return all loss components.

        Returns:
            kl_scaled: KL divergence * T² (for gradient scaling)
            kl_raw: KL divergence without T² scaling (for interpretable logging)
            ce_loss: Student cross-entropy loss
        """
        if attention_mask is not None:
            active_indices = attention_mask.view(-1) == 1
            s_logits = student_logits.view(-1, student_logits.size(-1))[active_indices]
            t_logits = teacher_logits.view(-1, teacher_logits.size(-1))[active_indices]
        else:
            s_logits = student_logits.view(-1, student_logits.size(-1))
            t_logits = teacher_logits.view(-1, teacher_logits.size(-1))

        # Compute KL in float32 for numerical stability, especially with bf16/fp16 training.
        s_logits = s_logits.float()
        t_logits = t_logits.float()
        p_s = F.log_softmax(s_logits / self.temperature, dim=-1)
        p_t_log = F.log_softmax(t_logits / self.temperature, dim=-1)
        kl_raw = F.kl_div(p_s, p_t_log, reduction="batchmean", log_target=True)
        kl_scaled = kl_raw * (self.temperature**2)

        return kl_scaled, kl_raw, student_ce_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mask = inputs.get("attention_mask")
        # Student pass (full batch)
        student_outputs = model(**inputs)

        if model.training:
            student_logits = student_outputs.logits

            # Teacher pass (possibly sub-batched)
            teacher_logits = self._teacher_forward(inputs)

            if teacher_logits.dtype != student_logits.dtype:
                teacher_logits = teacher_logits.to(student_logits.dtype)

            kl_scaled, kl_raw, ce_loss = self._compute_kl_and_ce(
                student_logits, teacher_logits, student_outputs.loss, mask
            )

            current_step = int(getattr(self.state, "global_step", -1))
            current_epoch = getattr(self.state, "epoch", None)
            self._raise_if_non_finite(
                "kl_scaled", kl_scaled, global_step=current_step, epoch=current_epoch
            )
            self._raise_if_non_finite(
                "kl_raw", kl_raw, global_step=current_step, epoch=current_epoch
            )
            self._raise_if_non_finite(
                "ce_loss", ce_loss, global_step=current_step, epoch=current_epoch
            )

            self.accumulated_kl_loss_scaled += kl_scaled.detach()
            self.accumulated_kl_loss_raw += kl_raw.detach()
            self.accumulated_ce_loss += ce_loss.detach()
            self.forward_passes_since_log += 1

            # Training: weighted loss
            loss = self.alpha * kl_scaled + (1.0 - self.alpha) * ce_loss
            self._raise_if_non_finite(
                "train_loss", loss, global_step=current_step, epoch=current_epoch
            )
        else:
            # Eval: return CE only so eval_loss is interpretable as perplexity
            loss = student_outputs.loss

        return (loss, student_outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        """Override to add KL (scaled + raw), CE, and perplexity to training logs."""
        if self.forward_passes_since_log > 0:
            n = self.forward_passes_since_log

            def _to_float(v):
                return v.item() if torch.is_tensor(v) else float(v)

            kl_scaled = _to_float(self.accumulated_kl_loss_scaled) / n
            kl_raw = _to_float(self.accumulated_kl_loss_raw) / n
            ce = _to_float(self.accumulated_ce_loss) / n

            logs["loss_kl"] = round(kl_scaled, 4)
            logs["loss_kl_raw"] = round(kl_raw, 4)
            logs["loss_ce"] = round(ce, 4)
            logs["perplexity"] = round(
                math.exp(min(ce, 20.0)), 4
            )  # clamp to avoid overflow

        self._reset_accumulators()
        super().log(logs, start_time)
