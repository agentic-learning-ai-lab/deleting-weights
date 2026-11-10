"""DistilTrainer that sends hard-masked weights to vLLM, so rollouts sample from the deployed (masked) student."""

import torch
from accelerate import Accelerator
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from distil_trainer import DistilTrainer, MemoryEfficientSyncRefModelCallback
from learnable_sparsity import sparse_linear


class TeacherLogitSyncCallback(TrainerCallback):
    """Move the teacher's mask logits toward the student's: teacher = alpha * student + (1 - alpha) * teacher."""

    def __init__(self, teacher: torch.nn.Module, accelerator: Accelerator) -> None:
        """Keep the teacher and the accelerator that wraps the student."""
        self.teacher = teacher
        self.accelerator = accelerator

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                    **kwargs) -> None:
        """Average the logits every ref_model_sync_steps steps."""
        if state.global_step % args.ref_model_sync_steps != 0:
            return
        student = dict(self.accelerator.unwrap_model(kwargs["model"]).named_parameters())
        alpha = args.ref_model_mixup_alpha
        for name, teacher_param in self.teacher.named_parameters():
            if name.endswith("diff_mask.gate"):
                teacher_param.data.mul_(1.0 - alpha).add_(student[name].data, alpha=alpha)


class MaskAwareDistilTrainer(DistilTrainer):
    """DistilTrainer for Deletion Tuning on one GPU.

    vLLM receives `weight * hard_mask` for every SparseLinear, both models build their masked weights once per optimizer
    step, and the teacher's moving average covers only its mask logits.
    """

    _step_weights_step = None  # optimizer step the current step weights were built for

    def __init__(self, *args, **kwargs):
        """Build the trainer, then restrict the teacher's moving average to the mask logits."""
        super().__init__(*args, **kwargs)
        sync = [cb for cb in self.callback_handler.callbacks if isinstance(cb, MemoryEfficientSyncRefModelCallback)]
        assert len(sync) == 1, "expected the trainer's teacher moving-average callback"
        self.callback_handler.remove_callback(sync[0])
        self.add_callback(TeacherLogitSyncCallback(teacher=self.ref_model, accelerator=self.accelerator))

    def training_step(self, model: torch.nn.Module, inputs: dict,
                      num_items_in_batch: int | None = None) -> torch.Tensor:
        """Run one micro-batch, rebuilding both models' masked weights once per optimizer step."""
        if self._step_weights_step != self.state.global_step:
            for net in (self.accelerator.unwrap_model(self.model), self.ref_model):
                sparse_linear.drop_step_weights(model=net)
                sparse_linear.build_step_weights(model=net)
            self._step_weights_step = self.state.global_step
        return super().training_step(model, inputs, num_items_in_batch)

    @staticmethod
    def _masked_weight(mod: sparse_linear.SparseLinear) -> torch.Tensor:
        """Return ``weight * hard_mask`` of one SparseLinear (the stored weight is not modified)."""
        mask = sparse_linear.local_hard_mask(module=mod).view(mod.weight.shape)
        return mod.weight.data * mask.to(mod.weight.dtype)

    def _move_model_to_vllm(self) -> None:
        """Load the student into the colocated vLLM engine, with SparseLinear weights hard-masked and logits skipped."""
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and getattr(deepspeed_plugin, "zero_stage", None) == 3
        assert not (self.is_fsdp_enabled or zero_stage_3), "mask-aware vLLM sync supports a single unsharded model"
        torch.cuda.empty_cache()

        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        sparse = {f"{name}.weight": mod for name, mod in self.model.named_modules()
                  if isinstance(mod, sparse_linear.SparseLinear)}
        for name, param in self.model.named_parameters():
            if name.endswith("diff_mask.gate"):
                continue
            # Built right before it is sent, so at most one extra weight is held at a time.
            data = self._masked_weight(sparse[name]) if name in sparse else param.data
            llm_model.load_weights([(self._fix_param_name_to_vllm(name), data)])
        self.llm.reset_prefix_cache()
