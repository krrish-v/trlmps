# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from transformers import TrainingArguments

from .base_config import BaseConfig


@dataclass
class SFTConfig(BaseConfig):
    r"""
    Configuration class for the [`SFTTrainer`].

    This class includes only the parameters that are specific to SFT training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`SFTTrainer`] is provided as a string. If you're training a MoE architecture and want to
            include the load balancing/auxiliary loss as a part of the final loss, remember to set
            `output_router_logits=True` in this dictionary.
        chat_template_path (`str`, *optional*):
            If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory
            or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must
            ensure that any special tokens referenced in the template are added to the tokenizer and that the model's
            embedding layer is resized accordingly.

        > Parameters that control the data preprocessing

        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column that contains text data in the dataset.
        dataset_kwargs (`dict[str, Any]`, *optional*):
            Dictionary of optional keyword arguments for the dataset preparation. The only supported key is
            `skip_prepare_dataset`. When the model is a VLM, `skip_prepare_dataset` is automatically treated as `True`
            regardless of the provided value, since preprocessing is done on the fly.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        eos_token (`str`, *optional*):
            Token used to indicate the end of a turn or sequence. If `None`, it defaults to
            `processing_class.eos_token`.
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right.
            If `None`, no truncation is applied. When packing is enabled, this value sets the sequence length.
        shuffle_dataset (`bool`, *optional*, defaults to `False`):
            Whether to shuffle the dataset.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce
            padding. Uses `max_length` to define sequence length.
        packing_strategy (`str`, *optional*, defaults to `"bfd"`):
            Strategy for packing sequences. Can be `"bfd"` (best-fit decreasing, truncates overflow), `"bfd-requeue"`
            (best-fit decreasing, re-queues overflow tokens), or `"wrapped"` (aggressive, cuts mid-sequence).
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure. When
            packing is enabled with strategy `"bfd"`, padding-free is enabled, regardless of the value of this
            parameter.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        eval_packing (`bool`, *optional*):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.

        > Parameters that control the training

        completion_only_loss (`bool`, *optional*):
            Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed
            only on the completion, which is supported only for [prompt-completion](#prompt-completion) datasets. If
            `False`, loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset:
            loss is computed on the completion for [prompt-completion](#prompt-completion) datasets, and on the full
            sequence for [language modeling](#language-modeling) datasets.
        assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only
            on the assistant responses, which is supported only for [conversational](#conversational) datasets. If
            `False`, loss is computed on the entire sequence.
        loss_type (`str`, *optional*, defaults to `"nll"`):
            Type of loss to use. Possible values are `"nll"` (negative log-likelihood, default) and `"dft"` (Dynamic
            Fine-Tuning, as described in [this paper](https://huggingface.co/papers/2508.05629)).
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string. If you're training a MoE architecture and want to include the "
            "load balancing/auxiliary loss as a part of the final loss, remember to set `output_router_logits=True` "
            "in this dictionary."
        },
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": "If specified, sets the model's chat template. This can either be the path to a tokenizer (local "
            "directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, "
            "you must ensure that any special tokens referenced in the template are added to the tokenizer and "
            "that the model's embedding layer is resized accordingly."
        },
    )

    # Parameters that control the data preprocessing
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the column that contains text data in the dataset."},
    )
    dataset_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments for the dataset preparation. The only supported key is "
            "`skip_prepare_dataset`. If the model is a VLM, `skip_prepare_dataset` value is ignored. When the model "
            "is a VLM, `skip_prepare_dataset` is automatically treated as `True` regardless of the provided value, "
            "since preprocessing is done on the fly."
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    eos_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`."
        },
    )
    pad_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from "
            "the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    shuffle_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the dataset."},
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to group multiple sequences into fixed-length blocks to improve computational efficiency "
            "and reduce padding. Uses `max_length` to define sequence length."
        },
    )
    packing_strategy: str = field(
        default="bfd",
        metadata={
            "help": "Strategy for packing sequences. Can be `'bfd'` (best-fit decreasing, truncates overflow), "
            "`'bfd-requeue'` (best-fit decreasing, re-queues overflow tokens), or `'wrapped'` (aggressive, cuts "
            "mid-sequence).",
            "choices": ["bfd", "bfd-requeue", "wrapped"],
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this "
            "is only supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch "
            "structure. When packing is enabled with strategy `'bfd'`, padding-free is enabled, regardless of the "
            "value of this parameter."
        },
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )
    eval_packing: bool | None = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."},
    )

    # Parameters that control the training
    completion_only_loss: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is "
                "computed only on the completion, which is supported only for prompt-completion datasets. If `False`, "
                "loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset: "
                "loss is computed on the completion for prompt-completion datasets, and on the full sequence for "
                "language modeling datasets."
            )
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is "
                "computed only on the assistant responses, which is supported only for conversational datasets. If `False`, "
                "loss is computed on the entire sequence."
            )
        },
    )
    loss_type: str = field(
        default="nll",
        metadata={
            "help": (
                'Type of loss to use. Possible values are `"nll"` (negative log-likelihood, default) and `"dft"` '
                "(Dynamic Fine-Tuning, as described in https://huggingface.co/papers/2508.05629)."
            )
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )

    # Parameters that control Apple Silicon (MPS) optimization
    use_mps_optimization: bool = field(
        default=False,
        metadata={
            "help": "Enable Apple Silicon MPS optimizations: sync barriers to prevent async tensor pile-up, "
            "shape bucketing to reduce Metal graph recompilation, and fused loss to avoid full logit "
            "materialization. No-op on non-MPS systems."
        },
    )
    mps_memory_fraction: float = field(
        default=0.9,
        metadata={
            "help": "Fraction of total system memory available to MPS (0.0 to 1.0). Maps to "
            "`torch.mps.set_per_process_memory_fraction()`. On M4 Pro with 48GB, 0.9 allows ~43GB "
            "for model + activations, leaving ~5GB for macOS."
        },
    )
    mps_fused_loss_chunk_size: int = field(
        default=65536,
        metadata={
            "help": "Vocabulary chunk size for the fused linear + cross-entropy loss. Smaller values use less "
            "memory but require more compute passes. 65536 reduces Python MPS dispatch latency to prioritize speed while still bounding RAM."
        },
    )
    use_metal_liger: bool = field(
        default=False,
        metadata={
            "help": "Enable MetalLiger fused Metal kernels for Apple Silicon. Replaces RMSNorm and SwiGLU "
            "with fused operations that eliminate intermediate tensor allocations, reducing memory by ~66% "
            "per op and cutting Metal dispatches by ~50%. Requires use_mps_optimization=True."
        },
    )
    use_metal_liger_compile: bool = field(
        default=False,
        metadata={
            "help": "Enable torch.compile graph capture for MetalLiger (Phase 4a). Uses aot_eager backend "
            "to trace the model forward into a single Metal command buffer burst, eliminating ~95 "
            "Python→C++ crossings per step. Requires use_metal_liger=True. "
            "Note: First step is slow (graph tracing). Steps 2+ are faster."
        },
    )
    mps_cleanup_frequency: int = field(
        default=10,
        metadata={
            "help": "How often to check MPS memory pressure (every N steps). "
            "Background gc.collect still runs every step regardless. "
            "Default raised to 10 from 1 — synchronize() now only fires under memory pressure "
            "(>85%% of peak) or at deep-clean intervals, not every step."
        },
    )
    mps_group_by_length: bool = field(
        default=False,
        metadata={
            "help": "When MPS optimization is enabled AND batch_size > 1, group training samples by sequence "
            "length to minimize padding waste. With per_device_train_batch_size=1 (typical on Mac), this adds "
            "CPU sorting overhead with zero benefit. Only enable for batch_size >= 2."
        },
    )
    mps_eval_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of dataloader workers during evaluation on MPS. During eval, the GPU has no backward "
            "pass and runs ~25x faster than training. Background workers add IPC serialization overhead that "
            "starves the GPU. Default 0 uses the main thread for direct feeding."
        },
    )
    mps_prefetch_factor: int = field(
        default=2,
        metadata={
            "help": "Prefetch factor for MPS DataLoader. mlx-data uses exactly 2 in-flight buffers (double-buffered "
            "prefetch) to keep the GPU fed without accumulating idle batches in RAM. Default 2 mirrors this pattern. "
            "Higher values waste RAM on 16GB Macs. Only applies when num_workers > 0."
        },
    )
    mps_max_tokens_per_batch: int | None = field(
        default=None,
        metadata={
            "help": "Maximum total tokens per batch for MPS dynamic batching (mlx-data DynamicBatch pattern). "
            "When set, the VLM data collator packs variable numbers of samples per batch so that the total "
            "token count stays under this budget. Shorter sequences get larger batches; longer sequences get "
            "smaller batches. This prevents RAM spikes from batches with many large images. "
            "Only applies to VLM datasets. Example: 4096 for 16GB Mac, 8192 for 48GB Mac."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.use_mps_optimization:
            # Sorted-by-length batching: group similar-length sequences to minimize
            # padding waste (MLX-LM sorts ALL sequences by length).
            if self.mps_group_by_length:
                self.group_by_length = True
                logger.info("MPS: auto-enabled group_by_length for sorted-by-length batching")

            # Disable NaN filtering: logging_nan_inf_filter=True forces .item() on EVERY
            # step (GPU→CPU sync). MLX-LM does zero metrics during training.
            if self.logging_nan_inf_filter:
                self.logging_nan_inf_filter = False
                logger.info("MPS: auto-disabled logging_nan_inf_filter to avoid per-step .item() sync")

            # Sequence length bucketing: Metal compiles a UNIQUE graph for each tensor shape.
            # With batch_size=1, every step has a different (1, seq_len) shape → each creates
            # a new Metal graph (~10-20MB) cached in wired memory FOREVER.
            # Over 1000 steps: 1000 graphs × 15MB = ~15GB wired memory growth → swap.
            # Padding to multiples of 64 caps unique graphs to max_length/64 ≈ 16.
            if self.pad_to_multiple_of is None:
                self.pad_to_multiple_of = 64
                logger.info("MPS: auto-set pad_to_multiple_of=64 to limit Metal graph recompilation")

            # Deep prefetch buffer: with num_workers=4 and prefetch_factor=4,
            # the DataLoader keeps 16 batches ready in its queue. Even if one
            # worker hits a slow JPEG decode, the GPU never starves.
            # Only valid when num_workers > 0 (PyTorch rejects prefetch with 0 workers).
            if self.dataloader_prefetch_factor is None and self.dataloader_num_workers > 0:
                self.dataloader_prefetch_factor = self.mps_prefetch_factor
                logger.info(
                    f"MPS: auto-set dataloader_prefetch_factor={self.mps_prefetch_factor} "
                    f"(mlx-data double-buffer pattern — max {self.mps_prefetch_factor} batches in flight)"
                )

            # Phase 2 hint: gradient checkpointing trades RAM for compute.
            # Once memory is stable (~22GB flat), disabling it eliminates ~30% recomputation.
            if self.gradient_checkpointing:
                logger.info(
                    "MPS: gradient_checkpointing=True trades RAM for compute. "
                    "If memory is stable at ~22GB, set gradient_checkpointing=False "
                    "to eliminate ~30%% recomputation overhead."
                )
