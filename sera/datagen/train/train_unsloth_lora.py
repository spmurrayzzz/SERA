"""
Unsloth LoRA/QLoRA training script optimized for single-node multi-GPU training.

Designed for:
- MoE models (Qwen3-30B-A3B) on 1-2 GPUs with QLoRA
- Large dense models (~100B) on 4-8 GPUs with LoRA/QLoRA

Memory optimizations:
- 4-bit quantization (QLoRA) reduces model memory by ~4x
- 8-bit AdamW optimizer reduces optimizer states from 8 bytes to 2 bytes per param
- Gradient checkpointing reduces activation memory by ~sqrt(layers)
- FlexAttention with block-sparse patterns for long sequences
- LoRA reduces trainable params to <1% of full model
"""

import argparse
import builtins
import gc
import os
import sys
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# from unsloth.chat_templates import train_on_responses_only

# Disable torch.compile - causes issues with MoE + FSDP
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# Set compile location before unsloth import - fixes NoneType error in distributed mode
os.environ["UNSLOTH_COMPILE_LOCATION"] = "/tmp/unsloth_compiled_cache"

# Unsloth's distributed helper assumes a `dist` symbol exists; expose torch.distributed
builtins.dist = torch.distributed

# Torchrun sets TORCHELASTIC_* envs before init_process_group. Unsloth's import-time
# patching checks those and tries to run collectives even before we initialize PG,
# which can deadlock or desync. Strip them temporarily during import, then restore.
_TORCHELASTIC_ENV_KEYS = [
    "TORCHELASTIC_RUN_ID",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
]
_saved_torchelastic_env = {k: os.environ.get(k) for k in _TORCHELASTIC_ENV_KEYS if k in os.environ}
for _k in _TORCHELASTIC_ENV_KEYS:
    os.environ.pop(_k, None)

# Monkey-patch xformers version to bypass Unsloth's outdated version check
# xformers 0.0.30 is required for torch 2.7.0, but Unsloth wants <0.0.27
# This is safe - newer xformers works fine with Unsloth
try:
    import xformers
    xformers.__version__ = "0.0.26"
except ImportError:
    pass

# Set up a shared compile cache path for Unsloth
_COMPILE_LOCATION = str(Path(os.environ["UNSLOTH_COMPILE_LOCATION"]).expanduser())
os.environ["UNSLOTH_COMPILE_LOCATION"] = _COMPILE_LOCATION
os.makedirs(_COMPILE_LOCATION, exist_ok=True)

# Ensure we import the external unsloth package
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_removed_train_dir = False
if _TRAIN_DIR in sys.path:
    sys.path.remove(_TRAIN_DIR)
    _removed_train_dir = True
try:
    from unsloth import FastLanguageModel, FastModel
    from unsloth.chat_templates import train_on_responses_only
finally:
    if _removed_train_dir:
        sys.path.insert(0, _TRAIN_DIR)

# Restore torchrun envs now that Unsloth has been imported
for _k, _v in _saved_torchelastic_env.items():
    if _v is not None:
        os.environ[_k] = _v

def _maybe_init_torch_distributed():
    if torch.distributed.is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        rank=int(os.environ.get("RANK", "0")),
        world_size=world_size,
    )

# Now initialize the process group after Unsloth import and env restoration
_maybe_init_torch_distributed()

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1e9
    free = total - reserved

    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total
    }


def log_memory(context: str):
    """Log GPU memory state."""
    mem = get_gpu_memory_info()
    print(f"[Memory] {context}: {mem['allocated']:.2f}GB allocated, "
          f"{mem['reserved']:.2f}GB reserved, {mem['free']:.2f}GB free / {mem['total']:.2f}GB total")


def enable_flash_sdpa():
    """
    Force Flash / memory-efficient SDPA and disable the math kernel to avoid huge attention buffers.
    Only applied on CUDA. You can opt out by setting ENABLE_FLASH_SDPA=0.

    Note: torch.backends.cuda.sdp_kernel() is a context manager, not a global setter.
    We must use the individual enable_* functions to set global defaults.
    """
    if os.environ.get("ENABLE_FLASH_SDPA", "1") != "1":
        return
    if not torch.cuda.is_available():
        return
    try:
        # Use the global backend setters (not the context manager)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        print("Enabled Flash/memory-efficient SDPA (math SDPA disabled).")
    except Exception as e:
        print(f"Warning: could not set SDPA policy: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def disable_accelerate_output_casting():
    """
    Accelerate wraps model.forward with autocast + a convert_to_fp32 step when bf16/fp16 is enabled.
    With long sequences (32k) this doubles logits memory and can OOM. Allow disabling via env flag.
    """
    if os.environ.get("DISABLE_ACCELERATE_OUTPUT_CAST", "1") != "1":
        return

    try:
        import accelerate.accelerator as accel_acc
        import accelerate.utils.operations as accel_ops

        def _identity(fwd):
            # Preserve __wrapped__ so Accelerate unwrap helpers don't break
            try:
                fwd.__wrapped__ = getattr(fwd, "__wrapped__", fwd)
            except Exception:
                pass
            return fwd

        accel_ops.convert_outputs_to_fp32 = _identity  # type: ignore[attr-defined]
        accel_ops.ConvertOutputsToFp32 = _identity  # type: ignore[attr-defined]
        accel_acc.convert_outputs_to_fp32 = _identity  # type: ignore[attr-defined]
        print("Disabled Accelerate output casting to float32 (DISABLE_ACCELERATE_OUTPUT_CAST=1).")
    except Exception as e:
        print(f"Warning: could not disable Accelerate output casting: {e}")


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """
    Initialize model with LoRA/QLoRA using Unsloth.

    Returns model and tokenizer with LoRA adapters applied.
    """
    model_config = config.get('model', config)  # Support both nested and flat config
    env_max_seq = os.environ.get("MAX_SEQ_LENGTH")
    if env_max_seq:
        try:
            env_max_seq_int = int(env_max_seq)
            model_config['max_seq_length'] = env_max_seq_int
            if 'model' in config:
                config['model']['max_seq_length'] = env_max_seq_int
            else:
                config['max_seq_length'] = env_max_seq_int
            print(f"Overriding max_seq_length via MAX_SEQ_LENGTH={env_max_seq_int}")
        except ValueError:
            print(f"Warning: MAX_SEQ_LENGTH={env_max_seq} is not an int, ignoring.")

    model_name = model_config['model_name']
    max_seq_length = model_config.get('max_seq_length', 32768)
    load_in_4bit = model_config.get('load_in_4bit', True)

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Quantization: {'4-bit (QLoRA)' if load_in_4bit else '16-bit (LoRA)'}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"{'='*60}")

    log_memory("Before model load")

    # Determine model loader based on architecture
    use_fast_model = any(x in model_name.lower() for x in ['qwen3-30b-a3b', 'qwen3-235b-a22b', 'moe'])

    # Common kwargs
    model_kwargs = dict(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,  # Always use LoRA in this script
        # Fast attention via torch SDPA (builtin, no extra install)
        attn_implementation="sdpa",
    )

    # Add dtype if specified
    if model_config.get('dtype') == 'bfloat16':
        model_kwargs['dtype'] = torch.bfloat16

    # Device placement strategy
    device_map_override = os.environ.get("DEVICE_MAP")
    want_model_shard = os.environ.get("MODEL_SHARDING", "0") == "1"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if device_map_override:
        model_kwargs['device_map'] = device_map_override
        print(f"Using device_map from DEVICE_MAP env: {device_map_override}")
    elif want_model_shard and torch.cuda.is_available() and torch.cuda.device_count() > 1 and world_size == 1:
        # Single process, shard layers across local GPUs
        model_kwargs['device_map'] = "balanced"
        print("MODEL_SHARDING=1 detected with single process - using device_map='balanced' to shard across local GPUs.")
    elif torch.cuda.is_available():
        # Default: keep each rank on its local GPU; avoids Unsloth's default sequential sharding onto GPU0
        current_device = torch.cuda.current_device()
        model_kwargs['device_map'] = {"": f"cuda:{current_device}"}

    if use_fast_model:
        print("Using FastModel (MoE architecture detected)")
        model, tokenizer = FastModel.from_pretrained(**model_kwargs)
    else:
        print("Using FastLanguageModel (dense architecture)")
        model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)

    log_memory("After model load, before LoRA")

    # LoRA configuration
    lora_config = config.get('lora', {})
    lora_r = lora_config.get('r', 64)
    lora_alpha = lora_config.get('alpha', 128)
    lora_dropout = lora_config.get('dropout', 0.05)

    # Target modules - comprehensive for MoE
    default_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP/MoE FFN
    ]
    target_modules = lora_config.get('target_modules', default_target_modules)

    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {lora_r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Dropout: {lora_dropout}")
    print(f"  Target modules: {target_modules}")

    # Apply LoRA using Unsloth's optimized method
    if use_fast_model:
        model = FastModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_gradient_checkpointing=True,  # Unsloth's optimized checkpointing
            random_state=config.get('seed', 42),
            use_rslora=lora_config.get('use_rslora', False),
            loftq_config=None,
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_gradient_checkpointing=True,
            random_state=config.get('seed', 42),
            use_rslora=lora_config.get('use_rslora', False),
            loftq_config=None,
        )

    log_memory("After LoRA applied")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")

    # Disable KV cache during training to avoid storing 32k-token keys/values on each layer
    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, tokenizer


def load_training_dataset(config: Dict[str, Any], tokenizer):
    """Load and format dataset with chat template."""
    dataset_config = config['dataset']
    dataset_type = dataset_config.get('type', 'jsonl')
    dataset_path = dataset_config['path']

    print(f"\nLoading dataset from: {dataset_path}")

    if dataset_type in ['json', 'jsonl']:
        dataset = load_dataset(
            "json",
            data_files={"train": dataset_path},
            split=dataset_config.get('split', 'train')
        )
    elif dataset_type == 'huggingface':
        dataset = load_dataset(
            dataset_path,
            split=dataset_config.get('split', 'train')
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Apply chat template if specified
    if dataset_config.get('use_chat_template', True):
        messages_field = dataset_config.get('messages_field', 'messages')

        def format_chat(example):
            formatted = tokenizer.apply_chat_template(
                example[messages_field],
                tokenize=False,
                add_generation_prompt=False
            )
            return {'text': formatted}

        # Avoid shared cache collisions across ranks by keeping the mapped dataset in memory
        dataset = dataset.map(
            format_chat,
            remove_columns=[messages_field],
            load_from_cache_file=False,
            keep_in_memory=True,
        )

    print(f"Dataset size: {len(dataset)} examples")

    # Show first example preview
    if len(dataset) > 0:
        text_field = dataset_config.get('text_field', 'text')
        first_text = dataset[0].get(text_field, str(dataset[0]))
        print(f"\nFirst example preview (first 500 chars):")
        print("-" * 50)
        print(first_text[:500] + "..." if len(first_text) > 500 else first_text)
        print("-" * 50)

    return dataset


def setup_training_config(config: Dict[str, Any]) -> SFTConfig:
    """Create SFTConfig from configuration."""
    training = config['training']

    # Build config dict
    args_dict = {
        'output_dir': training['output_dir'],
        'per_device_train_batch_size': training.get('per_device_train_batch_size', 1),
        'gradient_accumulation_steps': training.get('gradient_accumulation_steps', 8),
        'learning_rate': training.get('learning_rate', 2e-5),
        'logging_steps': training.get('logging_steps', 1),
        'optim': training.get('optim', 'adamw_8bit'),  # 8-bit Adam by default
        'weight_decay': training.get('weight_decay', 0.01),
        'lr_scheduler_type': training.get('lr_scheduler_type', 'cosine'),
        'seed': config.get('seed', 42),
        'save_strategy': training.get('save_strategy', 'epoch'),
        'bf16': training.get('bf16', True),
        'fp16': False,  # Use bf16, not fp16
        # Gradient checkpointing handled by Unsloth
        'gradient_checkpointing': training.get('gradient_checkpointing', False),
        # Dataset field
        'dataset_text_field': config['dataset'].get('text_field', 'text'),
        # Max sequence length
        'max_seq_length': config.get('model', config).get('max_seq_length', 32768),
        # Liger Kernel: Fused linear + cross-entropy avoids materializing full logits tensor
        # For 32k seq × 152k vocab × bf16 = ~9.4GB per sample - this is the OOM culprit
        # Liger chunks the computation to reduce memory by 60-80%
        'use_liger_kernel': training.get('use_liger_kernel', True),
    }

    # Warmup
    if 'warmup_ratio' in training:
        args_dict['warmup_ratio'] = training['warmup_ratio']
    elif 'warmup_steps' in training:
        args_dict['warmup_steps'] = training['warmup_steps']
    else:
        args_dict['warmup_ratio'] = 0.1

    # Training duration
    if 'max_steps' in training:
        args_dict['max_steps'] = training['max_steps']
    if 'num_train_epochs' in training:
        args_dict['num_train_epochs'] = training['num_train_epochs']

    # Save configuration
    if training.get('save_strategy') == 'steps' and 'save_steps' in training:
        args_dict['save_steps'] = training['save_steps']
    if 'save_total_limit' in training:
        args_dict['save_total_limit'] = training['save_total_limit']

    # Multi-GPU settings (DataParallel / DDP)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\nMulti-GPU training detected: {num_gpus} GPUs")
        args_dict['ddp_find_unused_parameters'] = False
        # For LoRA with DDP, we don't need FSDP
        # The model stays on each GPU (fits due to quantization)
        # and gradients are synchronized via DDP

    # WandB
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False):
        args_dict['report_to'] = 'wandb'
        if wandb_config.get('project'):
            os.environ['WANDB_PROJECT'] = wandb_config['project']
        if wandb_config.get('entity'):
            os.environ['WANDB_ENTITY'] = wandb_config['entity']
        if wandb_config.get('name'):
            os.environ['WANDB_NAME'] = wandb_config['name']
    else:
        args_dict['report_to'] = 'none'

    return SFTConfig(**args_dict)


def setup_distributed():
    """
    Initialize distributed training environment.

    When launched with torchrun, each process gets:
    - LOCAL_RANK: GPU index on this node (0 to num_gpus-1)
    - RANK: Global rank across all nodes
    - WORLD_SIZE: Total number of processes

    We must set the CUDA device before loading any model.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        # Set CUDA device for this process BEFORE any CUDA operations
        torch.cuda.set_device(local_rank)

        # Initialize process group
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
            )

        print(f"[Rank {rank}/{world_size}] Using GPU {local_rank}")

    return local_rank, rank, world_size


def main():
    # Setup distributed FIRST, before any CUDA operations
    local_rank, rank, world_size = setup_distributed()

    # Favor flash/mem-efficient SDPA kernels to reduce attention memory
    enable_flash_sdpa()

    parser = argparse.ArgumentParser(
        description='Unsloth LoRA/QLoRA training for MoE and large models'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()

    # Only print from rank 0 to avoid spam
    is_main_process = rank == 0

    # Load configuration
    if is_main_process:
        print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Create output directory (only rank 0 writes files)
    output_dir = config['training']['output_dir']
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

        # Save config for reproducibility
        config_save_path = os.path.join(output_dir, 'training_config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved configuration to: {config_save_path}")

    # Barrier to ensure output dir exists before other ranks proceed
    if world_size > 1:
        torch.distributed.barrier()

    # Setup model with LoRA
    model, tokenizer = setup_model_and_tokenizer(config)

    # Clear cache before dataset load
    gc.collect()
    torch.cuda.empty_cache()

    # Load dataset
    dataset = load_training_dataset(config, tokenizer)

    if is_main_process:
        log_memory("After dataset load")

    # Setup training config
    sft_config = setup_training_config(config)

    if is_main_process:
        print(f"\nTraining Configuration:")
        print(f"  Output dir: {sft_config.output_dir}")
        print(f"  Batch size: {sft_config.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {sft_config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps * world_size}")
        print(f"  Learning rate: {sft_config.learning_rate}")
        print(f"  Optimizer: {sft_config.optim}")
        print(f"  Epochs: {sft_config.num_train_epochs}")

    # Avoid Accelerate converting logits to float32 (adds a huge copy for 32k seqs)
    disable_accelerate_output_casting()

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user",
        response_part="<|im_start|>assistant",
    )

    if is_main_process:
        log_memory("After trainer creation")

    # Train
    if is_main_process:
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}")

    if args.resume_from_checkpoint:
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    if is_main_process:
        log_memory("After training")
        print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Save model (only from main process)
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Saving model to {output_dir}/final_model...")
        print(f"{'='*60}")

        final_model_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # Optionally merge LoRA weights and save full model
        if config.get('save_merged', False):
            print("Merging LoRA weights and saving full model...")
            merged_path = os.path.join(output_dir, "merged_model")
            model.save_pretrained_merged(
                merged_path,
                tokenizer,
                save_method="merged_16bit"  # or "merged_4bit" for quantized
            )

        print("\nTraining complete!")

    # Clean up distributed
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()