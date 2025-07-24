from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import sys
from pathlib import Path
import yaml
import shutil
import contextlib

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
# from adam_atan2 import AdamATan2 # Replaced with standard AdamW

from .language_dataset import LanguageDataset, load_metadata
from .utils.functions import load_model_class, get_model_source_path
from .models.losses import SoftmaxCrossEntropyLoss


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    dataset_dir: str

    # Hyperparams
    global_batch_size: int
    block_size: int
    epochs: int
    grad_accum_steps: int = 1

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    scaler: GradScaler
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int):
    dataset = LanguageDataset(data_dir=config.dataset_dir, split=split, block_size=config.block_size)
    
    # The sampler handles distributing data across GPUs for distributed training
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=config.global_batch_size // world_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        shuffle=(sampler is None) # Shuffle if not using a distributed sampler
    )
    return dataloader

def create_model(config: PretrainConfig, meta: dict, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // (world_size * config.grad_accum_steps),
        vocab_size=meta['vocab_size'],
        seq_len=config.block_size,
        causal=True  # This is now a standard causal language model
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = SoftmaxCrossEntropyLoss # Use our new, simple loss head

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            # DDP will handle this automatically
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[RANK])
            pass

    # A single AdamW optimizer for all parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        fused=True
    )

    optimizers = [optimizer]
    optimizer_lrs = [config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, meta: dict, world_size: int):
    # Estimated total training steps
    # We'll calculate this based on the dataset size and epochs later if needed
    
    # Model
    model, optimizers, optimizer_lrs = create_model(config, meta, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=0, # Will be updated later

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        scaler=GradScaler(),
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


# The old evaluate function is removed for now to simplify the refactor.
# A new, standard evaluation loop can be added later.

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.dataset_dir).capitalize()} HRM-Language"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    # Enable cuDNN benchmarking for static input sizes
    torch.backends.cudnn.benchmark = True
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset and metadata
    meta = load_metadata(config.dataset_dir)
    if meta is None:
        raise FileNotFoundError(f"meta.json not found in {config.dataset_dir}. Please run prepare_data.py first.")
    train_loader = create_dataloader(config, "train", rank=RANK, world_size=WORLD_SIZE)
    # val_loader for evaluation can be added here later

    # Train state
    train_state = init_train_state(config, meta, world_size=WORLD_SIZE)
    
    # Update total steps based on dataset size
    train_state.total_steps = (len(train_loader.dataset) // config.global_batch_size) * config.epochs

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Initialize the carry state before the loop begins
    # We need to access the .model attribute to get to the core HRM model,
    # as the top-level `train_state.model` is the SoftmaxCrossEntropyLoss wrapper.
    unwrapped_model = train_state.model.module if isinstance(train_state.model, nn.parallel.DistributedDataParallel) else train_state.model
    initial_batch = {"inputs": torch.zeros((config.global_batch_size // WORLD_SIZE, config.block_size), dtype=torch.long, device="cuda")}
    train_state.carry = unwrapped_model.model.initial_carry(initial_batch)


    # Training Loop
    for epoch in range(config.epochs):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {epoch}")

        ############ Train Iter
        train_state.model.train()
        for micro_step, (X, Y) in enumerate(train_loader):
            X, Y = X.cuda(), Y.cuda()
            
            is_accum_step = (micro_step + 1) % config.grad_accum_steps != 0

            # Forward with Automatic Mixed Precision
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # For DDP, we prevent gradient sync on accumulation steps
                with train_state.model.no_sync() if is_accum_step and isinstance(train_state.model, nn.parallel.DistributedDataParallel) else contextlib.nullcontext():
                    batch = {"inputs": X, "labels": Y}
                    # The trainer is now responsible for managing the carry state
                    train_state.carry, loss, metrics = train_state.model(train_state.carry, batch)

            # Scale loss and backward
            scaled_loss = train_state.scaler.scale(loss / config.grad_accum_steps)
            scaled_loss.backward()

            if not is_accum_step:
                # Allreduce (if distributed) and apply optimizer
                # Note: DDP handles gradient all-reduce automatically with no_sync context
                
                # Apply optimizer with scaler
                lr_this_step = None    
                for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
                    lr_this_step = compute_lr(base_lr, config, train_state)
                    for param_group in optim.param_groups:
                        param_group['lr'] = lr_this_step
                    
                    train_state.scaler.step(optim)
                    optim.zero_grad(set_to_none=True)
                
                train_state.scaler.update()
                train_state.step += 1

                if RANK == 0:
                    # Simplified metric logging
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_this_step
                    }, step=train_state.step)
                    progress_bar.update(1)

        # Evaluation would go here in a full implementation

        ############ Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (epoch == config.epochs - 1)):
            save_train_state(config, train_state)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
