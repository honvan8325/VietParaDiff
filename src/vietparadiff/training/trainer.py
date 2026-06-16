"""Complete PyTorch training loop for staged VietParaDiff training.

The loop treats ``max_steps`` as optimizer-update steps, not micro-batches.  This
matters when gradient accumulation is used for paper-scale models.  Checkpoints
store model, optimizer, and AMP scaler states, so ``--resume`` continues the run
faithfully instead of only reloading weights.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .stages import configure_stage, forward_stage


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move tensor leaves of a nested batch dict to a device."""

    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            out[key] = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            out[key] = value
    return out


class Trainer:
    """Stage-aware trainer with AMP, gradient accumulation, progress, and resume."""

    def __init__(self, model, cfg: dict, device: torch.device) -> None:
        self.model = model.to(device)
        self.model.move_diffusion_to_device(device)
        self.cfg = cfg
        self.device = device

    def _checkpoint_root(self, stage: str) -> Path:
        return Path(self.cfg["run_dir"]) / stage

    def fit(
        self,
        loader: DataLoader,
        stage: str,
        max_steps: int | None = None,
        resume: str | Path | None = None,
    ) -> None:
        """Train one stage.

        Parameters
        ----------
        loader:
            Training dataloader.
        stage:
            One of ``vae``, ``htr``, ``style_layout``, ``topology``, or
            ``diffusion``.
        max_steps:
            Number of completed optimizer updates to reach.  With
            ``grad_accum_steps=16`` and ``max_steps=400000``, the trainer performs
            400k optimizer updates, not 25k.
        resume:
            Optional full same-stage checkpoint.  Model, optimizer, scheduler
            placeholder, AMP scaler, and global step are restored.
        """

        configure_stage(self.model, stage)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters for stage {stage}")

        opt = torch.optim.AdamW(
            params,
            lr=float(self.cfg["training"]["lr"]),
            weight_decay=float(self.cfg["training"].get("weight_decay", 0.0)),
        )
        amp_enabled = bool(self.cfg["training"].get("amp", True)) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        accum = int(self.cfg["training"].get("grad_accum_steps", 1))
        max_steps = int(max_steps or self.cfg["training"]["max_steps"])
        save_every = int(self.cfg["training"].get("checkpoint_every_steps", 1000))
        log_every = int(self.cfg["training"].get("log_every_steps", 50))

        global_step = 0
        if resume:
            payload = load_checkpoint(resume, self.model, optimizer=opt, scaler=scaler, strict=True)
            global_step = int(payload.get("step", 0))
            print(f"resumed {stage} from {resume} at optimizer step {global_step}")

        if global_step >= max_steps:
            print(f"stage {stage} already reached max_steps={max_steps}; nothing to do")
            return

        self.model.train()
        opt.zero_grad(set_to_none=True)
        root = self._checkpoint_root(stage)
        pbar = tqdm(total=max_steps, initial=global_step, desc=f"train:{stage}", unit="step", dynamic_ncols=True)
        micro_step = 0
        last_losses: dict[str, torch.Tensor] = {}

        while global_step < max_steps:
            for batch in loader:
                batch = move_batch(batch, self.device)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    losses = forward_stage(self.model, batch, stage, global_step)
                    loss = losses["loss"] / accum
                scaler.scale(loss).backward()
                micro_step += 1
                last_losses = losses

                if micro_step % accum != 0:
                    continue

                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(self.cfg["training"].get("grad_clip", 1.0)))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                global_step += 1
                pbar.update(1)

                if global_step % log_every == 0 or global_step == 1:
                    postfix = {
                        k: float(v.detach().cpu())
                        for k, v in last_losses.items()
                        if torch.is_tensor(v) and v.numel() == 1
                    }
                    postfix["lr"] = opt.param_groups[0]["lr"]
                    pbar.set_postfix(postfix)

                if global_step % save_every == 0:
                    save_checkpoint(root / "latest.pt", self.model, opt, None, global_step, self.cfg, scaler=scaler)
                    save_checkpoint(root / f"step_{global_step:08d}.pt", self.model, opt, None, global_step, self.cfg, scaler=scaler)

                if global_step >= max_steps:
                    break
        pbar.close()
        save_checkpoint(root / "latest.pt", self.model, opt, None, global_step, self.cfg, scaler=scaler)
