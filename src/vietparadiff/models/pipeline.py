"""End-to-end modular VietParaDiff pipeline."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from vietparadiff.data.graphemes import VietnameseTokenizer

from .content import VisualGraphemeContentEncoder
from .diffusion import LatentDiffusion
from .htr import LineHTR, crop_lines
from .layout import AutoregressiveLayoutPlanner, layout_losses, rasterize_layout
from .style import FactorizedStyleEncoder, haar_highpass
from .topology import DiacriticTopologyDetector, exact_topology_from_parts, target_topology_from_layout, topology_loss
from .unet import ConditionalUNet, DiacriticStrokeRefiner
from .vae import DualBandVAE, vae_loss


class VietParaDiff(nn.Module):
    """Complete VietParaDiff implementation with stage-specific forward methods.

    The model is intentionally modular: VAE, HTR, topology, style-layout, and
    diffusion are trained in separate stages so dependencies are never random by
    accident.  This mirrors the paper training curriculum.
    """

    def __init__(self, cfg: dict, tokenizer: VietnameseTokenizer) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        vocab = tokenizer.vocab_sizes
        dim = int(cfg["text"]["dim"])
        self.content = VisualGraphemeContentEncoder(
            vocab,
            dim=dim,
            layers=cfg["text"]["layers"],
            heads=cfg["text"]["heads"],
            max_tokens=cfg["text"]["max_tokens"],
            visual_base=cfg["text"].get("visual_base", 96),
        )
        self.style = FactorizedStyleEncoder(
            dim=dim,
            tokens_per_group=cfg["style"]["tokens_per_group"],
            codebook_size=cfg["style"]["codebook_size"],
            writer_classes=cfg["style"]["writer_classes"],
            base_channels=cfg["style"].get("base_channels", 96),
            transformer_layers=cfg["style"].get("transformer_layers", 4),
            heads=cfg["style"].get("heads", cfg["text"]["heads"]),
            pooled_grid=cfg["style"].get("pooled_grid", [8, 32]),
        )
        self.layout = AutoregressiveLayoutPlanner(dim, cfg["layout"]["hidden_dim"], cfg["layout"]["fields"])
        self.vae = DualBandVAE(cfg["image"]["channels"], cfg["vae"]["latent_channels"], cfg["vae"]["base_channels"])
        self.htr = LineHTR(vocab, cfg["htr"]["hidden_dim"])
        self.topology = DiacriticTopologyDetector(cfg["image"]["channels"], cfg.get("topology", {}).get("base_channels", 96))
        self.unet = ConditionalUNet(
            latent_ch=cfg["vae"]["latent_channels"],
            layout_ch=cfg["layout"]["fields"],
            context_dim=dim,
            base=cfg["unet"]["base_channels"],
            time_dim=cfg["unet"]["time_dim"],
            channel_mults=cfg["unet"].get("channel_mults", [1, 2, 4, 4]),
            num_res_blocks=cfg["unet"].get("num_res_blocks", 2),
            heads=cfg["unet"].get("heads", cfg["text"]["heads"]),
        )
        self.refiner = DiacriticStrokeRefiner(cfg["vae"]["latent_channels"], cfg["vae"]["latent_channels"], cfg["layout"]["fields"], dim, cfg["unet"]["base_channels"])
        self.diffusion = LatentDiffusion(cfg["diffusion"]["steps"], min_snr_gamma=cfg["diffusion"].get("min_snr_gamma", 5.0))

    def move_diffusion_to_device(self, device: torch.device) -> None:
        self.diffusion.to(device)

    def encode_conditions(self, batch: dict, teacher_ratio: float = 0.0) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Encode text, reference style, and layout fields for generator stages."""

        content = self.content(batch["text"], batch["archetypes"], batch["text_mask"])
        style = self.style(batch["reference"])
        layout_pred = self.layout(
            content,
            batch["text_mask"],
            style["global_style"],
            style.get("layout"),
            batch.get("boxes"),
            batch.get("line_ids"),
            teacher_ratio,
        )
        down = self.cfg["image"]["downsample"]
        h = batch["image"].shape[-2] // down
        w = batch["image"].shape[-1] // down
        fields = rasterize_layout(layout_pred["boxes"], layout_pred["anchors"], batch["text_mask"], h, w)
        context = torch.cat([content, style["all_tokens"]], dim=1)
        style_mask = torch.ones(style["all_tokens"].shape[:2], dtype=torch.bool, device=context.device)
        context_mask = torch.cat([batch["text_mask"], style_mask], dim=1)
        return {"content": content, "style": style, "layout": layout_pred, "fields": fields, "context": context, "context_mask": context_mask}

    def forward_vae(self, batch: dict) -> dict[str, torch.Tensor]:
        out = self.vae(batch["image"])
        return vae_loss(out, batch["image"], self.cfg["loss"]["kl"], self.cfg["loss"]["frequency"], self.cfg["loss"]["orthogonality"])

    def forward_htr(self, batch: dict) -> dict[str, torch.Tensor]:
        crops, owner = crop_lines(batch["image"], batch["line_boxes"])
        if crops.numel() == 0:
            return {"loss": batch["image"].sum() * 0}
        targets, lengths = [], []
        # Use per-line text when present; otherwise fallback to the paragraph text.
        per_paragraph_line_index: dict[int, int] = {}
        for bi in owner.tolist():
            local_idx = per_paragraph_line_index.get(bi, 0)
            per_paragraph_line_index[bi] = local_idx + 1
            line_texts = batch.get("line_texts", [[]])[bi]
            text = line_texts[min(local_idx, len(line_texts) - 1)] if line_texts else batch["transcript"][bi]
            enc = self.tokenizer.encode(text)
            ids = torch.tensor(enc["full"], dtype=torch.long, device=crops.device)
            ids = ids[ids > 0]
            if ids.numel() == 0:
                continue
            targets.append(ids)
            lengths.append(ids.numel())
        if not targets:
            return {"loss": batch["image"].sum() * 0}
        target = torch.cat(targets)
        target_lengths = torch.tensor(lengths, dtype=torch.long, device=crops.device)
        loss = self.htr.ctc_loss(crops, target, target_lengths)
        return {"loss": loss, "htr/ctc": loss}

    def forward_topology(self, batch: dict) -> dict[str, torch.Tensor]:
        cond = self.encode_conditions(batch, teacher_ratio=1.0)
        weak = target_topology_from_layout(cond["fields"])
        exact = exact_topology_from_parts(batch.get("parts", []), cond["layout"]["boxes"], cond["layout"]["anchors"], batch["text_mask"], weak.shape[-2], weak.shape[-1]) if "parts" in batch else weak
        target = torch.maximum(weak, exact)
        pred = self.topology(batch["image"])
        return topology_loss(pred, target, cond["fields"][:, 0:1])

    def forward_style_layout(self, batch: dict, global_step: int = 0) -> dict[str, torch.Tensor]:
        start = self.cfg["layout"]["teacher_forcing_start"]
        end = self.cfg["layout"]["teacher_forcing_end"]
        decay = max(1, self.cfg["layout"]["teacher_forcing_decay_steps"])
        ratio = max(end, start - (start - end) * global_step / decay)
        cond = self.encode_conditions(batch, teacher_ratio=ratio)
        losses = layout_losses(cond["layout"], batch["boxes"], batch["text_mask"], batch.get("line_ids"), self.cfg["loss"].get("break", 0.1))
        writer = F.cross_entropy(cond["style"]["writer_logits"], batch["writer"].clamp_max(cond["style"]["writer_logits"].shape[-1] - 1))
        total = sum(losses.values()) + self.cfg["loss"]["style"] * (cond["style"]["vq_loss"] + writer)
        losses.update({
            "loss": total,
            "style/vq": cond["style"]["vq_loss"],
            "style/writer": writer,
            "style/perplexity": cond["style"]["perplexity"],
            "layout/teacher_ratio": torch.tensor(ratio, device=batch["image"].device),
        })
        return losses

    def forward_diffusion(self, batch: dict, global_step: int = 0) -> dict[str, torch.Tensor]:
        cond = self.encode_conditions(batch, teacher_ratio=0.0)
        with torch.no_grad():
            posts = self.vae.encode(batch["image"])
            low = posts["low"].mean
            high = posts["high"].mean
        self.diffusion.to(batch["image"].device)
        t = torch.randint(0, self.diffusion.steps, (low.shape[0],), device=low.device)
        noise = torch.randn_like(low)
        noisy = self.diffusion.q_sample(low, t, noise)
        if self.training and torch.rand((), device=low.device) < self.cfg["diffusion"].get("cond_drop_prob", 0.15):
            context = torch.zeros_like(cond["context"])
            fields = torch.zeros_like(cond["fields"])
        else:
            context = cond["context"]
            fields = cond["fields"]
        pred_noise = self.unet(noisy, t, context, fields, cond["context_mask"])
        noise_loss = self.diffusion.noise_loss(pred_noise, noise, t)
        pred_low = self.diffusion.predict_clean(noisy, t, pred_noise)
        pred_high = self.refiner(pred_low, cond["fields"], cond["style"]["stroke"])
        refiner_loss = F.l1_loss(pred_high, high)
        recon = self.vae.decode(pred_low, pred_high)
        freq = F.l1_loss(haar_highpass(recon), haar_highpass(batch["image"]))
        topo_tgt = target_topology_from_layout(cond["fields"])
        topo = topology_loss(self.topology(recon), topo_tgt)["loss"]
        total = noise_loss + self.cfg["loss"]["refiner"] * refiner_loss + self.cfg["loss"]["frequency"] * freq + self.cfg["loss"]["topology"] * topo
        return {"loss": total, "diffusion/noise": noise_loss, "diffusion/refiner": refiner_loss, "diffusion/frequency": freq, "diffusion/topology": topo}

    @torch.no_grad()
    def generate(self, batch: dict, steps: int = 50, guidance: float = 5.0) -> torch.Tensor:
        cond = self.encode_conditions(batch, teacher_ratio=0.0)
        b = batch["image"].shape[0]
        h, w = cond["fields"].shape[-2:]
        low = self.diffusion.ddim_sample(
            self.unet,
            (b, self.cfg["vae"]["latent_channels"], h, w),
            steps,
            cond["context"],
            cond["fields"],
            cond["context_mask"],
            guidance,
        )
        high = self.refiner(low, cond["fields"], cond["style"]["stroke"])
        return self.vae.decode(low, high)
