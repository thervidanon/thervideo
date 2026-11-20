"""RGB video -> thermal video with the offline TherVideo model."""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
import torch
from safetensors import safe_open

from .model import TherVideoTransformer


class LowRankLinear(torch.nn.Module):
    """A frozen Wan2.2 linear layer plus the TherVideo low-rank adapters, applied in order:
    y = W x + s_0 * up_0 (down_0 x) + s_1 * up_1 (down_1 x)."""

    def __init__(self, base: torch.nn.Linear, adapters):
        super().__init__()
        self.base = base
        self.scales = [float(s) for _, _, s in adapters]
        for i, (down, up, _) in enumerate(adapters):
            self.register_buffer(f"down{i}", down, persistent=False)
            self.register_buffer(f"up{i}", up, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        for i, s in enumerate(self.scales):
            down, up = getattr(self, f"down{i}"), getattr(self, f"up{i}")
            y = y + (x @ down.t().to(x.dtype)) @ up.t().to(x.dtype) * s
        return y


WINDOW = 33          # frames per generation window (9 latent frames)
SENSORS = ("radiometric", "agc8")


def flow_sigmas(steps: int, shift: float) -> torch.Tensor:
    """Shifted flow-matching noise levels from 1 towards 0 (the final step goes to 0)."""
    s = torch.linspace(1.0, 0.0, steps + 1)[:-1]
    return shift * s / (1 + (shift - 1) * s)


def windows(n: int, size: int = WINDOW) -> List[Tuple[int, int]]:
    """Non-overlapping windows over n frames as (start, first frame to keep). A trailing remainder is covered by
    the last `size` frames, keeping only the frames not generated yet."""
    if n < size:
        raise ValueError(f"need at least {size} frames, got {n}")
    out, s = [], 0
    while s + size <= n:
        out.append((s, 0))
        s += size
    if s < n:
        out.append((n - size, size - (n - s)))
    return out


class TherVideoPipeline:
    def __init__(self, transformer: TherVideoTransformer, vae, text_embeds: dict, device: torch.device):
        self.transformer, self.vae, self.text_embeds, self.device = transformer, vae, text_embeds, device
        cfg = vae.config
        self.lat_mean = torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.lat_std = torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1, 1)

    @classmethod
    def from_pretrained(cls, wan_dir: str, checkpoint: str, device: str = "cuda") -> "TherVideoPipeline":
        """wan_dir: local copy of Wan-AI/Wan2.2-TI2V-5B-Diffusers; checkpoint: TherVideo .safetensors file."""
        from diffusers import AutoencoderKLWan

        device = torch.device(device)
        torch.backends.cuda.enable_cudnn_sdp(False)
        transformer = TherVideoTransformer.from_pretrained(wan_dir, subfolder="transformer", torch_dtype=torch.bfloat16)
        text_embeds = {}
        with safe_open(checkpoint, "pt") as f:
            meta = f.metadata() or {}
            if meta.get("format") != "thervideo-offline-v1":
                raise ValueError(f"{checkpoint} is not a TherVideo offline checkpoint")
            scales = json.loads(meta["scales"])
            n = 0
            for key in f.keys():
                if key.startswith("lora.") and key.endswith(".0.down"):
                    name = key[len("lora."):-len(".0.down")]
                    adapters = [(f.get_tensor(f"lora.{name}.{i}.down").float(), f.get_tensor(f"lora.{name}.{i}.up").float(),
                                 s) for i, s in enumerate(scales)]
                    parent, attr = name.rsplit(".", 1)
                    parent = transformer.get_submodule(parent)
                    setattr(parent, attr, LowRankLinear(getattr(parent, attr), adapters))
                    n += 1
                elif key.startswith("text."):
                    _, sensor, media = key.split(".")
                    text_embeds[sensor, media] = f.get_tensor(key)
            transformer.embed_rgb.copy_(f.get_tensor("embed.rgb"))
            transformer.embed_thermal.copy_(f.get_tensor("embed.thermal"))
        if n != int(meta.get("num_lora", n)):
            raise ValueError(f"merged {n} adapters, expected {meta['num_lora']}")
        transformer = transformer.to(device).eval().requires_grad_(False)
        vae = AutoencoderKLWan.from_pretrained(wan_dir, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        return cls(transformer, vae, text_embeds, device)

    @torch.no_grad()
    def generate_window(self, rgb: np.ndarray, sensor: str = "radiometric", media: str = "video", steps: int = 30,
                        guidance: float = 2.0, shift: float = 5.0, seed: int = 0) -> np.ndarray:
        """rgb: uint8 [33, 512, 640, 3] -> thermal uint8 [33, 512, 640]."""
        dev = self.device
        x_rgb = torch.from_numpy(rgb).permute(3, 0, 1, 2)[None].float().to(dev) / 127.5 - 1.0
        cond = (self.vae.encode(x_rgb).latent_dist.mode() - self.lat_mean) / self.lat_std
        text = self.text_embeds[sensor, media].to(dev)[None]
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(cond.shape, generator=g, device=dev, dtype=torch.float32)
        sig = flow_sigmas(steps, shift)
        cc = torch.cat([cond, torch.zeros_like(cond)]).to(torch.bfloat16)
        tt = torch.cat([text, text])
        for i in range(steps):
            t = (sig[i] * 1000.0).view(1).expand(2).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = self.transformer(torch.cat([x, x]).to(torch.bfloat16), cc, t, tt).float()
            v_c, v_u = v.chunk(2)
            v = v_u + guidance * (v_c - v_u)
            nxt = sig[i + 1] if i + 1 < steps else 0
            x = x + v * (nxt - sig[i])
        out = self.vae.decode((x * self.lat_std + self.lat_mean).float()).sample.clamp(-1, 1)[0]
        return ((out.mean(0) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()

    def __call__(self, frames: np.ndarray, sensor: str = "radiometric", steps: int = 30, guidance: float = 2.0,
                 seed: int = 0, progress: bool = True) -> np.ndarray:
        """frames: uint8 [N, 512, 640, 3] at 10 Hz, N >= 33 -> thermal uint8 [N, 512, 640]."""
        if sensor not in SENSORS:
            raise ValueError(f"sensor must be one of {SENSORS}")
        if frames.shape[1:] != (512, 640, 3):
            raise ValueError(f"expected [N, 512, 640, 3] frames, got {frames.shape}")
        n = frames.shape[0]
        out = np.zeros((n, 512, 640), np.uint8)
        wins = windows(n)
        for i, (s, keep) in enumerate(wins):
            out[s + keep:s + WINDOW] = self.generate_window(frames[s:s + WINDOW], sensor, "video", steps, guidance,
                                                            seed=seed)[keep:]
            if progress:
                print(f"window {i + 1}/{len(wins)}", flush=True)
        return out
