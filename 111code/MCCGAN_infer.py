# MCCGAN_infer.py

"""
Batch Inference: Generate 10 images for each of the specified 12 groups (A, T, Temp, Time).

Usage Example:

  python infer_mcv_cgan_batch.py --weights runs/MCCGAN/weights/G_final.pt \
      --out_dir runs/MCCGAN/infer_12x10
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


# ============ Configuration (must be consistent with train) ============
@dataclass
class CFG:
    IMG_SIZE: int = 256
    IN_CHANNELS: int = 3
    Z_DIM: int = 128
    NGF: int = 64

    # Conditional keys and normalization range (consistent with training)
    COND_KEYS: Tuple[str, ...] = ("A", "T", "Temp", "Time")
    RANGES: Dict[str, Tuple[float, float]] = None

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.RANGES is None:
            self.RANGES = {

                "A":    (6.0, 9.0),
                "T":    (6.0, 9.0),
                "Temp": (350.0, 430.0),
                "Time": (4.0, 12.0),
            }

    @property
    def COND_DIM(self) -> int:
        return len(self.COND_KEYS)

    def clamp_and_normalize(self, cond_raw: Dict[str, float]) -> List[float]:
        """Fit the original conditions into the training range and normalize [0,1]"""
        out = []
        for k in self.COND_KEYS:
            lo, hi = self.RANGES[k]
            v = float(cond_raw[k])
            v = max(lo, min(hi, v))
            out.append((v - lo) / (hi - lo + 1e-12))
        return out

CFG = CFG()


# ============ Generator (consistent with training) ============
class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        h = F.relu(self.bn1(x))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.skip(x)
        return h + x

class Generator(nn.Module):
    def __init__(self, img_size, z_dim=128, cond_dim=4, ngf=64, out_ch=3):
        super().__init__()
        self.fc = nn.Linear(z_dim + cond_dim, 4*4*ngf*8)
        n_up = int(np.log2(img_size) - 2)
        blocks = []
        in_ch = ngf * 8
        for _ in range(n_up):
            out_ch_ = max(in_ch // 2, ngf // 16)
            blocks.append(ResBlockUp(in_ch, out_ch_))
            in_ch = out_ch_
        self.ups = nn.ModuleList(blocks)
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, z, cond):
        h = torch.cat([z, cond], dim=1)
        h = self.fc(h).view(-1, 8*CFG.NGF, 4, 4)
        for blk in self.ups:
            h = blk(h)
        h = F.relu(self.bn(h))
        x = torch.sigmoid(self.conv_out(h))
        return x


# ============ Batch Inference ============
@torch.no_grad()
def generate_for_combo(G: Generator, combo_name: str, cond_raw: Dict[str, float],
                       n: int, seed_base: int, out_root: Path):
    """Generate n images for a single condition combination and save them to a subdirectory combo_name/"""
    out_dir = out_root / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate n images in one batch
    cond_vec = CFG.clamp_and_normalize(cond_raw)
    cond = torch.tensor([cond_vec]*n, dtype=torch.float32, device=CFG.DEVICE)

    # For diversity, each image is given a different seed
    z_list = []
    for i in range(n):
        s = seed_base + i + 1
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        z_list.append(torch.randn(1, CFG.Z_DIM, device=CFG.DEVICE))
    z = torch.cat(z_list, dim=0)  # [n, Z]

    fake = G(z, cond)  # [n,3,H,W]

    # SAVE
    for i in range(n):
        fname = f"{combo_name}_{i+1:02d}.png"
        save_image(fake[i], str(out_dir / fname))




def main():
    parser = argparse.ArgumentParser(description="MCCGAN Batch reasoning (12 sets of conditions, 10 sheets each)")
    parser.add_argument("--weights", type=str, default="runs/MCCGAN/weights/G_final.pt")
    parser.add_argument("--out_dir", type=str, default="runs/MCCGAN/infer_12x100")
    parser.add_argument("--n_per", type=int, default=10, help="generated number per combination")
    parser.add_argument("--seed_base", type=int, default=20250919, help="Basic random seed")
    args = parser.parse_args()

    # Load generator
    G = Generator(CFG.IMG_SIZE, z_dim=CFG.Z_DIM, cond_dim=CFG.COND_DIM,
                  ngf=CFG.NGF, out_ch=CFG.IN_CHANNELS).to(CFG.DEVICE).eval()
    state = torch.load(args.weights, map_location=CFG.DEVICE)
    G.load_state_dict(state)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 12 parameters combined
    combos = [
        ("AT66_350_4",  {"A": 6, "T": 6, "Temp": 350, "Time": 4}),
        ("AT66_390_4",  {"A": 6, "T": 6, "Temp": 390, "Time": 4}),
        ("AT66_430_4",  {"A": 6, "T": 6, "Temp": 430, "Time": 4}),
        ("AT69_350_12", {"A": 6, "T": 9, "Temp": 350, "Time": 12}),
        ("AT69_390_12", {"A": 6, "T": 9, "Temp": 390, "Time": 12}),
        ("AT69_430_12", {"A": 6, "T": 9, "Temp": 430, "Time": 12}),
        ("AT96_350_12", {"A": 9, "T": 6, "Temp": 350, "Time": 12}),
        ("AT96_390_12", {"A": 9, "T": 6, "Temp": 390, "Time": 12}),
        ("AT96_430_12", {"A": 9, "T": 6, "Temp": 430, "Time": 12}),
        ("AT99_350_4",  {"A": 9, "T": 9, "Temp": 350, "Time": 4}),
        ("AT99_390_4",  {"A": 9, "T": 9, "Temp": 390, "Time": 4}),
        ("AT99_430_4",  {"A": 9, "T": 9, "Temp": 430, "Time": 4}),
    ]

    # Generated
    for idx, (name, cond) in enumerate(combos, start=1):
        seed_base = args.seed_base + idx * 1000
        print(f"[{idx:02d}/{len(combos)}] generate {name} × {args.n_per}  …")
        generate_for_combo(G, name, cond, n=args.n_per, seed_base=seed_base, out_root=out_root)

    print(f"Finish：all {len(combos)*args.n_per} ，Output directory：{out_root.resolve()}")


if __name__ == "__main__":
    main()
