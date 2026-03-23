# train_MCCGAN.py

"""
- Multidimensional Conditional Vectors: Default 4-dimensional (A, T, Temp, Time),
- Data Directory Structure (Example):

new2/
AT66_350_4/
*.tif (approx. 648 images)
AT69_430_12/
*.tif

Running case：
  python MCCGAN_train.py --data_root new2 --epochs 100 --batch_size 16
"""

import os, re, math, random, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


# ===================== Config（condition vector） =====================
@dataclass
class Config:
    # root
    DATA_ROOT: str = "new2"
    OUTPUT_ROOT: str = "runs/mcv_cgan"

    # image & model
    IMG_SIZE: int = 256
    IN_CHANNELS: int = 3
    Z_DIM: int = 128
    NGF: int = 64
    NDF: int = 64

    # Conditional dimensions (determined by keys, no manual modification required)
    COND_KEYS: Tuple[str, ...] = ("A", "T", "Temp", "Time")
    # The range of values ​​for each condition (used for normalization [0,1]）
    RANGES: Dict[str, Tuple[float, float]] = None

    # Training hyperparameters
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
    LR_G: float = 2e-4
    LR_D: float = 2e-4
    BETAS: Tuple[float, float] = (0.5, 0.999)
    NUM_WORKERS: int = 4
    USE_FLIP_AUG: bool = True

    # Loss weights
    LAMBDA_COND_D: float = 1.0
    LAMBDA_COND_G: float = 1.0
    LAMBDA_PALETTE: float = 5.0
    LAMBDA_R1: float = 0.0

    # Logs & Saving
    SAMPLE_EVERY: int = 1
    SAVE_EVERY: int = 5

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Default；Add here when you need to expand dimensions
        if self.RANGES is None:
            self.RANGES = {
                "A":    (6.0, 9.0),    # A
                "T":    (6.0, 9.0),    # T
                "Temp": (350.0, 430.0),# T
                "Time": (4.0, 12.0),   # t
            }

    @property
    def COND_DIM(self) -> int:
        return len(self.COND_KEYS)

    def normalize_cond(self, cond_dict: Dict[str, float]) -> List[float]:
        """Normalized by RANGES, the output vector follows the same order as COND_KEYS."""
        out = []
        for k in self.COND_KEYS:
            lo, hi = self.RANGES[k]
            v = float(cond_dict[k])
            out.append((v - lo) / (hi - lo + 1e-12))
        return out

CFG = Config()


# ===================== Function =====================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_folder_to_cond(name: str) -> Dict[str, float]:
    """
    Parse subdirectory names of the form ATxy_TTT_HH -> Condition dictionary
    {"A":x, "T":y, "Temp":TTT, "Time":HH}
    Like AT69_390_12 -> A=6, T=9, Temp=390, Time=12
    """
    m = re.match(r"^AT(\d)(\d)_(\d{3})_(\d+)$", name.strip())
    if not m:
        raise ValueError(f"The subdirectory name should be ATxy_TTT_HH, for example, AT69_390_12, currently：{name}")
    A = int(m.group(1))
    T = int(m.group(2))
    Temp = int(m.group(3))
    Time = int(m.group(4))
    return {"A":A, "T":T, "Temp":Temp, "Time":Time}

def save_image_safe(tensor, path: Path):
    try:
        save_image(tensor, str(path.with_suffix(".png")))
    except Exception:
        save_image(tensor, str(path.with_suffix(".tif")))


# ===================== Dataset =====================
class AlloyDatasetMCV(Dataset):
    def __init__(self, root: str, img_size: int, use_flip_aug=True):
        self.root = Path(root); self.img_size = img_size; self.use_flip_aug = use_flip_aug
        self.samples = []
        exts = {".tif",".tiff",".png",".jpg",".jpeg",".bmp"}
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir(): continue
            try:
                cond_raw = parse_folder_to_cond(sub.name)
            except Exception:
                continue
            cond_norm = CFG.normalize_cond(cond_raw)
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.suffix.lower() in exts:
                    self.samples.append((p, cond_norm))
        if not self.samples:
            raise RuntimeError(f"{self.root} Training images not found, or subdirectory name does not match. ATxy_TTT_HH")

        self.base_transform = transforms.ToTensor()

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, cond = self.samples[idx]
        img = Image.open(path)  # Do not change color/mode
        w,h = img.size

        if (w, h) == (CFG.IMG_SIZE, CFG.IMG_SIZE):
            pass
        elif w >= CFG.IMG_SIZE and h >= CFG.IMG_SIZE:
            L = (w - CFG.IMG_SIZE)//2
            T = (h - CFG.IMG_SIZE)//2
            img = img.crop((L, T, L+CFG.IMG_SIZE, T+CFG.IMG_SIZE))
        else:
            raise ValueError(f"Image size smaller {CFG.IMG_SIZE}: {path} -> {img.size}")

        if self.use_flip_aug:
            if random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5: img = img.transpose(Image.FLIP_TOP_BOTTOM)

        x = self.base_transform(img)  # ∈[0,1]
        cond = torch.tensor(cond, dtype=torch.float32)
        return x, cond


# ===================== Model =====================
def spectral_norm(m): return nn.utils.spectral_norm(m)

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
        h = F.relu(self.bn2(h)); h = self.conv2(h)
        x = F.interpolate(x, scale_factor=2, mode="nearest"); x = self.skip(x)
        return h + x

class Generator(nn.Module):
    def __init__(self, img_size, z_dim=128, cond_dim=4, ngf=64, out_ch=3):
        super().__init__()
        self.fc = nn.Linear(z_dim + cond_dim, 4*4*ngf*8)
        n_up = int(math.log2(img_size) - 2)
        blocks = []; in_ch = ngf*8
        for _ in range(n_up):
            out_ch_ = max(in_ch // 2, ngf//16)
            blocks.append(ResBlockUp(in_ch, out_ch_)); in_ch = out_ch_
        self.ups = nn.ModuleList(blocks)
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
    def forward(self, z, cond):
        h = torch.cat([z,cond], dim=1)
        h = self.fc(h).view(-1, 8*CFG.NGF, 4, 4)
        for blk in self.ups: h = blk(h)
        h = F.relu(self.bn(h))
        return torch.sigmoid(self.conv_out(h))

class Discriminator(nn.Module):
    def __init__(self, img_size, in_ch=3, cond_dim=4, ndf=64):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 16)
        self.cond_conv = nn.Conv2d(16, 16, 1, 1, 0)
        n_down = int(math.log2(img_size) - 2)
        ch_in = in_ch + 16; blocks = []; ch_out = ndf
        for _ in range(n_down):
            blocks += [spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 2, 1)),
                       nn.LeakyReLU(0.2, inplace=True)]
            ch_in = ch_out; ch_out = min(ch_out*2, ndf*8)
        self.main = nn.Sequential(*blocks)
        self.conv_last = spectral_norm(nn.Conv2d(ch_in, 1, 4, 1, 0))
        self.cond_reg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            spectral_norm(nn.Linear(ch_in, 64)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(64, cond_dim))
        )
    def forward(self, x, cond):
        B,_,H,W = x.shape
        c = self.cond_proj(cond).view(B,16,1,1).expand(B,16,H,W)
        c = self.cond_conv(c)
        h = torch.cat([x,c], dim=1)
        h = self.main(h)
        score = self.conv_last(h).view(B)
        cond_pred = self.cond_reg(h)
        return score, cond_pred


# ===================== Loss =====================
def d_hinge_loss(r,f): return F.relu(1.-r).mean() + F.relu(1.+f).mean()
def g_hinge_loss(f): return -f.mean()
def r1_gradient_penalty(d_out, x):
    grad = torch.autograd.grad(outputs=d_out.sum(), inputs=x, create_graph=True,
                               retain_graph=True, only_inputs=True)[0]
    return grad.pow(2).view(grad.size(0), -1).sum(1).mean()
def palette_loss_rgb(x):
    B,C,H,W = x.shape
    x_ = x.permute(0,2,3,1).contiguous()
    pal = x_.new_tensor([[0.,0.,0.],[1.,1.,1.],[1.,0.,0.]])
    d2 = (x_.unsqueeze(-2) - pal.view(1,1,1,3,3)).pow(2).sum(-1)
    return d2.min(dim=-1).values.mean()


# ===================== Sampling =====================
def discover_unique_conds(root: Path) -> List[Dict[str, float]]:
    uniq = set()
    for sub in sorted(root.iterdir()):
        if not sub.is_dir(): continue
        try:
            d = parse_folder_to_cond(sub.name)
            uniq.add((d["A"], d["T"], d["Temp"], d["Time"]))
        except Exception:
            pass
    combos = []
    for A,T,Temp,Time in sorted(list(uniq)):
        combos.append({"A":A,"T":T,"Temp":Temp,"Time":Time})
    return combos

@torch.no_grad()
def sample_grid(G, epoch, data_root: Path, out_dir: Path, device: str):
    combos = discover_unique_conds(data_root)
    if not combos: return
    conds = [CFG.normalize_cond(c) for c in combos]
    conds = torch.tensor(conds, dtype=torch.float32, device=device)
    z = torch.randn(conds.size(0), CFG.Z_DIM, device=device)
    G.eval()
    fake = G(z, conds)
    nrow = min(6, max(1, int(round(math.sqrt(len(combos))))))
    grid = make_grid(fake, nrow=nrow, normalize=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(grid, str((out_dir / f"epoch_{epoch:04d}_grid.png")))


# ===================== Train =====================
def train(args):
    # Override runtime configuration
    CFG.DATA_ROOT = args.data_root
    CFG.OUTPUT_ROOT = args.output_root
    CFG.EPOCHS = args.epochs
    CFG.BATCH_SIZE = args.batch_size

    set_seed(42)
    out_root = Path(CFG.OUTPUT_ROOT)
    (out_root/"weights").mkdir(parents=True, exist_ok=True)
    (out_root/"samples").mkdir(parents=True, exist_ok=True)

    dataset = AlloyDatasetMCV(CFG.DATA_ROOT, CFG.IMG_SIZE, use_flip_aug=CFG.USE_FLIP_AUG)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    print(f"Data: {len(dataset)}，Each iteration: {len(loader)} batch，IMG_SIZE={CFG.IMG_SIZE}，Conditional dimension={CFG.COND_DIM}")

    G = Generator(CFG.IMG_SIZE, z_dim=CFG.Z_DIM, cond_dim=CFG.COND_DIM, ngf=CFG.NGF, out_ch=CFG.IN_CHANNELS).to(CFG.DEVICE)
    D = Discriminator(CFG.IMG_SIZE, in_ch=CFG.IN_CHANNELS, cond_dim=CFG.COND_DIM, ndf=CFG.NDF).to(CFG.DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=CFG.LR_G, betas=CFG.BETAS)
    optD = torch.optim.Adam(D.parameters(), lr=CFG.LR_D, betas=CFG.BETAS)

    for epoch in range(1, CFG.EPOCHS+1):
        G.train(); D.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CFG.EPOCHS}", ncols=110)
        for real, cond in pbar:
            real = real.to(CFG.DEVICE); cond = cond.to(CFG.DEVICE); B = real.size(0)

            # ---- D ----
            z = torch.randn(B, CFG.Z_DIM, device=CFG.DEVICE)
            with torch.no_grad(): fake = G(z, cond)
            real.requires_grad_(True)
            real_s, cond_pred_r = D(real, cond)
            fake_s, _ = D(fake.detach(), cond)
            d_adv = d_hinge_loss(real_s, fake_s)
            d_cond = F.mse_loss(cond_pred_r, cond) * CFG.LAMBDA_COND_D
            d_loss = d_adv + d_cond
            if CFG.LAMBDA_R1 > 0:
                r1 = r1_gradient_penalty(real_s, real) * (CFG.LAMBDA_R1*0.5)
                d_loss = d_loss + r1
            optD.zero_grad(set_to_none=True); d_loss.backward(); optD.step()

            # ---- G ----
            z = torch.randn(B, CFG.Z_DIM, device=CFG.DEVICE)
            fake = G(z, cond)
            fake_s, cond_pred_f = D(fake, cond)
            g_adv = g_hinge_loss(fake_s)
            g_cond = F.mse_loss(cond_pred_f, cond) * CFG.LAMBDA_COND_G
            g_pal = palette_loss_rgb(fake) * CFG.LAMBDA_PALETTE
            g_loss = g_adv + g_cond + g_pal
            optG.zero_grad(set_to_none=True); g_loss.backward(); optG.step()

            pbar.set_postfix({
                "D_adv": f"{d_adv.item():.3f}",
                "D_c": f"{d_cond.item():.3f}",
                "G_adv": f"{g_adv.item():.3f}",
                "G_c": f"{g_cond.item():.3f}",
                "G_pal": f"{g_pal.item():.3f}"
            })

        if epoch % CFG.SAMPLE_EVERY == 0:
            sample_grid(G, epoch, Path(CFG.DATA_ROOT), out_root/"samples", CFG.DEVICE)
        if epoch % CFG.SAVE_EVERY == 0:
            torch.save(G.state_dict(), out_root/"weights"/f"G_epoch{epoch:04d}.pt")
            torch.save(D.state_dict(), out_root/"weights"/f"D_epoch{epoch:04d}.pt")

    torch.save(G.state_dict(), out_root/"weights"/"G_final.pt")
    torch.save(D.state_dict(), out_root/"weights"/"D_final.pt")
    print("Running sucessful！Weight save：", (out_root/"weights").resolve())


# ===================== CLI =====================
def build_argparser():
    ap = argparse.ArgumentParser(description="MCCGAN")
    ap.add_argument("--data_root", type=str, default=CFG.DATA_ROOT)
    ap.add_argument("--output_root", type=str, default=CFG.OUTPUT_ROOT)
    ap.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    ap.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
