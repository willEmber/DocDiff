import os
import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from src.config import load_config
from model.DocDiff import DocDiff
from data.docdata import DocData
from schedule.schedule import Schedule
from schedule.edict_sampler import EDICTSampler


def _to_bool_str(s: str) -> bool:
    if isinstance(s, bool):
        return s
    s = str(s).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


@torch.no_grad()
def _compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return 100.0
    return 10.0 * float(torch.log10(torch.tensor(1.0 / mse)))


@torch.no_grad()
def _compute_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    def _gaussian(window_size=11, sigma=1.5, device=None, dtype=None):
        coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(1)
        win = g @ g.t()
        return win

    win = _gaussian(11, 1.5, device=x.device, dtype=x.dtype)
    win = win.expand(x.shape[1], 1, 11, 11).contiguous()
    conv = F.conv2d
    mu_x = conv(x, win, padding=5, groups=x.shape[1])
    mu_y = conv(y, win, padding=5, groups=x.shape[1])
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = conv(x * x, win, padding=5, groups=x.shape[1]) - mu_x2
    sigma_y2 = conv(y * y, win, padding=5, groups=x.shape[1]) - mu_y2
    sigma_xy = conv(x * y, win, padding=5, groups=x.shape[1]) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean().item()


def _crop_concat(img: torch.Tensor, size: int = 128) -> torch.Tensor:
    shape = img.shape
    correct_shape = (size * (shape[2] // size + 1), size * (shape[3] // size + 1))
    one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]), device=img.device, dtype=img.dtype)
    one[:, :, :shape[2], :shape[3]] = img
    tiles = []
    for i in range(shape[2] // size + 1):
        for j in range(shape[3] // size + 1):
            tiles.append(one[:, :, i * size:(i + 1) * size, j * size:(j + 1) * size])
    return torch.cat(tiles, dim=0)


def _crop_concat_back(img: torch.Tensor, prediction: torch.Tensor, size: int = 128) -> torch.Tensor:
    shape = img.shape
    rows = []
    idx = 0
    for i in range(shape[2] // size + 1):
        row = []
        for j in range(shape[3] // size + 1):
            row.append(prediction[idx * shape[0]:(idx + 1) * shape[0], :, :, :])
            idx += 1
        rows.append(torch.cat(row, dim=3))
    out = torch.cat(rows, dim=2)
    return out[:, :, :shape[2], :shape[3]]


def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    # per-tensor min-max normalization for visualization
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    denom = (x_max - x_min).clamp_min(1e-6)
    return (x - x_min) / denom


def _save_labeled_strip(panels, labels, out_path, label_bar: int = 24):
    """Save a horizontal strip of panels with a text label above each panel.
    panels: list of torch.Tensor in [0,1], shapes [1,3,H,W] or [3,H,W]
    labels: list[str], same length as panels
    """
    assert len(panels) == len(labels)
    pil_panels = []
    widths, heights = [], []
    for p in panels:
        if isinstance(p, torch.Tensor):
            t = p.detach().cpu()
            if t.dim() == 4:
                t = t.squeeze(0)
            t = t.clamp(0, 1)
            pil = TF.to_pil_image(t)
        elif isinstance(p, Image.Image):
            pil = p
        else:
            raise TypeError("Panel must be Tensor or PIL.Image")
        pil_panels.append(pil)
        widths.append(pil.width)
        heights.append(pil.height)
    total_w = int(sum(widths))
    total_h = int(max(heights) + label_bar)
    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x = 0
    for pil, label, w in zip(pil_panels, labels, widths):
        # Paste panel below the label bar
        canvas.paste(pil, (x, label_bar))
        # Draw centered text
        if font is not None:
            try:
                text_w, text_h = draw.textsize(label, font=font)
            except Exception:
                text_w, text_h = (len(label) * 6, 11)
        else:
            text_w, text_h = (len(label) * 6, 11)
        tx = x + max(0, (w - text_w) // 2)
        ty = max(0, (label_bar - text_h) // 2)
        # outline then white text for readability
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
        x += w
    canvas.save(out_path)


# ---- Stego helpers (torch) ----
def _stego_keygen_codebook_torch(K: int, d: int, device, dtype, seed: int, normalize: bool = False):
    """Gaussian codebook U ~ N(0, I) with shape (K, d). Optionally L2-normalize rows.
    normalize=False is recommended to strictly preserve N(0, I) when encoding.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    U = torch.randn((K, d), device=device, dtype=dtype, generator=g)
    if normalize:
        U = torch.nn.functional.normalize(U, p=2, dim=1)
    return U


def _stego_encode_permutation_torch(U: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Row selection encoding. U: (K, d); m: (L,) in [0, K-1]; returns Z0: (L, d)."""
    return U.index_select(dim=0, index=m)


def _pack_rows_to_xT(Z0: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
    """Z0: (L=H, d=C*W) -> x_T: (C, H, W)"""
    return Z0.view(H, C, W).permute(1, 0, 2).contiguous()


def _unpack_xT_to_rows(x_T: torch.Tensor) -> torch.Tensor:
    """x_T: (C, H, W) -> Z: (L=H, d=C*W)"""
    C, H, W = x_T.shape
    return x_T.permute(1, 0, 2).contiguous().view(H, C * W)


def _stego_decode_cosine_torch(U: torch.Tensor, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (m_hat, margin_mean, margin_min)."""
    U_n = torch.nn.functional.normalize(U, p=2, dim=1)
    Z_n = torch.nn.functional.normalize(Z, p=2, dim=1)
    S = Z_n @ U_n.t()  # (L, K)
    m_hat = S.argmax(dim=1)
    L = Z.shape[0]
    idx = torch.arange(L, device=Z.device)
    true_scores = S[idx, m_hat]  # use predicted for robustness of margin; could also use GT later
    # For margin wrt GT, caller can recompute with true indices if needed
    return m_hat, true_scores.mean(dim=0), true_scores.min(dim=0).values


def _stego_margin_with_gt(U: torch.Tensor, Z: torch.Tensor, m_true: torch.Tensor) -> Tuple[float, float]:
    U_n = torch.nn.functional.normalize(U, p=2, dim=1)
    Z_n = torch.nn.functional.normalize(Z, p=2, dim=1)
    S = Z_n @ U_n.t()  # (L, K)
    L = Z.shape[0]
    idx = torch.arange(L, device=Z.device)
    true_scores = S[idx, m_true]
    S_mask = S.clone()
    S_mask[idx, m_true] = -float("inf")
    other = S_mask.max(dim=1).values
    margin = true_scores - other
    return float(margin.mean().item()), float(margin.min().item())


def main():
    parser = argparse.ArgumentParser(description="EDICT inversion visualization + stego encode/decode")
    parser.add_argument("--config", type=str, default="conf.yml", help="Path to config YAML")
    parser.add_argument("--weights-init", type=str, default="", help="Path to init_predictor weights (.pth)")
    parser.add_argument("--weights-denoiser", type=str, default="", help="Path to denoiser weights (.pth)")
    parser.add_argument("--img-dir", type=str, default="", help="Directory of input images")
    parser.add_argument("--gt-dir", type=str, default="", help="Directory of GT images")
    parser.add_argument("--out-dir", type=str, default="results/edict_viz", help="Output directory")
    parser.add_argument("--num", type=int, default=8, help="Max number of samples to visualize")
    parser.add_argument("--native-resolution", type=str, default="", help="Override NATIVE_RESOLUTION True/False")
    parser.add_argument("--tile-size", type=int, default=0, help="Tile size if using native resolution tiling")
    parser.add_argument("--edict-p", type=float, default=-1.0, help="EDICT coupling p (0<p<=1)")
    parser.add_argument("--edict-fp64", type=str, default="", help="Use float64 math in EDICT True/False")
    parser.add_argument("--pre-ori", type=str, default="", help="Model predicts x0 ('True') or eps ('False')")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise")

    # Stego-specific options
    parser.add_argument("--stego", type=str, default="False", help="Enable stego encode/decode True/False")
    parser.add_argument("--stego-K", type=int, default=256, help="Alphabet size K (default 256)")
    parser.add_argument("--stego-seed", type=int, default=0, help="Seed for stego codebook/message")
    parser.add_argument("--stego-message", type=str, default="random", help="'random' or 'seq'")
    parser.add_argument("--stego-sigma", type=float, default=0.0, help="Optional tiny jitter added to Z0 rows")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # Resolve paths and flags from args or config
    weights_init = args.weights_init or getattr(cfg, 'TEST_INITIAL_PREDICTOR_WEIGHT_PATH', '') or getattr(cfg, 'PRETRAINED_PATH_INITIAL_PREDICTOR', '')
    weights_deno = args.weights_denoiser or getattr(cfg, 'TEST_DENOISER_WEIGHT_PATH', '') or getattr(cfg, 'PRETRAINED_PATH_DENOISER', '')
    img_dir = args.img_dir or getattr(cfg, 'VAL_PATH_IMG', '') or getattr(cfg, 'TEST_PATH_IMG', '') or getattr(cfg, 'PATH_IMG', '')
    gt_dir = args.gt_dir or getattr(cfg, 'VAL_PATH_GT', '') or getattr(cfg, 'TEST_PATH_GT', '') or getattr(cfg, 'PATH_GT', '')
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(weights_init) or not os.path.isfile(weights_deno):
        raise FileNotFoundError(f"Weights not found. init='{weights_init}', denoiser='{weights_deno}'")
    if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"Image/GT dirs invalid. img='{img_dir}', gt='{gt_dir}'")

    native_res = _to_bool_str(args.native_resolution) if args.native_resolution else _to_bool_str(getattr(cfg, 'NATIVE_RESOLUTION', 'False'))
    pre_ori = _to_bool_str(args.pre_ori) if args.pre_ori else _to_bool_str(getattr(cfg, 'PRE_ORI', 'True'))
    p = args.edict_p if args.edict_p > 0 else float(getattr(cfg, 'EDICT_P', 0.93))
    use_fp64 = _to_bool_str(args.edict_fp64) if args.edict_fp64 else _to_bool_str(getattr(cfg, 'EDICT_FP64', 'True'))
    image_size = getattr(cfg, 'IMAGE_SIZE', [128, 128])
    tile_size = args.tile_size if args.tile_size > 0 else (int(image_size[0]) if isinstance(image_size, (list, tuple)) else int(image_size))

    # Seed
    if args.seed:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    stego_enabled = _to_bool_str(args.stego)

    # Build model and sampler
    net = DocDiff(
        input_channels=getattr(cfg, 'CHANNEL_X', 3) + getattr(cfg, 'CHANNEL_Y', 3),
        output_channels=getattr(cfg, 'CHANNEL_Y', 3),
        n_channels=getattr(cfg, 'MODEL_CHANNELS', 32),
        ch_mults=getattr(cfg, 'CHANNEL_MULT', [1, 2, 3, 4]),
        n_blocks=getattr(cfg, 'NUM_RESBLOCKS', 1),
    ).to(device)
    net.init_predictor.load_state_dict(torch.load(weights_init, map_location=device))
    net.denoiser.load_state_dict(torch.load(weights_deno, map_location=device))
    net.eval()

    schedule = Schedule(getattr(cfg, 'SCHEDULE', 'linear'), getattr(cfg, 'TIMESTEPS', 100))
    edict = EDICTSampler(net.denoiser, getattr(cfg, 'TIMESTEPS', 100), schedule.get_betas()).to(device)

    # Data
    dataset = DocData(img_dir, gt_dir, image_size, mode=0, resize_test=(not native_res))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # CSV metrics
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('name,psnr,ssim,inv_mse,inv_cos,stego_acc,stego_margin_mean,stego_margin_min\n')

    count = 0
    for img, gt, name in loader:
        if count >= args.num:
            break
        img = img.to(device)
        gt = gt.to(device)

        # Prepare tiling if needed
        H, W = img.shape[2], img.shape[3]
        do_tile = native_res and ((H % tile_size != 0) or (W % tile_size != 0) or (H != tile_size or W != tile_size))
        if do_tile:
            img_src = img
            img = _crop_concat(img, size=tile_size)

        # Sample residual with EDICT
        noisy = torch.randn_like(img)

        # ----- Stego encode on x_T (noisy) -----
        stego_acc = float('nan')
        stego_margin_mean = float('nan')
        stego_margin_min = float('nan')
        if stego_enabled:
            assert noisy.shape[0] == 1, "Stego path assumes batch_size=1"
            _, Cn, Hn, Wn = noisy.shape
            L = Hn
            d = Cn * Wn
            K = int(args.stego_K)
            # Codebook and message
            U = _stego_keygen_codebook_torch(K, d, device=noisy.device, dtype=noisy.dtype, seed=args.stego_seed + count, normalize=False)
            if args.stego_message.lower() == 'seq':
                m = torch.arange(L, device=noisy.device) % K
            else:
                g_msg = torch.Generator(device=noisy.device)
                g_msg.manual_seed(int(args.stego_seed + 0x9E3779B1 + count))
                m = torch.randint(low=0, high=K, size=(L,), device=noisy.device, generator=g_msg)
            Z0 = _stego_encode_permutation_torch(U, m)
            if args.stego_sigma and args.stego_sigma > 0:
                Z0 = Z0 + torch.randn_like(Z0) * float(args.stego_sigma)
            noisy = _pack_rows_to_xT(Z0, Cn, Hn, Wn)

        init_pred = net.init_predictor(img, torch.zeros(img.shape[0], dtype=torch.long, device=img.device))
        sampled = edict.sample(noisy, init_pred, pre_ori=pre_ori, p=p, use_fp64=use_fp64)
        recon = (sampled + init_pred).clamp(0, 1)

        # Inversion: recover x_T from sampled residual
        xT_rec = edict.invert(sampled, cond=init_pred, pre_ori=pre_ori, p=p, use_fp64=use_fp64)

        # ----- Stego decode (before untile to match encoding shape) -----
        if stego_enabled:
            # Decode on the same shape as encoding
            xT_for_decode = xT_rec
            Z_hat = _unpack_xT_to_rows(xT_for_decode.squeeze(0))  # (H, C*W); batch=1
            # Nearest neighbor (cosine) and margins wrt GT indices
            U_n = torch.nn.functional.normalize(U, p=2, dim=1)
            Z_n = torch.nn.functional.normalize(Z_hat, p=2, dim=1)
            S = Z_n @ U_n.t()
            m_hat = S.argmax(dim=1)
            stego_acc = float((m_hat == m).float().mean().item())
            idx = torch.arange(Z_hat.shape[0], device=Z_hat.device)
            true_scores = S[idx, m]
            S_mask = S.clone()
            S_mask[idx, m] = -float('inf')
            other = S_mask.max(dim=1).values
            margin = true_scores - other
            stego_margin_mean = float(margin.mean().item())
            stego_margin_min = float(margin.min().item())

        # Untile back if needed
        if do_tile:
            recon = _crop_concat_back(img_src, recon, size=tile_size)
            init_pred = _crop_concat_back(img_src, init_pred, size=tile_size)
            sampled = _crop_concat_back(img_src, sampled, size=tile_size)
            noisy_full = _crop_concat_back(img_src, noisy, size=tile_size)
            xT_rec = _crop_concat_back(img_src, xT_rec, size=tile_size)
            img = img_src
        else:
            noisy_full = noisy

        # Metrics
        psnr = _compute_psnr(recon, gt)
        ssim = _compute_ssim(recon, gt)
        inv_mse = F.mse_loss(xT_rec, noisy_full).item()
        inv_cos = torch.nn.functional.cosine_similarity(xT_rec.flatten(1), noisy_full.flatten(1), dim=1).mean().item()

        # Save labeled visualization strips
        residual_vis = _minmax_norm(sampled.detach().cpu())
        recon_path = os.path.join(out_dir, f"{name[0]}_recon.png")
        _save_labeled_strip(
            panels=[img.cpu().clamp(0, 1), gt.cpu().clamp(0, 1), init_pred.cpu().clamp(0, 1), residual_vis, recon.cpu().clamp(0, 1)],
            labels=["Input", "GT", "InitPred", "Residual(norm)", "Recon"],
            out_path=recon_path,
        )

        xT_vis = _minmax_norm(noisy_full.detach().cpu())
        xT_rec_vis = _minmax_norm(xT_rec.detach().cpu())
        diff_vis = _minmax_norm((xT_rec.detach().cpu() - noisy_full.detach().cpu()).abs())
        noise_path = os.path.join(out_dir, f"{name[0]}_noise.png")
        _save_labeled_strip(
            panels=[xT_vis, xT_rec_vis, diff_vis],
            labels=["x_T(norm)", "xT_rec(norm)", "|diff|(norm)"],
            out_path=noise_path,
        )

        # Persist metrics including stego
        with open(csv_path, 'a') as f:
            f.write(f"{name[0]},{psnr:.4f},{ssim:.4f},{inv_mse:.6f},{inv_cos:.6f},{stego_acc:.6f},{stego_margin_mean:.6f},{stego_margin_min:.6f}\n")

        print(
            f"{name[0]}: PSNR={psnr:.3f} SSIM={ssim:.3f} | "
            f"INV_MSE={inv_mse:.6f} INV_COS={inv_cos:.6f} | "
            f"StegoAcc={stego_acc:.3f} Margin(mean/min)={stego_margin_mean:.3f}/{stego_margin_min:.3f}"
        )
        count += 1


if __name__ == "__main__":
    main()

