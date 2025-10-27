import os
from schedule.schedule import Schedule
from model.DocDiff import DocDiff, EMA
from schedule.diffusionSample import GaussianDiffusion
from schedule.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from schedule.edict_sampler import EDICTSampler
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import csv
import time
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import copy
from src.sobel import Laplacian


def init__result_Dir(experiment_name: str = None):
    work_dir = os.path.join(os.getcwd(), 'Training')
    # Ensure base directory exists
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    # If a name is provided, use it directly (unique per experiment)
    if experiment_name and len(str(experiment_name)) > 0:
        path = os.path.join(work_dir, str(experiment_name))
        os.makedirs(path, exist_ok=True)
        return path
    # Else, allocate next incremental index with timestamp suffix for uniqueness
    max_model = 0
    for root, j, file in os.walk(work_dir):
        for dirs in j:
            try:
                # allow names like "12" or "12_20250101_1200"
                head = dirs.split('_')[0]
                temp = int(head)
                if temp > max_model:
                    max_model = temp
            except:
                continue
        break
    max_model += 1
    import time as _time
    ts = _time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(work_dir, f"{max_model}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


class Trainer:
    def __init__(self, config):
        # Keep full config reference for saving effective YAML
        self._config_obj = config
        self.mode = config.MODE
        self.schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        in_channels = config.CHANNEL_X + config.CHANNEL_Y
        out_channels = config.CHANNEL_Y
        self.out_channels = out_channels
        self.network = DocDiff(
            input_channels=in_channels,
            output_channels=out_channels,
            n_channels=config.MODEL_CHANNELS,
            ch_mults=config.CHANNEL_MULT,
            n_blocks=config.NUM_RESBLOCKS
        ).to(self.device)
        self.diffusion = GaussianDiffusion(self.network.denoiser, config.TIMESTEPS, self.schedule).to(self.device)
        # EDICT sampler (deterministic, near-invertible)
        self.edict_sampler = EDICTSampler(self.network.denoiser, config.TIMESTEPS, self.schedule.get_betas()).to(self.device)
        self.test_img_save_path = config.TEST_IMG_SAVE_PATH
        if not os.path.exists(self.test_img_save_path):
            os.makedirs(self.test_img_save_path)
        self.pretrained_path_init_predictor = config.PRETRAINED_PATH_INITIAL_PREDICTOR
        self.pretrained_path_denoiser = config.PRETRAINED_PATH_DENOISER
        self.continue_training = config.CONTINUE_TRAINING
        self.continue_training_steps = 0
        self.path_train_gt = config.PATH_GT
        self.path_train_img = config.PATH_IMG
        self.iteration_max = config.ITERATION_MAX
        self.LR = config.LR
        self.cross_entropy = nn.BCELoss()
        self.num_timesteps = config.TIMESTEPS
        self.ema_every = config.EMA_EVERY
        self.start_ema = config.START_EMA
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.EMA_or_not = config.EMA
        self.weight_save_path = config.WEIGHT_SAVE_PATH
        self.experiment_name = getattr(config, 'EXPERIMENT_NAME', '')
        self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH = config.TEST_INITIAL_PREDICTOR_WEIGHT_PATH
        self.TEST_DENOISER_WEIGHT_PATH = config.TEST_DENOISER_WEIGHT_PATH
        self.DPM_SOLVER = config.DPM_SOLVER
        self.DPM_STEP = config.DPM_STEP
        # EDICT controls
        self.EDICT = getattr(config, 'EDICT', 'True')
        self.EDICT_P = getattr(config, 'EDICT_P', 0.93)
        self.EDICT_FP64 = getattr(config, 'EDICT_FP64', 'True')
        self.test_path_img = config.TEST_PATH_IMG
        self.test_path_gt = config.TEST_PATH_GT
        self.beta_loss = config.BETA_LOSS
        self.pre_ori = config.PRE_ORI
        self.high_low_freq = config.HIGH_LOW_FREQ
        self.image_size = config.IMAGE_SIZE
        self.native_resolution = config.NATIVE_RESOLUTION
        # Invertibility consistency loss controls
        self.edict_inv1_enabled = getattr(config, 'EDICT_INV1', 'True')
        self.lambda_inv1 = float(getattr(config, 'LAMBDA_INV1', 0.1))
        if self.mode == 1 and self.continue_training == 'True':
            print('Continue Training')
            self.network.init_predictor.load_state_dict(torch.load(self.pretrained_path_init_predictor))
            self.network.denoiser.load_state_dict(torch.load(self.pretrained_path_denoiser))
            self.continue_training_steps = config.CONTINUE_TRAINING_STEPS
        from data.docdata import DocData

        # ---------- Basic config sanity checks (paths, image size) ----------
        def _fmt_size(sz):
            try:
                return f"{int(sz[0])}x{int(sz[1])}" if isinstance(sz, (list, tuple)) else str(sz)
            except Exception:
                return str(sz)

        # IMAGE_SIZE check: expect [H, W] and multiples of 8 for UNet down/up-sampling
        if isinstance(self.image_size, (list, tuple)) and len(self.image_size) == 2:
            h_ok = (int(self.image_size[0]) % 8 == 0)
            w_ok = (int(self.image_size[1]) % 8 == 0)
            if not (h_ok and w_ok):
                print(f"[WARN] IMAGE_SIZE={_fmt_size(self.image_size)} is not multiple of 8; training will still run (RandomCrop), but UNet alignment may be suboptimal.")
        else:
            print(f"[WARN] IMAGE_SIZE format unexpected: {_fmt_size(self.image_size)} (expect [H, W]).")

        # Helper to summarize directory and file intersection status
        def _dir_ok(p: str) -> bool:
            try:
                return isinstance(p, str) and len(p) > 0 and os.path.isdir(p)
            except Exception:
                return False

        def _list_images(p: str):
            try:
                return sorted([f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))]) if _dir_ok(p) else []
            except Exception:
                return []

        # Train dataset summary
        train_img_list = _list_images(self.path_train_img)
        train_gt_list = _list_images(self.path_train_gt)
        train_pairs = sorted(list(set(train_img_list) & set(train_gt_list)))
        if not _dir_ok(self.path_train_img) or not _dir_ok(self.path_train_gt):
            print(f"[ERROR] Training paths invalid: PATH_IMG='{self.path_train_img}', PATH_GT='{self.path_train_gt}'")
        else:
            print(f"[DATA] Train IMG='{self.path_train_img}' ({len(train_img_list)} imgs) | GT='{self.path_train_gt}' ({len(train_gt_list)} imgs) | pairs={len(train_pairs)}")

        if self.mode == 1:
            dataset_train = DocData(self.path_train_img, self.path_train_gt, config.IMAGE_SIZE, self.mode)
            self.batch_size = config.BATCH_SIZE
            self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                               num_workers=config.NUM_WORKERS)
        else:
            dataset_test = DocData(config.TEST_PATH_IMG, config.TEST_PATH_GT, config.IMAGE_SIZE, self.mode,
                                   resize_test=(self.native_resolution != 'True'))
            self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                              drop_last=False,
                                              num_workers=config.NUM_WORKERS)
        if self.mode == 1 and config.EMA == 'True':
            self.EMA = EMA(0.9999)
            self.ema_model = copy.deepcopy(self.network).to(self.device)
        if config.LOSS == 'L1':
            self.loss = nn.L1Loss()
        elif config.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else:
            print('Loss not implemented, setting the loss to L2 (default one)')
            self.loss = nn.MSELoss()
        if self.high_low_freq == 'True':
            self.high_filter = Laplacian().to(self.device)

        # Validation config and loader (used during training)
        self.val_every = int(getattr(config, 'VAL_EVERY', 0) or 0)
        self.val_max_samples = int(getattr(config, 'VAL_MAX_SAMPLES', 16) or 16)
        self.save_best_by = str(getattr(config, 'SAVE_BEST_BY', 'psnr'))
        self.best_metric = None
        # Resolve validation paths
        val_gt = getattr(config, 'VAL_PATH_GT', None) or getattr(config, 'TEST_PATH_GT', None) or self.path_train_gt
        val_img = getattr(config, 'VAL_PATH_IMG', None) or getattr(config, 'TEST_PATH_IMG', None) or self.path_train_img

        # Validation dataset summary and loader creation with diagnostics
        val_img_list = _list_images(val_img)
        val_gt_list = _list_images(val_gt)
        val_pairs = sorted(list(set(val_img_list) & set(val_gt_list)))
        if self.val_every <= 0:
            print("[INFO] Validation disabled (VAL_EVERY <= 0).")
            self.dataloader_val = None
        else:
            if not _dir_ok(val_img) or not _dir_ok(val_gt):
                print(f"[WARN] Validation paths invalid: VAL_PATH_IMG='{val_img}', VAL_PATH_GT='{val_gt}'. Validation will be skipped.")
                self.dataloader_val = None
            elif len(val_pairs) == 0:
                print(f"[WARN] Validation has 0 paired samples (IMG='{val_img_list[:3]}', GT='{val_gt_list[:3]}'). Validation will run with N=0 and be skipped effectively.")
                try:
                    dataset_val = DocData(val_img, val_gt, config.IMAGE_SIZE, 0,
                                          resize_test=(self.native_resolution != 'True'))
                    self.dataloader_val = DataLoader(dataset_val, batch_size=getattr(config, 'BATCH_SIZE_VAL', 1), shuffle=False,
                                                     drop_last=False, num_workers=config.NUM_WORKERS)
                except Exception as e:
                    print(f"[WARN] Creating validation loader failed: {e}. Validation disabled.")
                    self.dataloader_val = None
            else:
                try:
                    dataset_val = DocData(val_img, val_gt, config.IMAGE_SIZE, 0,
                                          resize_test=(self.native_resolution != 'True'))
                    self.dataloader_val = DataLoader(dataset_val, batch_size=getattr(config, 'BATCH_SIZE_VAL', 1), shuffle=False,
                                                     drop_last=False, num_workers=config.NUM_WORKERS)
                    print(f"[DATA] Val IMG='{val_img}' ({len(val_img_list)} imgs) | GT='{val_gt}' ({len(val_gt_list)} imgs) | pairs={len(val_pairs)} | will validate every {self.val_every} iters (max {self.val_max_samples} samples)")
                except Exception as e:
                    print(f"[WARN] Creating validation loader failed: {e}. Validation disabled.")
                    self.dataloader_val = None

    def test(self):
        def crop_concat(img, size=128):
            shape = img.shape
            correct_shape = (size*(shape[2]//size+1), size*(shape[3]//size+1))
            one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]))
            one[:, :, :shape[2], :shape[3]] = img
            # crop
            for i in range(shape[2]//size+1):
                for j in range(shape[3]//size+1):
                    if i == 0 and j == 0:
                        crop = one[:, :, i*size:(i+1)*size, j*size:(j+1)*size]
                    else:
                        crop = torch.cat((crop, one[:, :, i*size:(i+1)*size, j*size:(j+1)*size]), dim=0)
            return crop
        def crop_concat_back(img, prediction, size=128):
            shape = img.shape
            for i in range(shape[2]//size+1):
                for j in range(shape[3]//size+1):
                    if j == 0:
                        crop = prediction[(i*(shape[3]//size+1)+j)*shape[0]:(i*(shape[3]//size+1)+j+1)*shape[0], :, :, :]
                    else:
                        crop = torch.cat((crop, prediction[(i*(shape[3]//size+1)+j)*shape[0]:(i*(shape[3]//size+1)+j+1)*shape[0], :, :, :]), dim=3)
                if i == 0:
                    crop_concat = crop
                else:
                    crop_concat = torch.cat((crop_concat, crop), dim=2)
            return crop_concat[:, :, :shape[2], :shape[3]]

        def min_max(array):
            return (array - array.min()) / (array.max() - array.min())
        with torch.no_grad():
            self.network.init_predictor.load_state_dict(torch.load(self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH))
            self.network.denoiser.load_state_dict(torch.load(self.TEST_DENOISER_WEIGHT_PATH))
            print('Test Model loaded')
            self.network.eval()
            tq = tqdm(self.dataloader_test)
            sampler = self.diffusion
            iteration = 0
            inv_mse_total, inv_cos_total, inv_count = 0.0, 0.0, 0
            for img, gt, name in tq:
                tq.set_description(f'Iteration {iteration} / {len(self.dataloader_test.dataset)}')
                iteration += 1
                # use IMAGE_SIZE as tile size when using native resolution tiling
                tile_size = int(self.image_size[0]) if isinstance(self.image_size, (list, tuple)) else int(self.image_size)
                if self.native_resolution == 'True':
                    temp = img
                    img = crop_concat(img, size=tile_size)
                noisyImage = torch.randn_like(img).to(self.device)
                init_predict = self.network.init_predictor(img.to(self.device), 0)

                if self.DPM_SOLVER == 'True':
                    sampledImgs = dpm_solver(self.schedule.get_betas(), self.network,
                                             torch.cat((noisyImage, img.to(self.device)), dim=1), self.DPM_STEP)
                elif self.EDICT == 'True':
                    sampledImgs = self.edict_sampler.sample(
                        noisyImage.to(self.device),
                        init_predict.to(self.device),
                        pre_ori=(self.pre_ori == 'True'),
                        p=float(self.EDICT_P),
                        use_fp64=(self.EDICT_FP64 == 'True')
                    )
                else:
                    sampledImgs = sampler(noisyImage.cuda(), init_predict, self.pre_ori)

                # Inversion accuracy: recover x_T from predicted residual using EDICT inversion
                if self.EDICT == 'True':
                    xT_rec = self.edict_sampler.invert(
                        x0_residual=sampledImgs.to(self.device),
                        cond=init_predict.to(self.device),
                        pre_ori=(self.pre_ori == 'True'),
                        p=float(self.EDICT_P),
                        use_fp64=(self.EDICT_FP64 == 'True')
                    )
                    mse = F.mse_loss(xT_rec, noisyImage.to(self.device)).item()
                    cos = torch.nn.functional.cosine_similarity(
                        xT_rec.flatten(1), noisyImage.flatten(1), dim=1
                    ).mean().item()
                    inv_mse_total += mse
                    inv_cos_total += cos
                    inv_count += 1
                finalImgs = (sampledImgs + init_predict)
                if self.native_resolution == 'True':
                    finalImgs = crop_concat_back(temp, finalImgs)
                    init_predict = crop_concat_back(temp, init_predict)
                    sampledImgs = crop_concat_back(temp, sampledImgs)
                    img = temp
                img_save = torch.cat((img, gt, init_predict.cpu(), min_max(sampledImgs.cpu()), finalImgs.cpu()), dim=3)
                save_image(img_save, os.path.join(
                    self.test_img_save_path, f"{name[0]}.png"), nrow=4)

            if inv_count > 0:
                print(f"EDICT inversion metrics over {inv_count} samples: MSE={inv_mse_total/inv_count:.6f}, CosSim={inv_cos_total/inv_count:.6f}")


    def train(self):
        optimizer = optim.AdamW(self.network.parameters(), lr=self.LR, weight_decay=1e-4)
        iteration = self.continue_training_steps
        save_img_path = init__result_Dir(self.experiment_name)
        # Per-run checkpoint directory inside weight root
        run_id = os.path.basename(save_img_path)
        weight_save_dir = os.path.join(self.weight_save_path, run_id)
        os.makedirs(weight_save_dir, exist_ok=True)
        # Save the effective config for reproducibility
        try:
            cfg_out = os.path.join(save_img_path, 'config.effective.yml')
            with open(cfg_out, 'w') as f:
                f.write(self._config_obj.to_yaml() if hasattr(self._config_obj, 'to_yaml') else '')
        except Exception:
            pass
        # Maintain a convenient symlink Training/latest -> this run (best-effort)
        try:
            latest_link = os.path.join(os.path.dirname(save_img_path), 'latest')
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                try:
                    os.remove(latest_link)
                except Exception:
                    pass
            os.symlink(save_img_path, latest_link)
        except Exception:
            pass
        print('Starting Training', f"Step is {self.num_timesteps}")
        print(f"Run dir: {save_img_path} | Checkpoints: {weight_save_dir}")
        # Metrics CSV in this run directory
        metrics_csv = os.path.join(save_img_path, 'metrics.csv')
        if not os.path.exists(metrics_csv):
            with open(metrics_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'loss', 'ddpm_loss',
                    'pixel_total', 'pixel_plain', 'low_freq_pixel_loss',
                    'L_inv1', 'lr', 'step_time_sec'
                ])

        while iteration < self.iteration_max:

            tq = tqdm(self.dataloader_train)

            for img, gt, name in tq:
                tq.set_description(f'Iteration {iteration} / {self.iteration_max}')
                self.network.train()
                optimizer.zero_grad()
                step_start = time.time()
                t = torch.randint(0, self.num_timesteps, (img.shape[0],)).long().to(self.device)
                init_predict, noise_pred, noisy_image, noise_ref = self.network(gt.to(self.device), img.to(self.device),
                                                                                t, self.diffusion)
                if self.pre_ori == 'True':
                    if self.high_low_freq == 'True':
                        residual_high = self.high_filter(gt.to(self.device) - init_predict)
                        ddpm_loss = 2*self.loss(self.high_filter(noise_pred), residual_high) + self.loss(noise_pred, gt.to(self.device) - init_predict)
                    else:
                        ddpm_loss = self.loss(noise_pred, gt.to(self.device) - init_predict)
                else:
                    ddpm_loss = self.loss(noise_pred, noise_ref.to(self.device))
                # Pixel losses (coarse predictor)
                pixel_total = None
                pixel_plain = None
                low_freq_loss = None
                if self.high_low_freq == 'True':
                    pixel_plain = self.loss(init_predict, gt.to(self.device))
                    low_freq_loss = self.loss(init_predict - self.high_filter(init_predict), gt.to(self.device) - self.high_filter(gt.to(self.device)))
                    pixel_total = pixel_plain + 2*low_freq_loss
                else:
                    pixel_plain = self.loss(init_predict, gt.to(self.device))
                    pixel_total = pixel_plain
                # One-step invertibility consistency loss (EDICT) when predicting x0
                L_inv1 = torch.tensor(0.0, device=self.device)
                if self.pre_ori == 'True' and self.edict_inv1_enabled == 'True':
                    # Only compute for t < T-1 to ensure t+1 exists
                    mask = (t < (self.num_timesteps - 1))
                    if mask.any():
                        # Gather coefficients
                        abar = self.diffusion.gammas.to(self.device)  # alphas_cumprod, shape [T], possibly float64
                        # Cast to model dtype for training graph consistency
                        abar_t = abar.gather(0, t.clamp_min(0)).view(-1, 1, 1, 1).to(noisy_image.dtype)
                        abar_t1 = abar.gather(0, (t + 1).clamp_max(self.num_timesteps - 1)).view(-1, 1, 1, 1).to(noisy_image.dtype)
                        # True x0 and epsilon
                        x0_true = (gt.to(self.device) - init_predict).detach()  # avoid leaking pixel loss grads
                        eps_true = noise_ref
                        # Construct x_{t+1}
                        x_t1 = abar_t1.sqrt() * x0_true + (1.0 - abar_t1).clamp(min=1e-12).sqrt() * eps_true
                        # Epsilon estimate from predicted x0
                        eps_hat = (noisy_image - abar_t.sqrt() * noise_pred) / (1.0 - abar_t).clamp(min=1e-12).sqrt()
                        # EDICT forward one step from x_t
                        x_t1_hat = self.edict_sampler.forward_one_step(noisy_image, eps_hat, t, p=float(self.EDICT_P))
                        # Compute MSE only on valid items (t < T-1)
                        valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                        if valid_idx.numel() > 0:
                            # Ensure dtype consistency for loss
                            L_inv1 = F.mse_loss(
                                x_t1_hat.index_select(0, valid_idx).to(noisy_image.dtype),
                                x_t1.index_select(0, valid_idx).to(noisy_image.dtype),
                            )

                loss = ddpm_loss + self.beta_loss * (pixel_total) / self.num_timesteps + self.lambda_inv1 * L_inv1.float()
                loss.backward()
                optimizer.step()
                step_time = time.time() - step_start
                # Richer TQDM postfix
                if self.high_low_freq == 'True':
                    tq.set_postfix(
                        loss=loss.item(), ddpm_loss=ddpm_loss.item(),
                        pixel_total=pixel_total.item(), pixel_plain=pixel_plain.item(), low_freq_pixel_loss=low_freq_loss.item(),
                        inv1=L_inv1.item(), t_step=f"{step_time:.2f}s"
                    )
                else:
                    tq.set_postfix(
                        loss=loss.item(), ddpm_loss=ddpm_loss.item(),
                        pixel_total=pixel_total.item(), pixel_plain=pixel_plain.item(),
                        inv1=L_inv1.item(), t_step=f"{step_time:.2f}s"
                    )
                # Persist metrics per-iteration
                try:
                    with open(metrics_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            iteration, loss.item(), ddpm_loss.item(),
                            pixel_total.item() if pixel_total is not None else '',
                            pixel_plain.item() if pixel_plain is not None else '',
                            low_freq_loss.item() if low_freq_loss is not None else '',
                            L_inv1.item() if isinstance(L_inv1, torch.Tensor) else float(L_inv1),
                            optimizer.param_groups[0]['lr'], step_time
                        ])
                except Exception:
                    pass

                # Periodic validation: trigger on the next iteration number (post-update)
                iter_next = iteration + 1
                if self.val_every > 0 and iter_next % self.val_every == 0 and self.dataloader_val is not None:
                    val_csv = os.path.join(save_img_path, 'val_metrics.csv')
                    psnr_mean, ssim_mean, inv_mse_mean, inv_cos_mean, n_eval = self.validate(max_samples=self.val_max_samples)
                    # Print concise metrics for logs
                    print(f"VAL iter={iter_next}: PSNR={psnr_mean:.4f} SSIM={ssim_mean:.4f} | INV_MSE={inv_mse_mean:.6f} INV_COS={inv_cos_mean:.6f} (N={n_eval})")
                    # Append to CSV
                    try:
                        if not os.path.exists(val_csv):
                            with open(val_csv, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(['iteration', 'psnr', 'ssim', 'inv_mse', 'inv_cossim', 'n'])
                        with open(val_csv, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([iter_next, psnr_mean, ssim_mean, inv_mse_mean, inv_cos_mean, n_eval])
                    except Exception:
                        pass
                    # Save best checkpoint by selected metric
                    key = self.save_best_by.lower()
                    metric_val = {
                        'psnr': psnr_mean,
                        'ssim': ssim_mean,
                        'inv_mse': inv_mse_mean,
                        'inv_cos': inv_cos_mean,
                    }.get(key, psnr_mean)
                    better = False
                    if self.best_metric is None:
                        better = True
                    else:
                        if key in ('psnr', 'ssim', 'inv_cos'):
                            better = metric_val > self.best_metric
                        else:  # inv_mse smaller is better
                            better = metric_val < self.best_metric
                    if better:
                        self.best_metric = metric_val
                        # Save current weights as best
                    if not os.path.exists(weight_save_dir):
                        os.makedirs(weight_save_dir)
                    torch.save(self.network.init_predictor.state_dict(),
                               os.path.join(weight_save_dir, f'model_init_best_{key}.pth'))
                    torch.save(self.network.denoiser.state_dict(),
                               os.path.join(weight_save_dir, f'model_denoiser_best_{key}.pth'))
                    if self.EMA_or_not == 'True':
                        torch.save(self.ema_model.init_predictor.state_dict(),
                                   os.path.join(weight_save_dir, f'model_init_ema_best_{key}.pth'))
                        torch.save(self.ema_model.denoiser.state_dict(),
                                   os.path.join(weight_save_dir, f'model_denoiser_ema_best_{key}.pth'))
                if iteration % 500 == 0:
                    if not os.path.exists(save_img_path):
                        os.makedirs(save_img_path)
                    img_save = torch.cat([img, gt, init_predict.cpu()], dim=3)
                    if self.pre_ori == 'True':
                        if self.high_low_freq == 'True':
                            img_save = torch.cat([img, gt, init_predict.cpu(), noise_pred.cpu() + self.high_filter(init_predict).cpu(), noise_pred.cpu() + init_predict.cpu()], dim=3)
                        else:
                            img_save = torch.cat([img, gt, init_predict.cpu(), noise_pred.cpu() + init_predict.cpu()], dim=3)
                    save_image(img_save, os.path.join(
                        save_img_path, f"{iteration}.png"), nrow=4)
                iteration += 1
                if self.EMA_or_not == 'True':
                    if iteration % self.ema_every == 0 and iteration > self.start_ema:
                        print('EMA update')
                        self.EMA.update_model_average(self.ema_model, self.network)

                if iteration % self.save_model_every == 0:
                    print('Saving models')
                    if not os.path.exists(weight_save_dir):
                        os.makedirs(weight_save_dir)
                    torch.save(self.network.init_predictor.state_dict(),
                               os.path.join(weight_save_dir, f'model_init_{iteration}.pth'))
                    torch.save(self.network.denoiser.state_dict(),
                               os.path.join(weight_save_dir, f'model_denoiser_{iteration}.pth'))
                    if self.EMA_or_not == 'True':
                        torch.save(self.ema_model.init_predictor.state_dict(),
                                   os.path.join(weight_save_dir, f'model_init_ema_{iteration}.pth'))
                        torch.save(self.ema_model.denoiser.state_dict(),
                                   os.path.join(weight_save_dir, f'model_denoiser_ema_{iteration}.pth'))
                    # Also update latest pointers
                    torch.save(self.network.init_predictor.state_dict(),
                               os.path.join(weight_save_dir, f'model_init_latest.pth'))
                    torch.save(self.network.denoiser.state_dict(),
                               os.path.join(weight_save_dir, f'model_denoiser_latest.pth'))
                    if self.EMA_or_not == 'True':
                        torch.save(self.ema_model.init_predictor.state_dict(),
                                   os.path.join(weight_save_dir, f'model_init_ema_latest.pth'))
                        torch.save(self.ema_model.denoiser.state_dict(),
                                   os.path.join(weight_save_dir, f'model_denoiser_ema_latest.pth'))

    @torch.no_grad()
    def _compute_psnr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # x,y in [0,1]
        mse = F.mse_loss(x, y).item()
        if mse == 0:
            return 100.0
        return 10.0 * float(torch.log10(torch.tensor(1.0 / mse)))

    @torch.no_grad()
    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # Simple SSIM implementation with Gaussian window, works on [0,1]
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        # Gaussian window 11x11
        def _gaussian(window_size=11, sigma=1.5):
            coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = (g / g.sum()).unsqueeze(1)
            win = g @ g.t()
            return win
        win = _gaussian(11, 1.5)
        win = win.expand(x.shape[1], 1, 11, 11).contiguous()
        conv = F.conv2d
        mu_x = conv(x, win, padding=5, groups=x.shape[1])
        mu_y = conv(y, win, padding=5, groups=y.shape[1])
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = conv(x * x, win, padding=5, groups=x.shape[1]) - mu_x2
        sigma_y2 = conv(y * y, win, padding=5, groups=y.shape[1]) - mu_y2
        sigma_xy = conv(x * y, win, padding=5, groups=x.shape[1]) - mu_xy
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        return ssim_map.mean().item()

    @torch.no_grad()
    def validate(self, max_samples: int = 16):
        if self.dataloader_val is None:
            return 0.0, 0.0, 0.0, 0.0, 0
        self.network.eval()
        psnr_sum = 0.0
        ssim_sum = 0.0
        inv_mse_sum = 0.0
        inv_cos_sum = 0.0
        n = 0
        # helper tiling functions (mirror test())
        def crop_concat(img, size=128):
            shape = img.shape
            correct_shape = (size * (shape[2] // size + 1), size * (shape[3] // size + 1))
            one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]), device=img.device, dtype=img.dtype)
            one[:, :, :shape[2], :shape[3]] = img
            # crop
            tiles = []
            for i in range(shape[2] // size + 1):
                for j in range(shape[3] // size + 1):
                    tiles.append(one[:, :, i * size:(i + 1) * size, j * size:(j + 1) * size])
            return torch.cat(tiles, dim=0)

        def crop_concat_back(img, prediction, size=128):
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

        tile_size = int(self.image_size[0]) if isinstance(self.image_size, (list, tuple)) else int(self.image_size)

        for img, gt, _ in self.dataloader_val:
            if n >= max_samples:
                break
            img = img.to(self.device)
            gt = gt.to(self.device)
            # Inject noise
            noisyImage = torch.randn_like(img)

            # Tiling if needed to satisfy network down/up-sampling grid
            H, W = img.shape[2], img.shape[3]
            do_tile = (H % tile_size != 0) or (W % tile_size != 0) or (H != tile_size or W != tile_size)
            if do_tile:
                img_src = img
                img = crop_concat(img, size=tile_size)
                noisyImage = torch.randn_like(img)

            # Coarse prediction
            init_predict = self.network.init_predictor(img, 0)
            # Residual sampling
            if self.DPM_SOLVER == 'True':
                sampled = dpm_solver(self.schedule.get_betas(), self.network,
                                     torch.cat((noisyImage, img), dim=1), self.DPM_STEP)
            elif self.EDICT == 'True':
                sampled = self.edict_sampler.sample(noisyImage, init_predict, pre_ori=(self.pre_ori == 'True'),
                                                    p=float(self.EDICT_P), use_fp64=(self.EDICT_FP64 == 'True'))
            else:
                sampled = self.diffusion(noisyImage, init_predict, self.pre_ori)
            # Final reconstruction
            recon = (sampled + init_predict).clamp(0, 1)
            if do_tile:
                recon = crop_concat_back(img_src, recon, size=tile_size)
            # Metrics
            psnr_sum += self._compute_psnr(recon, gt)
            ssim_sum += self._compute_ssim(recon, gt)
            # Inversion metrics (EDICT only)
            if self.EDICT == 'True':
                xT_rec = self.edict_sampler.invert(sampled, cond=init_predict, pre_ori=(self.pre_ori == 'True'),
                                                   p=float(self.EDICT_P), use_fp64=(self.EDICT_FP64 == 'True'))
                inv_mse_sum += F.mse_loss(xT_rec, noisyImage).item()
                inv_cos_sum += torch.nn.functional.cosine_similarity(
                    xT_rec.flatten(1), noisyImage.flatten(1), dim=1
                ).mean().item()
            n += 1
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0
        return psnr_sum / n, ssim_sum / n, inv_mse_sum / max(n, 1), inv_cos_sum / max(n, 1), n



def dpm_solver(betas, model, x_T, steps, model_kwargs):
    # You need to firstly define your model and the extra inputs of your model,
    # And initialize an `x_T` from the standard normal distribution.
    # `model` has the format: model(x_t, t_input, **model_kwargs).
    # If your model has no extra inputs, just let model_kwargs = {}.

    # If you use discrete-time DPMs, you need to further define the
    # beta arrays for the noise schedule.

    # model = ....
    # model_kwargs = {...}
    # x_T = ...
    # betas = ....

    # 1. Define the noise schedule.
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

    # 2. Convert your discrete-time `model` to the continuous-time
    # noise prediction model. Here is an example for a diffusion model
    # `model` with the noise prediction type ("noise") .
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
    )

    # 3. Define dpm-solver and sample by singlestep DPM-Solver.
    # (We recommend singlestep DPM-Solver for unconditional sampling)
    # You can adjust the `steps` to balance the computation
    # costs and the sample quality.
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                            correcting_x0_fn="dynamic_thresholding")
    # Can also try
    # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

    # You can use steps = 10, 12, 15, 20, 25, 50, 100.
    # Empirically, we find that steps in [10, 20] can generate quite good samples.
    # And steps = 20 can almost converge.
    x_sample = dpm_solver.sample(
        x_T,
        steps=steps,
        order=1,
        skip_type="time_uniform",
        method="singlestep",
    )
    return x_sample
