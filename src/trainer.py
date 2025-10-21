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
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import copy
from src.sobel import Laplacian


def init__result_Dir():
    work_dir = os.path.join(os.getcwd(), 'Training')
    max_model = 0
    for root, j, file in os.walk(work_dir):
        for dirs in j:
            try:
                temp = int(dirs)
                if temp > max_model:
                    max_model = temp
            except:
                continue
        break
    max_model += 1
    path = os.path.join(work_dir, str(max_model))
    os.mkdir(path)
    return path


class Trainer:
    def __init__(self, config):
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
        if self.mode == 1:
            dataset_train = DocData(self.path_train_img, self.path_train_gt, config.IMAGE_SIZE, self.mode)
            self.batch_size = config.BATCH_SIZE
            self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                               num_workers=config.NUM_WORKERS)
        else:
            dataset_test = DocData(config.TEST_PATH_IMG, config.TEST_PATH_GT, config.IMAGE_SIZE, self.mode)
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
                if self.native_resolution == 'True':
                    temp = img
                    img = crop_concat(img)
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
        save_img_path = init__result_Dir()
        print('Starting Training', f"Step is {self.num_timesteps}")

        while iteration < self.iteration_max:

            tq = tqdm(self.dataloader_train)

            for img, gt, name in tq:
                tq.set_description(f'Iteration {iteration} / {self.iteration_max}')
                self.network.train()
                optimizer.zero_grad()

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
                if self.high_low_freq == 'True':
                    low_high_loss = self.loss(init_predict, gt.to(self.device))
                    low_freq_loss = self.loss(init_predict - self.high_filter(init_predict), gt.to(self.device) - self.high_filter(gt.to(self.device)))
                    pixel_loss = low_high_loss + 2*low_freq_loss
                else:
                    pixel_loss = self.loss(init_predict, gt.to(self.device))
                # One-step invertibility consistency loss (EDICT) when predicting x0
                L_inv1 = torch.tensor(0.0, device=self.device)
                if self.pre_ori == 'True' and self.edict_inv1_enabled == 'True':
                    # Only compute for t < T-1 to ensure t+1 exists
                    mask = (t < (self.num_timesteps - 1))
                    if mask.any():
                        # Gather coefficients
                        abar = self.diffusion.gammas  # alphas_cumprod, shape [T]
                        abar_t = abar.gather(0, t.clamp_min(0)).view(-1, 1, 1, 1)
                        abar_t1 = abar.gather(0, (t + 1).clamp_max(self.num_timesteps - 1)).view(-1, 1, 1, 1)
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
                            L_inv1 = F.mse_loss(x_t1_hat.index_select(0, valid_idx), x_t1.index_select(0, valid_idx))

                loss = ddpm_loss + self.beta_loss * (pixel_loss) / self.num_timesteps + self.lambda_inv1 * L_inv1
                loss.backward()
                optimizer.step()
                if self.high_low_freq == 'True':
                    tq.set_postfix(loss=loss.item(), high_freq_ddpm_loss=ddpm_loss.item(), low_freq_pixel_loss=low_freq_loss.item(), pixel_loss=low_high_loss.item())
                else:
                    tq.set_postfix(loss=loss.item(), ddpm_loss=ddpm_loss.item(), pixel_loss=pixel_loss.item())
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
                    if not os.path.exists(self.weight_save_path):
                        os.makedirs(self.weight_save_path)
                    torch.save(self.network.init_predictor.state_dict(),
                               os.path.join(self.weight_save_path, f'model_init_{iteration}.pth'))
                    torch.save(self.network.denoiser.state_dict(),
                               os.path.join(self.weight_save_path, f'model_denoiser_{iteration}.pth'))
                    if self.EMA_or_not == 'True':
                        torch.save(self.ema_model.init_predictor.state_dict(),
                                   os.path.join(self.weight_save_path, f'model_init_ema_{iteration}.pth'))
                        torch.save(self.ema_model.denoiser.state_dict(),
                                   os.path.join(self.weight_save_path, f'model_denoiser_ema_{iteration}.pth'))



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
