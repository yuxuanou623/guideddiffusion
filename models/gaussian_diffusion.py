"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import os
import copy
import random
import wandb
import numpy as np
import torch as th
from tqdm.auto import tqdm
from .nn import mean_flat
import blobfile as bf
from utils import logger
import torch.nn.functional as F
import torch.nn as nn

def normalize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def normalize_dim(img):
    _min = img.min(dim=3)[0].min(dim=2)[0].reshape(img.shape[0],1,1,1).repeat(1,1,img.shape[2],img.shape[3])
    _max = img.max(dim=3)[0].max(dim=2)[0].reshape(img.shape[0],1,1,1).repeat(1,1,img.shape[2],img.shape[3])
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def normalize_batch(img):
    _min = img.min(axis=1).reshape(img.shape[0],1).repeat(img.shape[1], 1)
    _max = img.max(axis=1).reshape(img.shape[0],1).repeat(img.shape[1], 1)
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0.5)
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps   # original code
        beta_start = scale * 0.0001
        beta_end = scale * 0.02


        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            rescale_timesteps=False,
            conditioning_noise="constant",
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.rescale_timesteps = rescale_timesteps
        self.conditioning_noise = conditioning_noise
        assert self.conditioning_noise in ["reverse", "constant"]

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        # assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        # alphas = 2.0 - betas
        self.alphas = alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.ssim_whole = []
        self.psnr_whole = []
        self.ssim_sample = []
        self.psnr_sample = []
        self.l1_whole = []
        self.mse_whole = []
        self.i = []

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, x_end, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * x_end
        )

    def q_sample_conditioning(self, model_kwargs, t, train: bool = True):
        # perturb conditioning image with noise as in the forward diffusion process
        '''if train:
            if self.conditioning_noise == 'reverse':
                t_reverse_diffusion = (self.num_timesteps - 1 - t).to(device=dist_util.dev())
                conditioning_x = self.q_sample(model_kwargs["image"], t_reverse_diffusion, noise=None)
            elif self.conditioning_noise == 'constant':
                conditioning_x = model_kwargs["image"] + th.randn_like(model_kwargs["image"])
        else:'''

        conditioning_x = model_kwargs["image"]

        model_kwargs["conditioning_x"] = conditioning_x

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, cond=None, clip_denoised=True, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # x=torch.zeros(x.shape).cuda()
        x_in = th.cat((x, cond), 1)

        model_output, masks, mask_logits = model(x_in, self._scale_timesteps(t), **model_kwargs)
        # model_output = model(x_in, **model_kwargs)
        # model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
                #pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=process_xstart(model_output)) didnt work

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        # assert (
        #         model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        # )x
        return {
            "model_output": model_output,
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "masks": masks,
            "masks_logits": mask_logits
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                       _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                       - pred_xstart
               ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        class_cond = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        # class_cond = cond_fn(model_kwargs['image'], self._scale_timesteps(t), **model_kwargs)
        eps = eps - (1 - alpha_bar).sqrt() * class_cond

        out = p_mean_var.copy()
        out["class_cond"] = class_cond
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
            self,
            model,
            x,
            cond,
            t,
            clip_denoised=True,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """

        out = self.p_mean_variance(
            model,
            x,
            t,
            cond,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "model_output": out["model_output"], "masks": out["masks"], "masks_logits": out["masks_logits"]}

    def p_sample_loop(
            self,
            model_forward,
            model_backward,
            test_data_input,
            num_batch,
            shape,
            model_name=None,
            clip_denoised=True,
            model_kwargs=None,
            device=None,
            eta=0.0,
            model_forward_name: str =None,
            model_backward_name: str =None,
            ddim: bool = False
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
      
        for sample, masks , mask_logits in self.p_sample_loop_progressive(
                model_forward,
                model_backward,
                test_data_input,
                num_batch,
                shape,
                model_name=model_name,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                eta=eta,
                model_forward_name=model_forward_name,
                model_backward_name=model_backward_name,
                ddim=ddim
        ):
            final = sample
             
            masks_logits = mask_logits
        return final, masks, masks_logits


    def p_sample_loop_progressive(
            self,
            model_forward,
            model_backward,
            test_data_input,
            num_batch,
            shape,
            model_name=None,
            clip_denoised=True,
            model_kwargs=None,
            device=None,
            eta=0.0,
            model_forward_name: str=None,
            model_backward_name: str=None,
            ddim: bool=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model_forward.parameters()).device
        assert isinstance(shape, (tuple, list))


        if model_name == 'diffusion':
            indices = list(range(self.num_timesteps))[::-1]
            # num_mask = list(range(128))
            # num_mask = tqdm(num_mask)

            noise = th.randn(*shape, device=device)
            
            img_forward = noise.cuda()

            masks_all = []

            for i_forward in indices:
                t_forward = th.tensor([i_forward] * shape[0], device=device)
                with th.no_grad():
                    noise1 = th.randn(shape, device=device)

                    cond_forward =  test_data_input

                    
                   
                    if ddim == 'False':
                
                        out_forward = self.p_sample(
                            model_forward,
                            img_forward,
                            cond_forward,
                            t_forward,
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs,
                        )
                    else:
                        out_forward = self.ddim_sample(
                            model_forward,
                            img_forward,
                            cond_forward,
                            t_forward,
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs,
                            eta=eta,
                        )
             
                    prev_img_forward = out_forward["sample"]
                    masks = out_forward["masks"].detach().cpu().numpy()
                   
                    mask_logits = out_forward["masks_logits"]

                    masks_all.append(masks)
                    

                    

                    
    
                x_yield = prev_img_forward
                img_forward = prev_img_forward

                


            yield x_yield, masks_all, mask_logits
        elif model_name == 'unet':
            with th.no_grad():
                x0_pred_forward = model_forward(test_data_input, **model_kwargs)
                x0_pred_backward = model_backward(x0_pred_forward, **model_kwargs)

                yield x0_pred_backward

    def ddim_sample(
            self,
            model,
            x,
            cond,
            t,
            clip_denoised=True,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            cond,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        score_mean = out["mean"]

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        result_dict = {"sample": sample, "pred_xstart": out["pred_xstart"], "score_mean": score_mean}
        if cond_fn is not None:
            result_dict.update({"class_cond": out["class_cond"]})

        return result_dict




    def training_losses(self, model, input_img, trans_img, brain_mask, model_name, t,iteration, x_start_t=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = th.randn_like(input_img)
        terms = {}

        if model_name == 'diffusion':
            # trans_img = (trans_img / 255.0) * 2 - 1
            # input_img = (input_img / 255.0) * 2 - 1
            x_t = self.q_sample(trans_img, noise, self._scale_timesteps(t))
           
            # print("noise min", th.min(noise))
            # print("noise max", th.max(noise))
            ### grid mask ###
            mask = th.ones(trans_img.shape).cuda()
            start_pix = random.randint(0, 63) * 2
            direction = random.randint(0, 1)
            nonmaskp = 0

            
            
            cond = input_img 
            x_t_input = th.cat((x_t, cond), 1)
            mask_difference = input_img- trans_img
           
            # x_start_pred, masks, mask_logits = model(x_t_input, self._scale_timesteps(t), **model_kwargs)
            x_start_pred, masks, mask_logits = model(x_t_input, self._scale_timesteps(t), **model_kwargs)
            # mask_logits = model(x_t_input, self._scale_timesteps(t), **model_kwargs)
        elif model_name == 'unet':
            x_t_input = input_img
            x_start_pred = model(x_t_input, **model_kwargs)
        
        
       

        def mask_dominance_penalty(masks):
            """
            Penalizes masks that have low variance in pixel values across the image.
            
            :param masks: Tensor of shape (batch_size, num_masks, H, W)
            :return: Variance-based punishment loss scalar
            """
            # Compute the mean of each mask (per mask, per image)
            mean = th.mean(masks, dim=(2, 3), keepdim=True)  # Shape: (batch_size, num_masks, 1, 1)
            
            # Compute squared deviation from the mean (differentiable)
            squared_deviation = (masks - mean) ** 2
            
            # Compute the mean squared deviation (differentiable variance-like term)
            pixel_variance = th.mean(squared_deviation, dim=(2, 3))  # Shape: (batch_size, num_masks)
            
            # We want to maximize variance, so we penalize low variance
            return -th.mean(pixel_variance)  # Negative to encourage higher variance


       
        # masks_penalty = mask_dominance_penalty(mask_logits)
        

        target = trans_img
    #     loss = masks[:, 0, :, :] * (target - x_start_pred) ** 2 + \
    #    masks[:, 1, :, :] * (input_img - x_start_pred) ** 2 
        loss = (target - x_start_pred) ** 2
        
        # terms["loss"] = mean_flat(loss) + masks_penalty
        
        # def smooth_any_positive(x, alpha=10):
        #     """
        #     Differentiable function to check if any value in x is greater than 0.
        #     Uses Log-Sum-Exp (Softplus) to approximate OR.
        #     :param x: Tensor of shape (...), any number of dimensions
        #     :param alpha: Sharpness control (higher = closer to hard OR)
        #     :return: Smoothly approximates max(x, 0)
        #     """
        #     return th.log(1 + th.sum(th.exp(alpha * x))) / alpha
        def smooth_any_positive_fixed(x, alpha=100):
            """
            Ensures that the function outputs ~0 when all values in x are <= 0.
            
            :param x: Tensor of shape (...), any number of dimensions.
            :param alpha: Sharpness control (higher = closer to hard OR).
            :return: A smoothly transitioning value > 0 if any x_i > 0, otherwise ~0.
            """
            n = x.numel()  # Number of elements in x
            n_tensor = th.tensor(1 + n, dtype=x.dtype, device=x.device)  # Convert to tensor
            return (th.log(1 + th.sum(th.exp(alpha * x))) - th.log(n_tensor)) / alpha  # Normalized


        def check_empty_masks(masks_logits, alpha=10):
            """
            Computes smooth_any_positive for each mask individually (per mask, per image).
            Then returns the mean value across all masks per image.

            :param masks_logits: Tensor of shape (batch_size, num_mask, H, W)
            :param alpha: Sharpness parameter for smooth_any_positive
            :return: Tensor of shape (batch_size,), mean empty mask score per image
            """
            batch_size, num_mask, H, W = masks_logits.shape

            # Shift logits by -1/num_mask
            shifted_logits = masks_logits - (1.0 / num_mask)

            smooth_values_list = []  # Store values for each mask per image

            for b in range(batch_size):  # Iterate over batch
                per_image_values = []
                for m in range(num_mask):  # Iterate over masks
                    # Flatten the spatial dimensions and compute smooth_any_positive
                    smooth_value = reverse_relu(smooth_any_positive_fixed(shifted_logits[b, m].flatten(), alpha=alpha))
                    per_image_values.append(smooth_value)
                
                # Compute mean across all masks in the image
                mean_smooth_value = th.stack(per_image_values).mean()
                smooth_values_list.append(mean_smooth_value)

            # Convert list to tensor and return mean across batch
            smooth_values = th.stack(smooth_values_list)  # Shape: (batch_size,)
            
            return smooth_values.mean()  # Mean value per image
        def reverse_relu(y):
            """
            Reverse ReLU: Returns y when y < 0, and 0 when y >= 0.
            
            :param y: Input tensor.
            :return: Reverse ReLU applied to y.
            """
            return th.relu(-y)  # Equivalent to max(-y, 0)

        

        def entropy_loss(masks_logits):
            """
            Encourages confident predictions by minimizing entropy of softmax probabilities.
            
            :param masks_logits: Tensor of shape (batch_size, num_masks, H, W)
            :return: Entropy loss scalar.
            """
           
            entropy = -masks_logits * th.log(masks_logits + 1e-8)  # Compute entropy
            return th.mean(entropy)  # Average over batch, masks, and spatial dimensions
        def smoothness_loss(masks_logits):
            """
            Encourages spatial smoothness by penalizing large differences between neighboring pixels.
            
            :param masks_logits: (batch_size, num_masks, H, W), raw logits.
            :return: Smoothness loss scalar.
            """
            batch_size, num_masks, H, W = masks_logits.shape
            loss = 0

            if H > 1:  # Ensure we can compute row-wise differences
                diff_x = masks_logits[:, :, 1:, :] - masks_logits[:, :, :-1, :]
                loss += th.mean(th.abs(diff_x))

            if W > 1:  # Ensure we can compute column-wise differences
                diff_y = masks_logits[:, :, :, 1:] - masks_logits[:, :, :, :-1]
                loss += th.mean(th.abs(diff_y))

            return loss
        # terms["loss"] = -0.01*mask_logits.var(dim=1).mean() + mean_flat(loss) -0.05*check_empty_masks(mask_logits)  #mask5noncon2con_losss wandb colorful_fire
        terms["loss"] = entropy_loss(mask_logits)+0.1*smoothness_loss(mask_logits)+mean_flat(loss) +check_empty_masks(mask_logits)  #mask5noncon2con_losss wandb colorful_fire
        print("smoothness_loss(mask_logits)",smoothness_loss(mask_logits))
        print(entropy_loss(mask_logits))
        

      

        # print("mean_flat(loss)",mean_flat(loss))
        
        max_index = th.argmax(t)
        
            # Log the image to W&B
        if iteration % 200 ==0:
            wandb.log({
        "Image": wandb.Image(x_start_pred[max_index, :, :, :].squeeze(0).detach().cpu().numpy()),
        "Target": wandb.Image(target[max_index, :, :, :].squeeze(0).detach().cpu().numpy()),
        "Noncon": wandb.Image(input_img[max_index, :, :, :].squeeze(0).detach().cpu().numpy()),
        "Mask0": wandb.Image(masks[max_index, 0, :, :].squeeze(0).detach().cpu().numpy()),
        "Mask1": wandb.Image(masks[max_index, 1, :, :].squeeze(0).detach().cpu().numpy()),
        "Mask2": wandb.Image(masks[max_index, 2, :, :].squeeze(0).detach().cpu().numpy()),
        "Mask3": wandb.Image(masks[max_index, 3, :, :].squeeze(0).detach().cpu().numpy()),
        "Mask4": wandb.Image(masks[max_index, 4, :, :].squeeze(0).detach().cpu().numpy())}, step=iteration)


        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    res = th.from_numpy(arr)
    _dev = timesteps.device
    res = res.to(device=_dev)
    res = res[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

