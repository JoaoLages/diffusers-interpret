import inspect
import warnings
from abc import ABC, abstractmethod
import random
from typing import List, Optional, Union, Dict, Any, Tuple

from functorch import make_functional, vmap, grad
from tqdm.auto import tqdm

import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, LDMTextToImagePipeline, DiffusionPipeline
from transformers import BatchEncoding

from diffusers_interpret.attribution import gradient_x_inputs_attribution


class BasePipelineExplainer(ABC):
    def __init__(self, pipe: DiffusionPipeline, verbose: bool = True):
        self.pipe = pipe
        self.verbose = verbose
        self.make_pipe_functional()

    @abstractmethod
    def make_pipe_functional(self):
        raise NotImplementedError

    def __call__(
        self,
        prompt: Union[str, List[str]],
        attribution_method: str = 'grad_x_input',
        normalize_attributions: bool = False,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = 'pil',
        run_safety_checker: bool = False
    ) -> Dict[str, Any]:
        # TODO: add description

        if attribution_method != 'grad_x_input':
            raise NotImplementedError("Only `attribution_method='grad_x_input'` is implemented for now")

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_input, text_embeddings = self.get_prompt_token_ids_and_embeds(prompt=prompt)

        # Generator cant be None
        generator = generator or torch.Generator(self.pipe.device).manual_seed(random.randint(0, 9999))

        # Get prediction with their associated gradients
        #output = self._mimic_pipeline_call(
        #    text_input=text_input,
        #    text_embeddings=text_embeddings,
        #    batch_size=batch_size,
        #    height=height,
        #    width=width,
        #    num_inference_steps=num_inference_steps,
        #    guidance_scale=guidance_scale,
        #    eta=eta,
        #    generator=generator,
        #    output_type=None,
        #    run_safety_checker=run_safety_checker,
        #    enable_grad=True
        #)

        #if output['nsfw_content_detected']:
        #    raise Exception(
        #        "NSFW content was detected, it is not possible to provide an explanation. "
        #        "Try to set `run_safety_checker=False` if you really want to skip the NSFW safety check."
        #    )

        def get_pred_logit(logit_idx, text_input, text_embeddings):
            i, j, k = logit_idx
            return self._mimic_pipeline_call(
                text_input=text_input,
                text_embeddings=text_embeddings,
                batch_size=batch_size,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                generator=generator,
                output_type=None,
                run_safety_checker=run_safety_checker,
                enable_grad=True
            )['sample'][0][i][j][k]

        logits_idx = []
        for i in range(width):
            for j in range(height):
                for k in range(3):
                    logits_idx.append((i, j, k))

        i = 0
        per_sample_grads = []
        while True:
            per_sample_grads.append(
                vmap(grad(get_pred_logit), (None, None, 0))(
                    logits_idx[i: i + 100],
                    [text_input] * len(logits_idx[i: i + 100]),
                    [text_embeddings] * len(logits_idx[i: i + 100])
                )
            )
            i += 100
            if not logits_idx[i: i + 100]:
                break

        import ipdb; ipdb.set_trace()

        # Get primary attribution scores
        if self.verbose:
            print("Calculating primary attribution scores...")
        #if attribution_method == 'grad_x_input':
        #    output['token_attributions'] = gradient_x_inputs_attribution(
        #        pred_logits=output['sample'][0], input_embeds=text_embeddings,
        #        normalize_attributions=normalize_attributions
        #    ).detach().cpu().numpy()
        #else:
        #    raise NotImplementedError("Only `attribution_method='grad_x_input'` is implemented for now")

        # convert to PIL Image if requested
        #if output_type == "pil":
        #    output['sample'] = self.pipe.numpy_to_pil(output['sample'].detach().cpu().numpy())

        #return output

    @abstractmethod
    def get_prompt_token_ids_and_embeds(self, prompt: Union[str, List[str]]) -> Tuple[BatchEncoding, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _mimic_pipeline_call(
        self,
        text_input: BatchEncoding,
        text_embeddings: torch.Tensor,
        batch_size: int,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = 'pil',
        run_safety_checker: bool = True,
        enable_grad: bool = False
    ) -> Dict[str, Any]:
        raise NotImplementedError


class LDMTextToImagePipelineExplainer(BasePipelineExplainer):
    def __init__(self, pipe: LDMTextToImagePipeline, verbose: bool = True):
        super().__init__(pipe=pipe, verbose=verbose)

    def get_prompt_token_ids_and_embeds(self, prompt: Union[str, List[str]]) -> Tuple[BatchEncoding, torch.Tensor]:
        self.pipe: LDMTextToImagePipeline
        text_input = self.pipe.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.pipe.bert(text_input.input_ids.to(self.pipe.device))[0]
        return text_input, text_embeddings

    def _mimic_pipeline_call(
        self,
        text_input: BatchEncoding,
        text_embeddings: torch.Tensor,
        batch_size: int,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = 'pil',
        run_safety_checker: bool = True,
        enable_grad: bool = False
    ) -> Dict[str, Any]:
        # TODO: add description

        self.pipe: LDMTextToImagePipeline
        torch.set_grad_enabled(enable_grad)

        # get unconditional embeddings for classifier free guidance
        if guidance_scale != 1.0:
            uncond_input = self.pipe.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
            uncond_embeddings = self.pipe.bert(uncond_input.input_ids.to(self.pipe.device))[0]

        # get prompt text embeddings

        latents = torch.randn(
            (batch_size, self.pipe.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(self.pipe.device)

        self.pipe.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.pipe.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in tqdm(self.pipe.scheduler.timesteps, desc="Diffusion process", disable=not self.verbose):

            if guidance_scale == 1.0:
                # guidance_scale of 1 means no guidance
                latents_input = latents
                context = text_embeddings
            else:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                latents_input = torch.cat([latents] * 2)
                context = torch.cat([uncond_embeddings, text_embeddings])

            # predict the noise residual
            noise_pred = self.pipe.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            # perform guidance
            if guidance_scale != 1.0:
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_kwargs)["prev_sample"]

        # scale and decode the image latents with vqvae
        latents = 1 / 0.18215 * latents
        image = self.pipe.vqvae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)

        has_nsfw_concept = None
        if run_safety_checker:
            warnings.warn(
                f"{self.__class__.__name__} has no safety checker, `run_safety_checker=True` will be ignored"
            )

        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image.detach().cpu().numpy())

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}


class StableDiffusionPipelineExplainer(BasePipelineExplainer):
    def __init__(self, pipe: StableDiffusionPipeline, verbose: bool = True):
        super().__init__(pipe=pipe, verbose=verbose)

    def make_pipe_functional(self):
        self.pipe: StableDiffusionPipeline
        self.pipe.text_encoder, self.pipe.text_encoder.params = make_functional(self.pipe.text_encoder)
        self.pipe.unet, self.pipe.unet.params = make_functional(self.pipe.unet)
        self.pipe.vae.decoder, self.pipe.vae.decoder.params = make_functional(self.pipe.vae.decoder)
        self.pipe.vae.post_quant_conv, self.pipe.vae.post_quant_conv.params = make_functional(self.pipe.vae.post_quant_conv)

    def get_prompt_token_ids_and_embeds(self, prompt: Union[str, List[str]]) -> Tuple[BatchEncoding, torch.Tensor]:
        self.pipe: StableDiffusionPipeline
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(
            self.pipe.text_encoder.params,
            text_input.input_ids.to(self.pipe.device)
        )[0]
        return text_input, text_embeddings

    def _mimic_pipeline_call(
        self,
        text_input: BatchEncoding,
        text_embeddings: torch.Tensor,
        batch_size: int,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = 'pil',
        run_safety_checker: bool = True,
        enable_grad: bool = False
    ) -> Dict[str, Any]:
        # TODO: add description

        self.pipe: StableDiffusionPipeline
        torch.set_grad_enabled(enable_grad)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.pipe.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.pipe.text_encoder(
                self.pipe.text_encoder.params,
                uncond_input.input_ids.to(self.pipe.device)
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise
        latents = torch.randn(
            (batch_size, self.pipe.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.pipe.device,
        )

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.pipe.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
        if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
            latents = latents * self.pipe.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.pipe.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in tqdm(
            enumerate(self.pipe.scheduler.timesteps),
            total=len(self.pipe.scheduler.timesteps),
            disable=not self.verbose,
            desc="Diffusion process"
        ):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
                sigma = self.pipe.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.pipe.unet(
                self.pipe.unet.params,
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
                latents = self.pipe.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
            else:
                latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        z = self.pipe.vae.post_quant_conv(self.pipe.vae.post_quant_conv.params, latents)
        image = self.pipe.vae.decoder(self.pipe.vae.decoder.params, z)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)

        has_nsfw_concept = None
        if run_safety_checker:
            image = image.detach().cpu().numpy()
            safety_cheker_input = self.pipe.feature_extractor(
                self.pipe.numpy_to_pil(image), return_tensors="pt"
            ).to(self.pipe.device)
            image, has_nsfw_concept = self.pipe.safety_checker(
                images=image, clip_input=safety_cheker_input.pixel_values
            )

        if output_type == "pil":
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            image = self.pipe.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}
