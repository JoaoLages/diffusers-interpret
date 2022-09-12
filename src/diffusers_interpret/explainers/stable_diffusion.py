import inspect
from typing import List, Optional, Union, Tuple

import torch
from PIL.Image import Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import preprocess_mask
from torch.utils.checkpoint import checkpoint
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, StableDiffusionImg2ImgPipeline, \
    StableDiffusionInpaintPipeline
from transformers import BatchEncoding, PreTrainedTokenizerBase

from diffusers_interpret import BasePipelineExplainer
from diffusers_interpret.explainer import BaseMimicPipelineCallOutput, BasePipelineImg2ImgExplainer
from diffusers_interpret.utils import transform_images_to_pil_format


def decode_latents(
    latents: torch.Tensor,
    pipe: Union[StableDiffusionImg2ImgPipeline, StableDiffusionPipeline],
    gradient_checkpointing: bool,
    run_safety_checker: bool
) -> Tuple[torch.Tensor, Optional[List[bool]]]:
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    if not gradient_checkpointing or not torch.is_grad_enabled():
        image = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
    else:
        image = checkpoint(pipe.vae.decode, latents.to(pipe.vae.dtype), use_reentrant=False).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1)

    has_nsfw_concept = None
    if run_safety_checker:
        image = image.detach().cpu().numpy()
        safety_cheker_input = pipe.feature_extractor(
            pipe.numpy_to_pil(image), return_tensors="pt"
        ).to(pipe.device)
        image, has_nsfw_concept = pipe.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values
        )

    return image, has_nsfw_concept


class BaseStableDiffusionPipelineExplainer(BasePipelineExplainer):
    pipe: Union[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.pipe.tokenizer

    def get_prompt_tokens_token_ids_and_embeds(self, prompt: Union[str, List[str]]) -> Tuple[
        List[List[str]], BatchEncoding, torch.Tensor]:
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.pipe.device))[0]
        tokens = [self.pipe.tokenizer.convert_ids_to_tokens(sample) for sample in text_input['input_ids']]
        return tokens, text_input, text_embeddings

    def gradient_checkpointing_enable(self) -> None:
        self.pipe.text_encoder.gradient_checkpointing_enable()
        super().gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.pipe.text_encoder.gradient_checkpointing_disable()
        super().gradient_checkpointing_disable()


class StableDiffusionPipelineExplainer(BaseStableDiffusionPipelineExplainer):
    pipe: StableDiffusionPipeline

    def _mimic_pipeline_call(
        self,
        text_input: BatchEncoding,
        text_embeddings: torch.Tensor,
        batch_size: int,
        init_image: Optional[torch.FloatTensor] = None,
        mask_image: Optional[Union[torch.FloatTensor, Image]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        run_safety_checker: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        get_images_for_all_inference_steps: bool = False
    ) -> Union[
        BaseMimicPipelineCallOutput,
        Tuple[Union[List[Image], torch.Tensor], Optional[Union[List[List[Image]], List[torch.Tensor]]], Optional[
            List[bool]]]
    ]:
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if init_image is not None:
            if mask_image is not None:
                raise ValueError(
                    "`init_image` and `mask_image` were passed to StableDiffusionPipelineExplainer and are not expected.\n"
                    "Were you trying to use StableDiffusionInpaintPipelineExplainer ?"
                )
            else:
                raise ValueError(
                    "`init_image` was passed to StableDiffusionPipelineExplainer and is not expected.\n"
                    "Were you trying to use StableDiffusionImg2ImgPipelineExplainer ?"
                )

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
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.pipe.device))[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = "cpu" if self.pipe.device.type == "mps" else self.pipe.device
        latents_shape = (batch_size, self.pipe.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=latents_device,
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        latents = latents.to(self.pipe.device)

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

        all_generated_images = [] if get_images_for_all_inference_steps else None
        for i, t in enumerate(self.pipe.progress_bar(self.pipe.scheduler.timesteps)):

            if n_last_diffusion_steps_to_consider_for_attributions:
                if i < len(self.pipe.scheduler.timesteps) - n_last_diffusion_steps_to_consider_for_attributions:
                    torch.set_grad_enabled(False)
                else:
                    torch.set_grad_enabled(True)

            # decode latents
            if get_images_for_all_inference_steps:
                with torch.no_grad():
                    image, _ = decode_latents(
                        latents=latents, pipe=self.pipe,
                        gradient_checkpointing=self.gradient_checkpointing, run_safety_checker=run_safety_checker
                    )
                    all_generated_images.append(image)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
                sigma = self.pipe.scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            if not self.gradient_checkpointing or not torch.is_grad_enabled():
                noise_pred = self.pipe.unet(latent_model_input, t, text_embeddings).sample
            else:
                noise_pred = checkpoint(
                    self.pipe.unet.forward, latent_model_input, t, text_embeddings, use_reentrant=False
                ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
                latents = self.pipe.scheduler.step(noise_pred, i, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        image, has_nsfw_concept = decode_latents(
            latents=latents, pipe=self.pipe,
            gradient_checkpointing=self.gradient_checkpointing, run_safety_checker=run_safety_checker
        )
        if all_generated_images:
            all_generated_images.append(image)

        if output_type == "pil":
            if all_generated_images:
                all_generated_images = transform_images_to_pil_format(all_generated_images, self.pipe)
                image = all_generated_images[-1]
            else:
                image = transform_images_to_pil_format([image], self.pipe)[0]

        if return_dict:
            return BaseMimicPipelineCallOutput(
                images=image, nsfw_content_detected=has_nsfw_concept,
                all_images_during_generation=all_generated_images
            )
        else:
            return (image, all_generated_images, has_nsfw_concept)


class StableDiffusionImg2ImgPipelineExplainer(BasePipelineImg2ImgExplainer, BaseStableDiffusionPipelineExplainer):
    pipe: Union[StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline]

    def _mimic_pipeline_call(
        self,
        text_input: BatchEncoding,
        text_embeddings: torch.Tensor,
        batch_size: int,
        init_image: Optional[torch.FloatTensor] = None,
        mask_image: Optional[Union[torch.FloatTensor, Image]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        run_safety_checker: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        get_images_for_all_inference_steps: bool = False
    ) -> Union[
        BaseMimicPipelineCallOutput,
        Tuple[Union[List[Image], torch.Tensor], Optional[Union[List[List[Image]], List[torch.Tensor]]], Optional[
            List[bool]]]
    ]:

        if latents is not None:
            raise ValueError(
                f"`latents` was passed to {self.__class__.__name__} and it is not expected."
            )

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.pipe.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # save all generated images during diffusion, if get_images_for_all_inference_steps
        all_generated_images = [(init_image / 2 + 0.5).clamp(0, 1)] if get_images_for_all_inference_steps else None

        # encode the init image into latents and scale the latents
        init_latent_dist = self.pipe.vae.encode(init_image.permute(0, 3, 1, 2)).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        # expand init_latents for batch_size
        init_latents = torch.cat([init_latents] * batch_size)
        init_latents_orig = init_latents

        mask = None
        if mask_image is not None:
            # preprocess mask
            mask = preprocess_mask(mask_image).to(self.pipe.device)
            mask = torch.cat([mask] * batch_size)

            # check sizes
            if not mask.shape == init_latents.shape:
                raise ValueError("The mask and init_image should be the same size!")

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
            timesteps = torch.tensor(
                [num_inference_steps - init_timestep] * batch_size, dtype=torch.long, device=self.pipe.device
            )
        else:
            timesteps = self.pipe.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.pipe.device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.pipe.device)
        init_latents = self.pipe.scheduler.add_noise(init_latents, noise, timesteps).to(self.pipe.device)

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
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.pipe.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.pipe.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in enumerate(self.pipe.progress_bar(self.pipe.scheduler.timesteps[t_start:])):
            t_index = t_start + i

            if n_last_diffusion_steps_to_consider_for_attributions:
                if t_index < len(self.pipe.scheduler.timesteps) - n_last_diffusion_steps_to_consider_for_attributions:
                    torch.set_grad_enabled(False)
                else:
                    torch.set_grad_enabled(True)

            # decode latents
            if get_images_for_all_inference_steps:
                with torch.no_grad():
                    image, _ = decode_latents(
                        latents=latents, pipe=self.pipe,
                        gradient_checkpointing=self.gradient_checkpointing, run_safety_checker=run_safety_checker
                    )
                    all_generated_images.append(image)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
            if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
                sigma = self.pipe.scheduler.sigmas[t_index]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
                latent_model_input = latent_model_input.to(self.pipe.unet.dtype)
                t = t.to(self.pipe.unet.dtype)

            # predict the noise residual
            if not self.gradient_checkpointing or not torch.is_grad_enabled():
                noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            else:
                noise_pred = checkpoint(
                    self.pipe.unet.forward, latent_model_input, t, text_embeddings, use_reentrant=False
                ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
                latents = self.pipe.scheduler.step(noise_pred, t_index, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # masking
            if mask is not None:
                init_latents_proper = self.pipe.scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

        image, has_nsfw_concept = decode_latents(
            latents=latents, pipe=self.pipe,
            gradient_checkpointing=self.gradient_checkpointing, run_safety_checker=run_safety_checker
        )
        if all_generated_images:
            all_generated_images.append(image)

        if output_type == "pil":
            if all_generated_images:
                all_generated_images = transform_images_to_pil_format(all_generated_images, self.pipe)
                image = all_generated_images[-1]
            else:
                image = transform_images_to_pil_format([image], self.pipe)[0]

        if return_dict:
            return BaseMimicPipelineCallOutput(
                images=image, nsfw_content_detected=has_nsfw_concept,
                all_images_during_generation=all_generated_images
            )
        else:
            return (image, all_generated_images, has_nsfw_concept)


class StableDiffusionInpaintPipelineExplainer(StableDiffusionImg2ImgPipelineExplainer):
    # Actually the same as StableDiffusionImg2ImgPipelineExplainer
    pass