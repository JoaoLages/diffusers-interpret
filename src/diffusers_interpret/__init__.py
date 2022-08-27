def forward(
    text_embeddings: torch.Tensor,
    pipe: StableDiffusionPipeline,
    max_length: int,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    eta: Optional[float] = 0.0,
    generator: Optional[torch.Generator] = None,
    output_type: Optional[str] = "pil",
    run_safety_checker: bool = False
    **kwargs
):
    if "torch_device" in kwargs:
        device = kwargs.pop("torch_device")
        warnings.warn(
            "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
            " Consider using `pipe.to(torch_device)` instead."
        )

        # Set device as before (to be removed in 0.3.0)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    assert isinstance(text_embeddings, torch.Tensor)
    batch_size = text_embeddings.shape[0]

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # get the intial random noise
    latents = torch.randn(
        (batch_size, self.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=self.device,
    )

    # set timesteps
    accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1

    self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
    if isinstance(self.scheduler, LMSDiscreteScheduler):
        latents = latents * self.scheduler.sigmas[0]

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    for i, t in tqdm(enumerate(self.scheduler.timesteps), total=num_inference_steps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)["prev_sample"]
        else:
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents)

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    # run safety checker
    has_nsfw_concept = None
    if run_safety_checker:
        safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

    if output_type == "pil":
        image = self.numpy_to_pil(image)

    return {"sample": image, "nsfw_content_detected": has_nsfw_concept}

    return pipe([inputs], num_inference_steps=1)
