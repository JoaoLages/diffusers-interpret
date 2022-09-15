from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Set, Dict, Any

import torch
from PIL import ImageDraw
from PIL.Image import Image
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess
from transformers import BatchEncoding, PreTrainedTokenizerBase

from diffusers_interpret.attribution import gradients_attribution
from diffusers_interpret.data import PipelineExplainerOutput, PipelineImg2ImgExplainerOutput, \
    BaseMimicPipelineCallOutput, AttributionMethods, AttributionAlgorithm, PipelineExplainerForBoundingBoxOutput, \
    PipelineImg2ImgExplainerForBoundingBoxOutputOutput
from diffusers_interpret.generated_images import GeneratedImages
from diffusers_interpret.pixel_attributions import PixelAttributions
from diffusers_interpret.saliency_map import SaliencyMap
from diffusers_interpret.token_attributions import TokenAttributions
from diffusers_interpret.utils import clean_token_from_prefixes_and_suffixes


class BasePipelineExplainer(ABC):
    """
    Core base class to explain all DiffusionPipeline: text2img, img2img and inpaint pipelines
    """

    def __init__(self, pipe: DiffusionPipeline, verbose: bool = True, gradient_checkpointing: bool = False) -> None:
        self.pipe = pipe
        self.verbose = verbose
        self.pipe._progress_bar_config = {
            **(getattr(self.pipe, '_progress_bar_config', {}) or {}),
            'disable': not verbose
        }
        self.gradient_checkpointing = gradient_checkpointing
        if self.gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def _preprocess_input(
        self,
        prompt: str,
        init_image: Optional[Union[torch.FloatTensor, Image]] = None,
        mask_image: Optional[Union[torch.FloatTensor, Image]] = None
    ) -> Tuple[Any, Any, Any]:
        return prompt, init_image, mask_image

    def __call__(
        self,
        prompt: str,
        init_image: Optional[Union[torch.FloatTensor, Image]] = None,
        mask_image: Optional[Union[torch.FloatTensor, Image]] = None,
        attribution_method: Union[str, AttributionMethods] = None,
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        run_safety_checker: bool = False,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        get_images_for_all_inference_steps: bool = True,
        output_type: Optional[str] = 'pil',
        **kwargs
    ) -> Union[
        PipelineExplainerOutput,
        PipelineExplainerForBoundingBoxOutput,
        PipelineImg2ImgExplainerOutput,
        PipelineImg2ImgExplainerForBoundingBoxOutputOutput
    ]:
        """
        Calls a DiffusionPipeline and generates explanations for a given prompt.

        Args:
            prompt (`str`):
                Input string for the diffusion model
            init_image (`torch.FloatTensor` or `PIL.Image.Image`, *optional*):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the process.
                If provided, output will be of type `PipelineImg2ImgExplainerOutput` or `PipelineImg2ImgExplainerForBoundingBoxOutputOutput`.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`, *optional*):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. The mask image will be
                converted to a single channel (luminance) before use.
            attribution_method (`Union[str, AttributionMethods]`, *optional*):
                `AttributionMethods` or `str` with the attribution algorithms to compute.
                Only one algorithm per type of attribution. If `str` is provided, the same algorithm
                will be applied to calculate both token and pixel attributions.
            explanation_2d_bounding_box (`Tuple[Tuple[int, int], Tuple[int, int]]`, *optional*):
                Tuple with the bounding box coordinates to calculate attributions for.
                The tuple is like (upper left corner, bottom right corner). Example: `((0, 0), (300, 300))`
                If this argument is provided, the output will be of type `PipelineExplainerForBoundingBoxOutput`
                or `PipelineImg2ImgExplainerForBoundingBoxOutputOutput`-
            consider_special_tokens (bool, defaults to `True`):
                If True, token attributions will also show attributions for `pipe.tokenizer.SPECIAL_TOKENS_ATTRIBUTES`
            clean_token_prefixes_and_suffixes (bool, defaults to `True`):
                If True, tries to clean prefixes and suffixes added by the `pipe.tokenizer`.
            run_safety_checker (bool, defaults to `False`):
                If True, will run the NSFW checker and return a black image if the safety checker says so.
            n_last_diffusion_steps_to_consider_for_attributions (int, *optional*):
                If not provided, it will calculate explanations for the output image based on all the diffusion steps.
                If given a number, it will only use the last provided diffusion steps.
                Set to `n_last_diffusion_steps_to_consider_for_attributions=0` for deactivating attributions calculation.
            get_images_for_all_inference_steps (bool, defaults to `True`):
                If True, will return all the images during diffusion in `output.all_images_during_generation`
            output_type (str, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `torch.Tensor`.
            **kwargs:
                Used to pass more arguments to DiffusionPipeline.__call__.
        Returns:
            [`PipelineExplainerOutput`], [`PipelineExplainerForBoundingBoxOutput`],
            [`PipelineImg2ImgExplainerOutput`] or [`PipelineImg2ImgExplainerForBoundingBoxOutputOutput`]

            [`PipelineExplainerOutput`] if `init_image=None` and `explanation_2d_bounding_box=None`
            [`PipelineExplainerForBoundingBoxOutput`] if `init_image=None` and `explanation_2d_bounding_box is not None`
            [`PipelineImg2ImgExplainerOutput`] if `init_image is not None` and `explanation_2d_bounding_box=None`
            [`PipelineImg2ImgExplainerForBoundingBoxOutputOutput`] if `init_image is not None` and `explanation_2d_bounding_box is not None`
        """

        attribution_method = attribution_method or AttributionMethods()

        if isinstance(attribution_method, str):
            attribution_method = AttributionMethods(
                tokens_attribution_method=AttributionAlgorithm(attribution_method),
                pixels_attribution_method=AttributionAlgorithm(attribution_method)
            )
        else:
            if not isinstance(attribution_method, AttributionMethods):
                raise ValueError("`attribution_method` has to be of type `str` or `AttributionMethods`")

            for k in ['tokens_attribution_method', 'pixels_attribution_method']:
                v = getattr(attribution_method, k)
                if not isinstance(v, AttributionAlgorithm):
                    setattr(attribution_method, k, AttributionAlgorithm(v))

        if isinstance(prompt, str):
            batch_size = 1 # TODO: make compatible with bigger batch sizes
        elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], str):
            batch_size = len(prompt)
            raise NotImplementedError("Passing a list of strings in `prompt` is still not implemented yet.")
        else:
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")

        # TODO: add asserts for out of bounds
        if explanation_2d_bounding_box:
            pass

        prompt, init_image, mask_image = self._preprocess_input(prompt=prompt, init_image=init_image, mask_image=mask_image)

        # get prompt text embeddings
        tokens, text_input, text_embeddings = self.get_prompt_tokens_token_ids_and_embeds(prompt=prompt)

        # Enable gradient, if `n_last_diffusion_steps_to_consider_for_attributions > 0`
        calculate_attributions = n_last_diffusion_steps_to_consider_for_attributions is None \
                                 or n_last_diffusion_steps_to_consider_for_attributions > 0
        if not calculate_attributions:
            torch.set_grad_enabled(False)
        else:
            torch.set_grad_enabled(True)

        # Get prediction with their associated gradients
        output: BaseMimicPipelineCallOutput = self._mimic_pipeline_call(
            text_input=text_input,
            text_embeddings=text_embeddings,
            init_image=init_image,
            mask_image=mask_image,
            batch_size=batch_size,
            output_type=None,
            run_safety_checker=run_safety_checker,
            n_last_diffusion_steps_to_consider_for_attributions=n_last_diffusion_steps_to_consider_for_attributions,
            get_images_for_all_inference_steps=get_images_for_all_inference_steps,
            **kwargs
        )

        # transform BaseMimicPipelineCallOutput to PipelineExplainerOutput or PipelineExplainerForBoundingBoxOutput
        output_kwargs = {
            'image': output.images[0],
            'nsfw_content_detected': output.nsfw_content_detected,
            'all_images_during_generation': output.all_images_during_generation,
        }
        if explanation_2d_bounding_box is not None:
            output['explanation_2d_bounding_box'] = explanation_2d_bounding_box
            output: PipelineExplainerForBoundingBoxOutput = PipelineExplainerForBoundingBoxOutput(**output_kwargs)
        else:
            output: PipelineExplainerOutput = PipelineExplainerOutput(**output_kwargs)

        if output.nsfw_content_detected:
            raise Exception(
                "NSFW content was detected, it is not possible to provide an explanation. "
                "Try to set `run_safety_checker=False` if you really want to skip the NSFW safety check."
            )

        # Calculate primary attribution scores
        if calculate_attributions:
            output: Union[PipelineExplainerOutput, PipelineImg2ImgExplainerOutput] = self._get_attributions(
                output=output,
                attribution_method=attribution_method,
                tokens=tokens,
                text_embeddings=text_embeddings,
                init_image=init_image,
                mask_image=mask_image,
                explanation_2d_bounding_box=explanation_2d_bounding_box,
                consider_special_tokens=consider_special_tokens,
                clean_token_prefixes_and_suffixes=clean_token_prefixes_and_suffixes,
                n_last_diffusion_steps_to_consider_for_attributions=n_last_diffusion_steps_to_consider_for_attributions,
                **kwargs
            )

        if batch_size == 1:
            # squash batch dimension
            for k in ['nsfw_content_detected', 'token_attributions', 'pixel_attributions']:
                if getattr(output, k, None) is not None:
                    output[k] = output[k][0]
            if output.all_images_during_generation:
                output.all_images_during_generation = [b[0] for b in output.all_images_during_generation]

        else:
            raise NotImplementedError

        # convert to PIL Image if requested
        # also draw bounding box in the last image if requested
        if output.all_images_during_generation or output_type == "pil":
            all_images = GeneratedImages(
                all_generated_images=output.all_images_during_generation or [output.image],
                pipe=self.pipe,
                remove_batch_dimension=batch_size==1,
                prepare_image_slider=bool(output.all_images_during_generation)
            )
            if output.all_images_during_generation:
                output.all_images_during_generation = all_images
                image = output.all_images_during_generation[-1]
            else:
                image = all_images[-1]

            if explanation_2d_bounding_box:
                draw = ImageDraw.Draw(image)
                draw.rectangle(explanation_2d_bounding_box, outline="red")

            if output_type == "pil":
                output.image = image

        return output

    def _post_process_token_attributions(
        self,
        output: PipelineExplainerOutput,
        tokens: List[List[str]],
        token_attributions: torch.Tensor,
        consider_special_tokens: bool,
        clean_token_prefixes_and_suffixes: bool
    ) -> PipelineExplainerOutput:
        # remove special tokens
        assert len(token_attributions) == len(tokens)
        output.token_attributions = []
        for image_token_attributions, image_tokens in zip(token_attributions, tokens):
            assert len(image_token_attributions) == len(image_tokens)

            # Add token attributions
            output.token_attributions.append([])
            for attr, token in zip(image_token_attributions, image_tokens):
                if consider_special_tokens or token not in self.special_tokens_attributes:

                    if clean_token_prefixes_and_suffixes:
                        token = clean_token_from_prefixes_and_suffixes(token)

                    output.token_attributions[-1].append(
                        (token, attr)
                    )

            output.token_attributions[-1] = TokenAttributions(output.token_attributions[-1])

        return output

    def _get_attributions(
        self,
        output: Union[PipelineExplainerOutput, PipelineExplainerForBoundingBoxOutput],
        attribution_method: AttributionMethods,
        tokens: List[List[str]],
        text_embeddings: torch.Tensor,
        init_image: Optional[torch.FloatTensor] = None,
        mask_image: Optional[Union[torch.FloatTensor, Image]] = None,
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        **kwargs
    ) -> Union[
        PipelineExplainerOutput,
        PipelineExplainerForBoundingBoxOutput,
        PipelineImg2ImgExplainerOutput,
        PipelineImg2ImgExplainerForBoundingBoxOutputOutput
    ]:
        if self.verbose:
            print("Calculating token attributions... ", end='')

        token_attributions = gradients_attribution(
            pred_logits=output.image,
            input_embeds=(text_embeddings,),
            attribution_algorithms=[attribution_method.tokens_attribution_method],
            explanation_2d_bounding_box=explanation_2d_bounding_box
        )[0].detach().cpu().numpy()

        output = self._post_process_token_attributions(
            output=output,
            tokens=tokens,
            token_attributions=token_attributions,
            consider_special_tokens=consider_special_tokens,
            clean_token_prefixes_and_suffixes=clean_token_prefixes_and_suffixes
        )

        if self.verbose:
            print("Done!")

        return output

    @property
    def special_tokens_attributes(self) -> Set[str]:

        # remove verbosity
        verbose = self.tokenizer.verbose
        self.tokenizer.verbose = False

        # get special tokens
        special_tokens = []
        for attr in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            t = getattr(self.tokenizer, attr, None)

            if isinstance(t, str):
                special_tokens.append(t)
            elif isinstance(t, list) and len(t) > 0 and isinstance(t[0], str):
                special_tokens += t

        # reset verbosity
        self.tokenizer.verbose = verbose

        return set(special_tokens)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError

    @abstractmethod
    def get_prompt_tokens_token_ids_and_embeds(self, prompt: Union[str, List[str]]) -> Tuple[List[List[str]], BatchEncoding, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
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
        Tuple[Union[List[Image], torch.Tensor], Optional[Union[List[List[Image]], List[torch.Tensor]]], Optional[List[bool]]]
    ]:
        r"""
        Mimics DiffusionPipeline.__call__ but adds extra functionality to calculate explanations.

        Args:
            text_input (`BatchEncoding`):
                Tokenized input string.
            text_embeddings (`torch.Tensor`):
                Output of the text encoder.
            batch_size (`int`):
                Batch size to be used.
            init_image (`torch.FloatTensor`, *optional*):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`, *optional*):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. The mask image will be
                converted to a single channel (luminance) before use.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength`
                is 1, the denoising process will be run on the masked area for the full number of iterations specified
                in `num_inference_steps`. `init_image` will be used as a reference for the masked area, adding more
                noise to that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The reference number of denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. This parameter will be modulated by `strength`, as explained above.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`BaseMimicPipelineCallOutput`] or `tuple`:
            [`BaseMimicPipelineCallOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is a list with the generated images,
            the second element contains all the generated images during the diffusion process and the third element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker` .
        """
        raise NotImplementedError


class BasePipelineImg2ImgExplainer(BasePipelineExplainer):
    """
    Core base class to explain img2img and inpaint pipelines
    """
    def _preprocess_input(
        self,
        prompt: str,
        init_image: Optional[Union[torch.FloatTensor, Image]] = None,
        mask_image: Optional[Union[torch.FloatTensor, Image]] = None
    ) -> Tuple[Any, Any, Any]:
        """
        Converts input image to tensor
        """
        prompt, init_image, mask_image = super()._preprocess_input(
            prompt=prompt, init_image=init_image, mask_image=mask_image
        )
        if init_image is None:
            raise TypeError("missing 1 required positional argument: 'init_image'")

        init_image = preprocess(init_image).to(self.pipe.device).permute(0, 2, 3, 1)
        init_image.requires_grad = True

        return prompt, init_image, mask_image

    def _get_attributions(
        self,
        output: Union[PipelineExplainerOutput, PipelineExplainerForBoundingBoxOutput],
        attribution_method: AttributionMethods,
        tokens: List[List[str]],
        text_embeddings: torch.Tensor,
        init_image: Optional[torch.FloatTensor] = None,
        mask_image: Optional[Union[torch.FloatTensor, Image]] = None,
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        **kwargs
    ) -> Union[
        PipelineExplainerOutput,
        PipelineExplainerForBoundingBoxOutput,
        PipelineImg2ImgExplainerOutput,
        PipelineImg2ImgExplainerForBoundingBoxOutputOutput
    ]:
        if init_image is None:
            raise TypeError("missing 1 required positional argument: 'init_image'")

        input_embeds = (text_embeddings,)
        if n_last_diffusion_steps_to_consider_for_attributions is None:
            input_embeds = (text_embeddings, init_image)

        if self.verbose:
            if n_last_diffusion_steps_to_consider_for_attributions is None:
                print("Calculating token and image pixel attributions... ", end='')
            else:
                print(
                    "Can't calculate image pixel attributions "
                    "with a specified `n_last_diffusion_steps_to_consider_for_attributions`. "
                    "Set `n_last_diffusion_steps_to_consider_for_attributions=None` "
                    "if you wish to calculate image pixel attributions"
                )
                print("Calculating token attributions... ", end='')

        attributions = gradients_attribution(
            pred_logits=output.image,
            input_embeds=input_embeds,
            attribution_algorithms=[
                attribution_method.tokens_attribution_method, attribution_method.pixels_attribution_method
            ],
            explanation_2d_bounding_box=explanation_2d_bounding_box
        )

        token_attributions = attributions[0].detach().cpu().numpy()

        pixel_attributions = None
        if n_last_diffusion_steps_to_consider_for_attributions is None:
            pixel_attributions = attributions[1].detach().cpu().numpy()

        output = self._post_process_token_attributions(
            output=output,
            tokens=tokens,
            token_attributions=token_attributions,
            consider_special_tokens=consider_special_tokens,
            clean_token_prefixes_and_suffixes=clean_token_prefixes_and_suffixes
        )

        # removes preprocessing done in diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
        init_image = (init_image + 1.0) / 2.0

        # add batch dimension to mask if needed
        masks = mask_image
        if isinstance(masks, Image):
            masks = [masks]
        elif torch.is_tensor(masks) and len(masks.shape) == 3:
            masks = masks.unsqueeze(0)

        # construct PixelAttributions objects
        images = init_image.detach().cpu().numpy()
        assert len(images) == len(pixel_attributions)
        if masks is not None:
            assert len(images) == len(masks)
        pixel_attributions = [
            PixelAttributions(
                attr,
                saliency_map=SaliencyMap(
                    image=img,
                    pixel_attributions=attr,
                    mask=mask
                )
            ) for img, attr, mask in zip(images, pixel_attributions, masks or [None] * len(images))
        ]

        output_kwargs = {
            'image': output.image,
            'nsfw_content_detected': output.nsfw_content_detected,
            'all_images_during_generation': output.all_images_during_generation,
            'token_attributions': output.token_attributions,
            'pixel_attributions': pixel_attributions
        }
        if explanation_2d_bounding_box is not None:
            output_kwargs['explanation_2d_bounding_box'] = explanation_2d_bounding_box
            output = PipelineImg2ImgExplainerForBoundingBoxOutputOutput(**output_kwargs)
        else:
            output = PipelineImg2ImgExplainerOutput(**output_kwargs)

        if self.verbose:
            print("Done!")

        return output
