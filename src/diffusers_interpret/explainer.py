from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Set, Dict, Any

import torch
from PIL import ImageDraw
from PIL.Image import Image
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess
from transformers import BatchEncoding, PreTrainedTokenizerBase

from diffusers_interpret.attribution import gradient_x_inputs_attribution
from diffusers_interpret.generated_images import GeneratedImages
from diffusers_interpret.utils import clean_token_from_prefixes_and_suffixes


@dataclass
class BaseMimicPipelineCallOutput:
    """
    Output class for BasePipelineExplainer._mimic_pipeline_call

    Args:
        images (`List[Image]` or `torch.Tensor`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`Optional[List[bool]]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content.
        all_images_during_generation (`Optional[Union[List[List[Image]]], List[torch.Tensor]]`)
            A list with all the batch images generated during diffusion
    """
    images: Union[List[Image], torch.Tensor]
    nsfw_content_detected: Optional[List[bool]] = None
    all_images_during_generation: Optional[Union[List[List[Image]], List[torch.Tensor]]] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


@dataclass
class PipelineExplainerOutput:
    image: Union[Image, torch.Tensor]
    nsfw_content_detected: Optional[List[bool]] = None
    all_images_during_generation: Optional[Union[GeneratedImages, List[torch.Tensor]]] = None
    token_attributions: Optional[List[Tuple[str, float]]] = None
    normalized_token_attributions: Optional[List[Tuple[str, float]]] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


@dataclass
class PipelineImg2ImgExplainerOutput(PipelineExplainerOutput):
    pixel_attributions: Optional[List[Tuple[str, float]]] = None
    normalized_pixel_attributions: Optional[List[Tuple[str, float]]] = None
    pixel_attributions_heatmap: Optional[Any] = None #TODO: add typing


class CorePipelineExplainer(ABC):
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

    def _preprocess_input(self, **kwargs) -> Dict[str, Any]:
        return kwargs

    def __call__(
        self,
        prompt: str,
        attribution_method: str = 'grad_x_input',
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, # (upper left corner, bottom right corner)
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        run_safety_checker: bool = False,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        get_images_for_all_inference_steps: bool = True,
        output_type: Optional[str] = 'pil',
        **kwargs
    ) -> Union[PipelineExplainerOutput, PipelineImg2ImgExplainerOutput]:
        # TODO: add description

        if attribution_method != 'grad_x_input':
            raise NotImplementedError("Only `attribution_method='grad_x_input'` is implemented for now")

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

        kwargs = self._preprocess_input(**kwargs)

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
        output = self._mimic_pipeline_call(
            text_input=text_input,
            text_embeddings=text_embeddings,
            batch_size=batch_size,
            output_type=None,
            run_safety_checker=run_safety_checker,
            n_last_diffusion_steps_to_consider_for_attributions=n_last_diffusion_steps_to_consider_for_attributions,
            get_images_for_all_inference_steps=get_images_for_all_inference_steps,
            **kwargs
        )

        # transform BaseMimicPipelineCallOutput to PipelineExplainerOutput
        output = PipelineExplainerOutput(
            image=output.images[0], nsfw_content_detected=output.nsfw_content_detected,
            all_images_during_generation=output.all_images_during_generation
        )

        if output.nsfw_content_detected:
            raise Exception(
                "NSFW content was detected, it is not possible to provide an explanation. "
                "Try to set `run_safety_checker=False` if you really want to skip the NSFW safety check."
            )

        # Calculate primary attribution scores
        if calculate_attributions:
            if attribution_method == 'grad_x_input':
                output = self._get_attributions(
                    output=output,
                    tokens=tokens,
                    text_embeddings=text_embeddings,
                    explanation_2d_bounding_box=explanation_2d_bounding_box,
                    consider_special_tokens=consider_special_tokens,
                    clean_token_prefixes_and_suffixes=clean_token_prefixes_and_suffixes,
                    n_last_diffusion_steps_to_consider_for_attributions=n_last_diffusion_steps_to_consider_for_attributions,
                    **kwargs
                )
            else:
                raise NotImplementedError("Only `attribution_method='grad_x_input'` is implemented for now")

        if batch_size == 1:
            # squash batch dimension
            for k in ['image', 'token_attributions', 'normalized_token_attributions', 'pixel_attributions',
                      'normalized_pixel_attributions', 'pixel_attributions_heatmap']:
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
        output.normalized_token_attributions = []
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

            # Add normalized
            total = sum([attr for _, attr in output.token_attributions[-1]])
            output.normalized_token_attributions.append(
                [
                    (token, round(100 * attr / total, 3))
                    for token, attr in output.token_attributions[-1]
                ]
            )
        return output

    def _get_attributions(
        self,
        output: PipelineExplainerOutput,
        tokens: List[List[str]],
        text_embeddings: torch.Tensor,
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        **kwargs
    ) -> PipelineExplainerOutput:
        if self.verbose:
            print("Calculating token attributions... ", end='')

        token_attributions = gradient_x_inputs_attribution(
            pred_logits=output.image,
            input_embeds=(text_embeddings,),
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
        *args,
        **kwargs
    ) -> BaseMimicPipelineCallOutput:
        raise NotImplementedError


class BasePipelineExplainer(CorePipelineExplainer):
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
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        run_safety_checker: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        get_images_for_all_inference_steps: bool = False
    ) -> BaseMimicPipelineCallOutput:
        raise NotImplementedError


class BasePipelineImg2ImgExplainer(CorePipelineExplainer):
    @abstractmethod
    def _mimic_pipeline_call(
        self,
        text_input: BatchEncoding,
        text_embeddings: torch.Tensor,
        batch_size: int,
        init_image: Union[torch.FloatTensor, Image],
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        run_safety_checker: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        get_images_for_all_inference_steps: bool = False
    ) -> BaseMimicPipelineCallOutput:
        raise NotImplementedError

    def _preprocess_input(self, **kwargs) -> Dict[str, Any]:
        """
        Converts input image to tensor
        """
        kwargs = super()._preprocess_input(**kwargs)
        if 'init_image' not in kwargs:
            raise TypeError("missing 1 required positional argument: 'init_image'")

        kwargs['init_image'] = preprocess(kwargs['init_image']).to(self.pipe.device).permute(0, 2, 3, 1)
        kwargs['init_image'].requires_grad = True

        return kwargs

    def _get_attributions(
        self,
        output: PipelineExplainerOutput,
        tokens: List[List[str]],
        text_embeddings: torch.Tensor,
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        **kwargs
    ) -> PipelineImg2ImgExplainerOutput:

        if 'init_image' not in kwargs:
            raise TypeError("missing 1 required positional argument: 'init_image'")
        init_image: torch.Tensor = kwargs['init_image']

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

        attributions = gradient_x_inputs_attribution(
            pred_logits=output.image,
            input_embeds=input_embeds,
            explanation_2d_bounding_box=explanation_2d_bounding_box,
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

        output = PipelineImg2ImgExplainerOutput(
            image=output.image,
            nsfw_content_detected=output.nsfw_content_detected,
            all_images_during_generation=output.all_images_during_generation,
            token_attributions=output.token_attributions,
            normalized_token_attributions=output.normalized_token_attributions,
            pixel_attributions=pixel_attributions,
            normalized_pixel_attributions=100 * (pixel_attributions / pixel_attributions.sum()),
            pixel_attributions_heatmap=None # TODO
        )

        if self.verbose:
            print("Done!")

        return output