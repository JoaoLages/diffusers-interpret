from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any, Tuple, Set

import torch
from PIL import ImageDraw
from diffusers import DiffusionPipeline
from transformers import BatchEncoding, PreTrainedTokenizerBase

from diffusers_interpret.attribution import gradient_x_inputs_attribution
from diffusers_interpret.generated_images import GeneratedImages
from diffusers_interpret.utils import clean_token_from_prefixes_and_suffixes


class BasePipelineExplainer(ABC):
    def __init__(self, pipe: DiffusionPipeline, verbose: bool = True, gradient_checkpointing: bool = False) -> None:
        self.pipe = pipe
        self.verbose = verbose
        self.gradient_checkpointing = gradient_checkpointing
        if self.gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def __call__(
        self,
        prompt: str,
        attribution_method: str = 'grad_x_input',
        explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, # (upper left corner, bottom right corner)
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = 'pil',
        run_safety_checker: bool = False,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        get_images_for_all_inference_steps: bool = True
    ) -> Dict[str, Any]:
        # TODO: add description

        if attribution_method != 'grad_x_input':
            raise NotImplementedError("Only `attribution_method='grad_x_input'` is implemented for now")

        if isinstance(prompt, str):
            batch_size = 1 # TODO: make compatible with bigger batch sizes
        else:
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")

        # TODO: add asserts for out of bounds
        if explanation_2d_bounding_box:
            pass

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
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=generator,
            output_type=None,
            run_safety_checker=run_safety_checker,
            n_last_diffusion_steps_to_consider_for_attributions=n_last_diffusion_steps_to_consider_for_attributions,
            get_images_for_all_inference_steps=get_images_for_all_inference_steps
        )

        if output['nsfw_content_detected']:
            raise Exception(
                "NSFW content was detected, it is not possible to provide an explanation. "
                "Try to set `run_safety_checker=False` if you really want to skip the NSFW safety check."
            )

        # Get primary attribution scores
        output['token_attributions'] = None
        output['normalized_token_attributions'] = None
        if calculate_attributions and attribution_method == 'grad_x_input':

            if self.verbose:
                print("Calculating token attributions... ", end='')

            token_attributions = gradient_x_inputs_attribution(
                pred_logits=output['sample'][0], input_embeds=text_embeddings,
                explanation_2d_bounding_box=explanation_2d_bounding_box
            )
            token_attributions = token_attributions.detach().cpu().numpy()

            # remove special tokens
            assert len(token_attributions) == len(tokens)
            output['token_attributions'] = []
            output['normalized_token_attributions'] = []
            for sample_token_attributions, sample_tokens in zip(token_attributions, tokens):
                assert len(sample_token_attributions) == len(sample_tokens)

                # Add token attributions
                output['token_attributions'].append([])
                for attr, token in zip(sample_token_attributions, sample_tokens):
                    if consider_special_tokens or token not in self.special_tokens_attributes:

                        if clean_token_prefixes_and_suffixes:
                            token = clean_token_from_prefixes_and_suffixes(token)

                        output['token_attributions'][-1].append(
                            (token, attr)
                        )

                # Add normalized
                total = sum([attr for _, attr in output['token_attributions'][-1]])
                output['normalized_token_attributions'].append(
                    [
                        (token, round(100 * attr / total, 3))
                        for token, attr in output['token_attributions'][-1]
                    ]
                )

            if self.verbose:
                print("Done!")

        else:
            raise NotImplementedError("Only `attribution_method='grad_x_input'` is implemented for now")

        if batch_size == 1:
            # squash batch dimension
            for k in ['sample', 'token_attributions', 'normalized_token_attributions']:
                if output[k] is not None:
                    output[k] = output[k][0]
            if output['all_samples_during_generation']:
                output['all_samples_during_generation'] = [b[0] for b in output['all_samples_during_generation']]

        # convert to PIL Image if requested
        # also draw bounding box in the last image if requested
        if output['all_samples_during_generation'] or output_type == "pil":
            all_samples = GeneratedImages(
                all_generated_images=output['all_samples_during_generation'] or [output['sample']],
                pipe=self.pipe,
                remove_batch_dimension=batch_size==1,
                prepare_image_slider=bool(output['all_samples_during_generation'])
            )
            if output['all_samples_during_generation']:
                output['all_samples_during_generation'] = all_samples
                sample = output['all_samples_during_generation'][-1]
            else:
                sample = all_samples[-1]

            if explanation_2d_bounding_box:
                draw = ImageDraw.Draw(sample)
                draw.rectangle(explanation_2d_bounding_box, outline="red")

            if output_type == "pil":
                output['sample'] = sample

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
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = 'pil',
        run_safety_checker: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        get_images_for_all_inference_steps: bool = False
    ) -> Dict[str, Any]:
        raise NotImplementedError
