from dataclasses import dataclass
from typing import Union, List, Optional, Tuple, Any

import torch
from PIL.Image import Image

from diffusers_interpret.generated_images import GeneratedImages
from diffusers_interpret.saliency_map import SaliencyMap


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
    input_saliency_map: Optional[SaliencyMap] = None


@dataclass
class AttributionMethod:
    tokens_attribution_method: str = 'grad_x_input'
    pixels_attribution_method: Optional[str] = 'max_grad'