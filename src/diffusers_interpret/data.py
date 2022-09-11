from dataclasses import dataclass
from enum import Enum
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
    explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None  # (upper left corner, bottom right corner)
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


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class AttributionAlgorithm(ExplicitEnum):
    """
    Possible values for `tokens_attribution_method` and `pixels_attribution_method` arguments in `AttributionMethods`
    """
    GRAD_X_INPUT = "grad_x_input"
    MAX_GRAD = "max_grad"
    MEAN_GRAD = "mean_grad"
    MIN_GRAD = "min_grad"


@dataclass
class AttributionMethods:
    tokens_attribution_method: Union[str, AttributionAlgorithm] = AttributionAlgorithm.GRAD_X_INPUT
    pixels_attribution_method: Optional[Union[str, AttributionAlgorithm]] = AttributionAlgorithm.MAX_GRAD