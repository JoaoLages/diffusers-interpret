import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL.Image import Image

from diffusers_interpret.generated_images import GeneratedImages
from diffusers_interpret.pixel_attributions import PixelAttributions
from diffusers_interpret.saliency_map import SaliencyMap
from diffusers_interpret.token_attributions import TokenAttributions


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
    """
    Output class for BasePipelineExplainer.__call__ if `init_image=None` and `explanation_2d_bounding_box=None`

    Args:
        image (`Image` or `torch.Tensor`)
            The denoised PIL output image or torch.Tensor of shape `(height, width, num_channels)`.
        nsfw_content_detected (`Optional[bool]`)
            A flag denoting whether the generated image likely represents "not-safe-for-work"
            (nsfw) content.
        all_images_during_generation (`Optional[Union[GeneratedImages, List[torch.Tensor]]]`)
            A GeneratedImages object to visualize all the generated images during diffusion OR a list of tensors of those images
        token_attributions (`Optional[TokenAttributions]`)
            TokenAttributions that contains a list of tuples with (token, token_attribution)
    """
    image: Union[Image, torch.Tensor]
    nsfw_content_detected: Optional[bool] = None
    all_images_during_generation: Optional[Union[GeneratedImages, List[torch.Tensor]]] = None
    token_attributions: Optional[TokenAttributions] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getattr__(self, attr):
        if attr == 'normalized_token_attributions':
            warnings.warn(
                f"`normalized_token_attributions` is deprecated as an attribute of `{self.__class__.__name__}` "
                f"and will be removed in a future version. Consider using `output.token_attributions.normalized` instead",
                DeprecationWarning, stacklevel=2
            )
            return self.token_attributions.normalized
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")


@dataclass
class PipelineExplainerForBoundingBoxOutput(PipelineExplainerOutput):
    """
    Output class for BasePipelineExplainer.__call__ if `init_image=None` and `explanation_2d_bounding_box is not None`

    Args:
        image (`Image` or `torch.Tensor`)
            The denoised PIL output image or torch.Tensor of shape `(height, width, num_channels)`.
        nsfw_content_detected (`Optional[bool]`)
            A flag denoting whether the generated image likely represents "not-safe-for-work"
            (nsfw) content.
        all_images_during_generation (`Optional[Union[GeneratedImages, List[torch.Tensor]]]`)
            A GeneratedImages object to visualize all the generated images during diffusion OR a list of tensors of those images
        token_attributions (`Optional[TokenAttributions]`)
            TokenAttributions that contains a list of tuples with (token, token_attribution)
        explanation_2d_bounding_box: (`Tuple[Tuple[int, int], Tuple[int, int]]`)
            Tuple with the bounding box coordinates where the attributions were calculated for.
            The tuple is like (upper left corner, bottom right corner). Example: `((0, 0), (300, 300))`
    """
    explanation_2d_bounding_box: Tuple[Tuple[int, int], Tuple[int, int]] = None  # (upper left corner, bottom right corner)


@dataclass
class PipelineImg2ImgExplainerOutput(PipelineExplainerOutput):
    """
    Output class for BasePipelineExplainer.__call__ if `init_image is not None` and `explanation_2d_bounding_box=None`

    Args:
        image (`Image` or `torch.Tensor`)
            The denoised PIL output image or torch.Tensor of shape `(height, width, num_channels)`.
        nsfw_content_detected (`Optional[bool]`)
            A flag denoting whether the generated image likely represents "not-safe-for-work"
            (nsfw) content.
        all_images_during_generation (`Optional[Union[GeneratedImages, List[torch.Tensor]]]`)
            A GeneratedImages object to visualize all the generated images during diffusion OR a list of tensors of those images
        token_attributions (`Optional[TokenAttributions]`)
            TokenAttributions that contains a list of tuples with (token, token_attribution)
        pixel_attributions (`Optional[PixelAttributions]`)
            PixelAttributions that is a numpy array of shape `(height, width)` with an attribution score per pixel in the input image
        input_saliency_map (`Optional[SaliencyMap]`)
            A SaliencyMap object to visualize the pixel attributions of the input image
    """
    pixel_attributions: Optional[PixelAttributions] = None

    def __getattr__(self, attr):
        if attr == 'normalized_pixel_attributions':
            warnings.warn(
                f"`normalized_pixel_attributions` is deprecated as an attribute of `{self.__class__.__name__}` "
                f"and will be removed in a future version. Consider using `output.pixel_attributions.normalized` instead",
                DeprecationWarning, stacklevel=2
            )
            return self.token_attributions.normalized
        elif attr == 'input_saliency_map':
            return self.pixel_attributions.saliency_map
        return super().__getattr__(attr)


@dataclass
class PipelineImg2ImgExplainerForBoundingBoxOutputOutput(PipelineExplainerForBoundingBoxOutput, PipelineImg2ImgExplainerOutput):
    """
    Output class for BasePipelineExplainer.__call__ if `init_image is not None` and `explanation_2d_bounding_box=None`

    Args:
        image (`Image` or `torch.Tensor`)
            The denoised PIL output image or torch.Tensor of shape `(height, width, num_channels)`.
        nsfw_content_detected (`Optional[bool]`)
            A flag denoting whether the generated image likely represents "not-safe-for-work"
            (nsfw) content.
        all_images_during_generation (`Optional[Union[GeneratedImages, List[torch.Tensor]]]`)
            A GeneratedImages object to visualize all the generated images during diffusion OR a list of tensors of those images
        token_attributions (`Optional[TokenAttributions]`)
            TokenAttributions that contains a list of tuples with (token, token_attribution)
        pixel_attributions (`Optional[np.ndarray]`)
            PixelAttributions that is a numpy array of shape `(height, width)` with an attribution score per pixel in the input image
        input_saliency_map (`Optional[SaliencyMap]`)
            A SaliencyMap object to visualize the pixel attributions of the input image
        explanation_2d_bounding_box: (`Tuple[Tuple[int, int], Tuple[int, int]]`)
            Tuple with the bounding box coordinates where the attributions were calculated for.
            The tuple is like (upper left corner, bottom right corner). Example: `((0, 0), (300, 300))`
    """
    pass


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