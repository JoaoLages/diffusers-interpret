from typing import List

import torch
from PIL.Image import Image
from diffusers import DiffusionPipeline


def clean_token_from_prefixes_and_suffixes(token: str) -> str:
    """
    Removes all the known token prefixes and suffixes

    Args:
        token (`str`): string with token

    Returns:
        `str`: clean token
    """

    # removes T5 prefix
    token = token.lstrip('▁')

    # removes BERT/GPT-2 prefix
    token = token.lstrip('Ġ')

    # removes CLIP suffix
    token = token.rstrip('</w>')

    return token


def transform_images_to_pil_format(all_generated_images: List[torch.Tensor], pipe: DiffusionPipeline) -> List[List[Image]]:
    pil_images = []
    for im in all_generated_images:
        if isinstance(im, torch.Tensor):
            im = im.detach().cpu().numpy()
        im = pipe.numpy_to_pil(im)
        pil_images.append(im)
    return pil_images