from typing import Any, Union

import numpy as np

from diffusers_interpret.saliency_map import SaliencyMap


class PixelAttributions(np.ndarray):
    def __init__(self, pixel_attributions: np.ndarray, saliency_map: SaliencyMap) -> None:
        super().__init__(pixel_attributions)
        self.pixel_attributions = pixel_attributions
        self.normalized = 100 * (pixel_attributions / pixel_attributions.sum())
        self.saliency_map = saliency_map

        # Calculate normalized
        total = sum([attr for _, attr in pixel_attributions])
        self.normalized = [
            (pixel, round(100 * attr / total, 3))
            for pixel, attr in pixel_attributions
        ]

    def __getitem__(self, item: Union[str, int]) -> Any:
        return getattr(self, item) if isinstance(item, str) else self.pixel_attributions[item]

    def __setitem__(self, key: Union[str, int], value: Any) -> None:
        setattr(self, key, value)

    def __getattr__(self, attr):