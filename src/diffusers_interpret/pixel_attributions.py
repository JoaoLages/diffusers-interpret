from typing import Any, Union

import numpy as np

from diffusers_interpret.saliency_map import SaliencyMap


class PixelAttributions(np.ndarray):
    def __new__(cls, pixel_attributions: np.ndarray, saliency_map: SaliencyMap) -> "PixelAttributions":
        # Construct new ndarray
        obj = np.asarray(pixel_attributions).view(cls)
        obj.pixel_attributions = pixel_attributions
        obj.normalized = 100 * (pixel_attributions / pixel_attributions.sum())
        obj.saliency_map = saliency_map

        # Calculate normalized
        obj.normalized = 100 * (pixel_attributions / pixel_attributions.sum())

        return obj

    def __getitem__(self, item: Union[str, int]) -> Any:
        return getattr(self, item) if isinstance(item, str) else self.pixel_attributions[item]

    def __setitem__(self, key: Union[str, int], value: Any) -> None:
        setattr(self, key, value)

    def __repr__(self) -> str:
        return self.pixel_attributions.__repr__()