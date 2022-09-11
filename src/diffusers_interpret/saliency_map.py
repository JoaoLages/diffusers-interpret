from typing import Optional, Union

import cv2
import cmapy
import numpy as np
import torch
from PIL.Image import Image
from matplotlib import pyplot as plt


class SaliencyMap:
    def __init__(
        self,
        images: np.ndarray,
        pixel_attributions: np.ndarray,
        masks: Optional[Union[torch.FloatTensor, Image]] = None
    ):
        assert len(images) == len(pixel_attributions)
        self.imgs = np.float32(images)
        self.pixel_attributions = pixel_attributions
        self.masks = masks

    def show(self, cmap='jet', image_weight=0.5, tight=True, **kwargs) -> None:
        for img, attrs in zip(self.imgs, self.pixel_attributions):
            saliency_map = cv2.applyColorMap(np.uint8(attrs), cmapy.cmap(cmap))
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
            saliency_map = np.float32(saliency_map) / 255.0

            overlayed = (1 - image_weight) * saliency_map + image_weight * img
            overlayed = overlayed / np.max(overlayed)
            overlayed = np.uint8(255 * overlayed)

            # Visualize the image and the saliency map
            fig, ax = plt.subplots(1, 3, **kwargs)
            ax[0].imshow(img)
            ax[0].axis('off')
            ax[0].title.set_text('Image')

            ax[1].imshow(saliency_map)
            ax[1].axis('off')
            ax[1].title.set_text('Pixel attributions')

            ax[2].imshow(overlayed)
            ax[2].axis('off')
            ax[2].title.set_text('Image Overlayed')

            if tight:
                plt.tight_layout()
            plt.show()