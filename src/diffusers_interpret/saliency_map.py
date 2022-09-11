import cv2
import numpy as np
from matplotlib import pyplot as plt


class SaliencyMap:
    def __init__(
        self,
        image: np.ndarray,
        normalized_pixel_attributions: np.ndarray
    ):
        self.img = image
        self.img_greyscale = np.float32(image) / 255
        self.normalized_pixel_attributions = normalized_pixel_attributions

    def show(self, colormap=cv2.COLORMAP_JET, image_weight=0.5, tight=True, **kwargs) -> None:
        saliency_map = cv2.applyColorMap(np.uint8(255 * self.normalized_pixel_attributions), colormap)
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
        saliency_map = np.float32(saliency_map) / 255

        overlayed = (1 - image_weight) * saliency_map + image_weight * self.img_greyscale
        overlayed = overlayed / np.max(overlayed)
        overlayed = np.uint8(255 * overlayed)

        # Visualize the image and the saliency map
        fig, ax = plt.subplots(1, 3, **kwargs)
        ax[0].imshow(self.img)
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