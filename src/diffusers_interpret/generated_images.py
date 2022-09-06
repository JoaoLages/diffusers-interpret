import base64
import json
import os
import random
from typing import List, Union

import torch
from IPython import display as d
from PIL.Image import Image
from diffusers import DiffusionPipeline

import diffusers_interpret
from diffusers_interpret.utils import transform_images_to_pil_format


class GeneratedImages:
    def __init__(
            self,
            all_generated_images: List[torch.Tensor],
            pipe: DiffusionPipeline,
            remove_batch_dimension: bool = True,
            prepare_image_slider: bool = True
    ) -> None:

        assert all_generated_images, "Can't create GeneratedImages object with empty `all_generated_images`"

        # Convert images to PIL and draw box if requested
        self.images = []
        for list_im in transform_images_to_pil_format(all_generated_images, pipe):
            batch_images = []
            for im in list_im:
                batch_images.append(im)

            if remove_batch_dimension:
                self.images.extend(batch_images)
            else:
                self.images.append(batch_images)

        self.loading_iframe = None
        self.image_slider_iframe = None
        if prepare_image_slider:
            self.prepare_image_slider()

    def prepare_image_slider(self) -> None:
        """
        Creates auxiliary HTML file to be displayed in self.__repr__
        """

        # Get data dir
        image_slider_dir = os.path.join(os.path.dirname(diffusers_interpret.__file__), "dataviz", "image-slider")

        # Convert images to base64
        json_payload = []
        for i, image in enumerate(self.images):
            image.save(f"{image_slider_dir}/to_delete.png")
            with open(f"{image_slider_dir}/to_delete.png", "rb") as image_file:
                json_payload.append(
                    {"image": "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')}
                )
        os.remove(f"{image_slider_dir}/to_delete.png")

        # get HTML file
        with open(os.path.join(image_slider_dir, "index.html")) as fp:
            html = fp.read()

        # get CSS file
        with open(os.path.join(image_slider_dir, "css/index.css")) as fp:
            css = fp.read()

        # get JS file
        with open(os.path.join(image_slider_dir, "js/index.js")) as fp:
            js = fp.read()

        # replace CSS text in CSS file
        html = html.replace("""<link href="css/index.css" rel="stylesheet" />""",
                            f"""<style type="text/css">\n{css}</style>""")

        # replace JS text in HTML file
        html = html.replace("""<script type="text/javascript" src="js/index.js"></script>""", ""
                            f"""<script type="text/javascript">\n{js}</script>""")

        # get html with image slider JS call
        index = html.find("<!-- INSERT STARTING SCRIPT HERE -->")
        add = """
            <script type="text/javascript">
              ((d) => {
                const $body = d.querySelector("body");
    
                if ($body) {
                  $body.addEventListener("INITIALIZE_IS_READY", ({ detail }) => {
                    const initialize = detail?.initialize ?? null;
    
                    if (initialize) initialize(%s);
                  });
                }
              })(document);
            </script>
        """ % json.dumps(json_payload)
        html_with_image_slider = html[:index] + add + html[index:]

        # save files and load IFrame to be displayed in self.__repr__
        with open(os.path.join(image_slider_dir, "loading.html"), 'w') as fp:
            fp.write(html)
        with open(os.path.join(image_slider_dir, "final.html"), 'w') as fp:
            fp.write(html_with_image_slider)

        self.loading_iframe = d.IFrame(
            os.path.relpath(
                os.path.join(os.path.dirname(diffusers_interpret.__file__), "dataviz", "image-slider", "loading.html"),
                '.'
            ),
            width="100%", height="400px"
        )

        self.image_slider_iframe = d.IFrame(
            os.path.relpath(
                os.path.join(os.path.dirname(diffusers_interpret.__file__), "dataviz", "image-slider", "final.html"),
                '.'
            ),
            width="100%", height="400px"
        )

    def __getitem__(self, item: int) -> Union[Image, List[Image]]:
        return self.images[item]

    def show(self, width: Union[str, int] = "100%", height: Union[str, int] = "400px") -> None:

        if len(self.images) == 0:
            raise Exception("`self.images` is an empty list, can't show any images")

        if isinstance(self.images[0], list):
            raise NotImplementedError("GeneratedImages.show visualization is not supported "
                                      "when `self.images` is a list of lists of images")

        if self.image_slider_iframe is None:
            self.prepare_image_slider()

        # display loading
        self.loading_iframe.width = width
        self.loading_iframe.height = height
        display = d.display(self.loading_iframe, display_id=random.randint(0, 9999999))

        # display image slider
        self.image_slider_iframe.width = width
        self.image_slider_iframe.height = height
        display.update(self.image_slider_iframe)
