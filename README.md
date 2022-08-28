<div align="center">

# Diffusers-Interpret 🤗🧨🕵️‍♀️

![PyPI - Latest Package Version](https://img.shields.io/pypi/v/ratransformers?logo=pypi&style=flat&color=orange) ![GitHub - License](https://img.shields.io/github/license/JoaoLages/diffusers-interpret?logo=github&style=flat&color=green)

`diffusers-interpret` is a model explainability tool built on top of [🤗 Diffusers](https://github.com/huggingface/diffusers).
</div>

## Installation

Install directly from PyPI:

    pip install diffusers-interpret

## Usage

Let's see how we can interpret the **[new 🎨🎨🎨 Stable Diffusion](https://github.com/huggingface/diffusers#new--stable-diffusion-is-now-fully-compatible-with-diffusers)!**

```python
# make sure you're logged in with `huggingface-cli login`
import torch
from contextlib import nullcontext
from diffusers import StableDiffusionPipeline
from diffusers_interpret import StableDiffusionPipelineExplainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    
    # FP16 is not working for 'cpu'
    revision='fp16' if device != 'cpu' else None,
    torch_dtype=torch.float16 if device != 'cpu' else None
).to(device)

# pass pipeline to the explainer class
explainer = StableDiffusionPipelineExplainer(pipe)

# generate an image with `explainer`
prompt = "a photo of an astronaut riding a horse on mars"
generator = torch.Generator(device).manual_seed(1024)
with torch.autocast('cuda') if device == 'cuda' else nullcontext():
    output = explainer(
        prompt, 
        num_inference_steps=15,
        generator=generator
    )
```

Check more implementation examples in [here](https://github.com/JoaoLages/diffusers-interpret/blob/main/notebooks/).

## Future Development
- [ ] Add example for `diffusers_interpret.LDMTextToImagePipelineExplainer`
- [ ] Do not require another generation every time the `explanation_2d_bounding_box` argument is changed
- [ ] Add interactive bounding-box and token attributions visualization
- [ ] Add more explainability methods

## Contributing
Feel free to open an [Issue](https://github.com/JoaoLages/diffusers-interpret/issues) or create a [Pull Request](https://github.com/JoaoLages/diffusers-interpret/pulls) and let's get started 🚀