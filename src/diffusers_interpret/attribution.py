from typing import Dict

import torch
from tqdm.auto import tqdm


def gradient_x_inputs_attribution(pred_logits: torch.Tensor, input_embeds: torch.Tensor) -> Dict[str, torch.Tensor]:
    # TODO: add description

    assert len(pred_logits.shape) == 3

    # retrain gradient of input embeddings
    input_embeds.retain_grad()

    i, j , k = pred_logits.shape
    with tqdm(total=i * j * k, desc="Calculating InputXGradient attributions") as pbar:
        attrs = {}
        for i, x in enumerate(pred_logits):
            for j, y in enumerate(x):
                for k, pred_logit in enumerate(y):
                    # back-prop gradient
                    grad = torch.autograd.grad(pred_logit, input_embeds, retain_graph=True)[0] # TODO: transform in batch call

                    # Grad X Input
                    grad_enc_x_input = grad * input_embeds
                    grad_x_input = grad_enc_x_input

                    # Turn into a scalar value for each input token by taking L2 norm
                    feature_importance = torch.norm(grad_x_input, dim=1)

                    # Normalize so we can show scores as percentages
                    feature_importance_normalized = feature_importance / torch.sum(feature_importance)

                    # save
                    attrs[f"{i},{j},{k}"] = feature_importance_normalized

                    # update prog bar
                    pbar.update(1)

    # Zero the gradient for the tensor so next backward() calls don't have
    # gradients accumulating
    input_embeds.grad.data.zero_()

    return attrs