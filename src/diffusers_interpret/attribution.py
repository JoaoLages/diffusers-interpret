from typing import Tuple, Optional

import torch


def gradient_x_inputs_attribution(
    pred_logits: torch.Tensor,
    input_embeds: torch.Tensor,
    explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
) -> torch.Tensor:
    # TODO: add description

    assert len(pred_logits.shape) == 3
    if explanation_2d_bounding_box:
        upper_left, bottom_right = explanation_2d_bounding_box
        pred_logits = pred_logits[upper_left[0]: bottom_right[0], upper_left[1]: bottom_right[1], :]

    # Construct tuple of scalar tensors with all `pred_logits`
    # The code below is equivalent to `tuple_of_pred_logits = tuple(torch.flatten(pred_logits))`,
    #  but for some reason the gradient calculation is way faster if the tensor is flattened like this
    tuple_of_pred_logits = []
    for x in pred_logits:
        for y in x:
            for z in y:
                tuple_of_pred_logits.append(z)
    tuple_of_pred_logits = tuple(tuple_of_pred_logits)

    # get the sum of back-prop gradients for all predictions with respect to the inputs
    grad = torch.autograd.grad(tuple_of_pred_logits, input_embeds)[0]

    # Grad X Input
    grad_x_input = grad * input_embeds

    # Turn into a scalar value for each input token by taking L2 norm
    feature_importance = torch.norm(grad_x_input, dim=-1)

    return feature_importance