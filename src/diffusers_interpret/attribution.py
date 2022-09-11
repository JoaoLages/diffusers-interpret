from typing import Tuple, Optional, List

import torch

from diffusers_interpret.data import AttributionAlgorithm


def gradients_attribution(
    pred_logits: torch.Tensor,
    input_embeds: Tuple[torch.Tensor],
    attribution_algorithms: List[AttributionAlgorithm],
    explanation_2d_bounding_box: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    retain_graph: bool = False
) -> List[torch.Tensor]:
    # TODO: add description

    assert len(pred_logits.shape) == 3
    if explanation_2d_bounding_box:
        upper_left, bottom_right = explanation_2d_bounding_box
        pred_logits = pred_logits[upper_left[0]: bottom_right[0], upper_left[1]: bottom_right[1], :]

    assert len(input_embeds) == len(attribution_algorithms)

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
    if torch.is_autocast_enabled():
        # FP16 may cause NaN gradients https://github.com/pytorch/pytorch/issues/40497
        # TODO: this is still an issue, the code below does not solve it
        with torch.autocast(input_embeds[0].device.type, enabled=False):
            grads = torch.autograd.grad(tuple_of_pred_logits, input_embeds, retain_graph=retain_graph)
    else:
        grads = torch.autograd.grad(tuple_of_pred_logits, input_embeds, retain_graph=retain_graph)

    if torch.isnan(grads[-1]).any():
        raise RuntimeError(
            "Found NaNs while calculating gradients. "
            "This is a known issue of FP16 (https://github.com/pytorch/pytorch/issues/40497).\n"
            "Try to rerun the code or deactivate FP16 to not face this issue again."
        )

    # Aggregate
    aggregated_grads = []
    for grad, inp, attr_alg in zip(grads, input_embeds, attribution_algorithms):

        if attr_alg == AttributionAlgorithm.GRAD_X_INPUT:
            aggregated_grads.append(torch.norm(grad * inp, dim=-1))
        elif attr_alg == AttributionAlgorithm.MAX_GRAD:
            aggregated_grads.append(grad.abs().max(-1).values)
        elif attr_alg == AttributionAlgorithm.MEAN_GRAD:
            aggregated_grads.append(grad.abs().mean(-1).values)
        elif attr_alg == AttributionAlgorithm.MIN_GRAD:
            aggregated_grads.append(grad.abs().min(-1).values)
        else:
            raise NotImplementedError(f"aggregation type `{attr_alg}` not implemented")

    return aggregated_grads