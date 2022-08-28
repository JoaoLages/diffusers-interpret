import torch, time


def gradient_x_inputs_attribution(pred_logits: torch.Tensor, input_embeds: torch.Tensor) -> torch.Tensor:
    # TODO: add description

    assert len(pred_logits.shape) == 3

    # Construct tuple of scalar tensors with all `pred_logits`
    tuple_of_pred_logits = tuple(torch.flatten(pred_logits))

    # back-prop gradients for all predictions with respect to the inputs and sum them
    print("calculating gradients")
    start = time.time()
    grad = torch.autograd.grad(tuple_of_pred_logits, input_embeds, retain_graph=True)[0]
    print("Done", time.time()-start)

    # Grad X Input
    grad_x_input = grad * input_embeds

    # Turn into a scalar value for each input token by taking L2 norm
    feature_importance = torch.norm(grad_x_input, dim=1)

    # Normalize so we can show scores as percentages
    feature_importance_normalized = feature_importance / torch.sum(feature_importance)

    return feature_importance_normalized