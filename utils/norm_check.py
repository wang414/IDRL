import torch

def get_inf_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0
    device = parameters[0].grad.device
    norms = [p.grad.detach().abs().max().to(device) for p in parameters]
    total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    return total_norm.item()