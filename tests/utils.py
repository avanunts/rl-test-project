import torch

def must_coincide(t1: torch.Tensor, t2: torch.Tensor, eps: float=1.0e-3): 
    is_diff = torch.any(torch.abs(t1 - t2) > eps)
    if is_diff.numpy(): 
        print(f'two tensors do not coincide, t1: {t1}, t2: {t2}')
        print(torch.abs(t1 - t2) > eps)
        assert False