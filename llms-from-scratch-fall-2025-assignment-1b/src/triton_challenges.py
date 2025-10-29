import torch
import triton
import triton.language as tl

# WRITE YOUR CODE HERE!
def outer_product_with_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: