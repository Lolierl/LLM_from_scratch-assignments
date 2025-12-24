import torch
import torch.nn.functional as F

class StickBreakingAttention(torch.autograd.Function):
    """
    PyTorch implementation of Stick-Breaking Attention with a custom backward pass.
    
    This implementation is causal and operates in log-space for numerical stability.
    """

    @staticmethod
    def forward(ctx, q, k, v):
        """
        Forward pass for Stick-Breaking Attention.
        
        Args:
            q, k, v: Tensors of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        
        device = q.device
        bs, h, l, d = q.shape
        scale = 1.0 / d**0.5
        
        z = torch.einsum("bhid,bhjd->bhij", q, k) * scale
        causal_mask = torch.tril(torch.ones(l, l, device=device, dtype=torch.bool))
        z_masked = z.masked_fill(~causal_mask, -float('inf'))
        
        log_beta = F.logsigmoid(z_masked)
        log_alpha = F.logsigmoid(-z_masked)

        log_alpha_strict_lower = torch.tril(log_alpha, diagonal=-1)
        flipped_log_alpha = torch.flip(log_alpha_strict_lower, dims=[-1])
        flipped_cumsum = torch.cumsum(flipped_log_alpha, dim=-1)
        suffix_sum_log_alpha = torch.flip(flipped_cumsum, dims=[-1])
        sum_term = F.pad(suffix_sum_log_alpha, (0, 1))[:, :, :, 1:]
        
        log_A_tilde = log_beta + sum_term
        A = torch.exp(log_A_tilde).masked_fill(~causal_mask, 0.0)
        o = A @ v
        
        ctx.save_for_backward(q, k, v, A, z)
        ctx.scale = scale
        
        return o

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass for Stick-Breaking Attention.
        
        Args:
            dO: Gradient of the loss with respect to the output O.
            
        Returns:
            Gradients w.r.t. q, k, and v.
        """
        q, k, v, A, z = ctx.saved_tensors
        scale = ctx.scale
        bs, h, l, d = q.shape
        device = q.device
        
        dL_dV = A.transpose(-2, -1) @ dO
        dL_dA = dO @ v.transpose(-2, -1)
        dL_dA_tilde = dL_dA * A
        sigma_z = torch.sigmoid(z)
        sigma_neg_z = 1.0 - sigma_z
        dL_dz = torch.zeros_like(z)
        causal_mask = torch.tril(torch.ones(l, l, device=device, dtype=torch.bool))

        # Think about why we compute diagonal and triangle differently.
        diag_dL_dA_tilde = torch.diagonal(dL_dA_tilde, dim1=-2, dim2=-1)
        diag_sigma_neg_z = torch.diagonal(sigma_neg_z, dim1=-2, dim2=-1)
        diag_dL_dz = diag_sigma_neg_z * diag_dL_dA_tilde
        dL_dz.diagonal(dim1=-2, dim2=-1).copy_(diag_dL_dz)
        
        dL_dA_tilde_causal = dL_dA_tilde.masked_fill(~causal_mask, 0.0)
        cumsum_dL_dA_tilde = torch.cumsum(dL_dA_tilde_causal, dim=-1)
        lower_tri_part = dL_dA_tilde - sigma_z * cumsum_dL_dA_tilde
        strict_lower_mask = torch.tril(torch.ones(l, l, device=device, dtype=torch.bool), diagonal=-1)
        strict_lower_mask = strict_lower_mask.unsqueeze(0).unsqueeze(0).expand(dL_dz.shape)
        
        dL_dq = (dL_dz @ k) * scale
        dL_dk = (dL_dz.transpose(-2, -1) @ q) * scale
        
        return dL_dq, dL_dk, dL_dV
