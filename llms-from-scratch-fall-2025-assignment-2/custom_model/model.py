# add the code in your model.py here
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple

@dataclass
class ModelArgs:
    vocab_size: int
    embed_dim: int
    inter_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    n_layers: int
    rope_theta: float

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.view(-1)  
        out = self.weight[flat]
        return out.view(*input_ids.shape, -1)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.matmul(self.weight.t())
        return out 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)**2
        rms = norm_x / x.size(-1)
        x_normed = x / (rms + self.eps).sqrt()
        return self.weight * x_normed

class MLP(nn.Module): 
    def __init__(self, embed_dim: int, inter_dim: int):
        super().__init__()
        self.gate_proj = Linear(embed_dim, inter_dim)
        self.up_proj = Linear(embed_dim, inter_dim)
        self.down_proj = Linear(inter_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 100000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

        idx = torch.arange(0, head_dim, 2, dtype=torch.float32)  

        inv_freq = 1.0 / (theta ** (idx / head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = position_ids.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0)
        C_pos = torch.cos(angles).detach()
        S_pos = torch.sin(angles).detach()
        return C_pos, S_pos

def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated
    
class KVCache:
    def __init__(self, num_layers: int):
        self.cache = [(None, None) for _ in range(num_layers)]
    
    def update(self, layer_idx, k, v):
        cached_k, cached_v = self.cache[layer_idx]
        if cached_k is None:
            self.cache[layer_idx] = (k, v)
            return
        self.cache[layer_idx] = (
            torch.cat([cached_k, k], dim=2),
            torch.cat([cached_v, v], dim=2)
        )
    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cache[layer_idx]
    
class MHA(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, num_attention_heads: int,
                num_key_value_heads: int, layer_idx: int):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        self.q_proj = Linear(embed_dim, num_attention_heads * head_dim)
        self.k_proj = Linear(embed_dim, num_key_value_heads * head_dim)
        self.v_proj = Linear(embed_dim, num_key_value_heads * head_dim)
        self.o_proj = Linear(num_attention_heads * head_dim, embed_dim)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_values: Optional['KVCache'] = None) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        C_pos, S_pos = position_embeddings

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_embedding(q, C_pos, S_pos)
        k = apply_rotary_embedding(k, C_pos, S_pos)

        if past_key_values is not None:
            past_key_values.update(self.layer_idx, k, v)
            k, v = past_key_values.get(self.layer_idx)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask).contiguous().transpose(1, 2).reshape(B, L, self.num_attention_heads * self.head_dim)
        return self.o_proj(attn_output)

class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = MHA(args.embed_dim, args.head_dim, args.num_attention_heads, args.num_key_value_heads, layer_idx)
        self.mlp = MLP(args.embed_dim, args.inter_dim)
        self.attn_norm = RMSNorm(args.embed_dim)
        self.mlp_norm = RMSNorm(args.embed_dim)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, **attention_kwargs) -> torch.Tensor:
        attention_mask = attention_kwargs.get("attention_mask", None)
        position_embeddings = attention_kwargs.get("position_embeddings", None)
        past_key_values = attention_kwargs.get("past_key_values", None)

        x = x + self.attn(self.attn_norm(x), attention_mask, position_embeddings, past_key_values=past_key_values)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed = Embedding(args.vocab_size, args.embed_dim)
        self.layers = nn.ModuleList([Block(args, i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.embed_dim)
        self.lm_head = Linear(args.embed_dim, args.vocab_size)
        self.rotary_emb = RotaryEmbedding(args.head_dim, args.rope_theta)
        self.apply(self._init_weights)
        L = args.n_layers
        self._init_o_proj(L)
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module):
        if isinstance(module, Embedding) or isinstance(module, Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def _init_o_proj(self, L):
        std = math.sqrt(0.02 / (2 * L))
        for layer in self.layers:
            o_proj = layer.attn.o_proj
            nn.init.normal_(o_proj.weight, mean=0.0, std=std)
    
    def forward(self, *, input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.Tensor]=None,
        past_key_values: Optional["KVCache"]=None):
        
        # causal mask (B, L, L)
        B, L = input_ids.shape
        if past_key_values is not None and past_key_values.cache[0][0] != None:
            cache_len = past_key_values.cache[0][0].size(2)
            total_len = cache_len + L
            causal_mask = torch.tril(torch.ones((total_len, total_len), device=input_ids.device, dtype=torch.bool))
            causal_mask = causal_mask[-L:, :].unsqueeze(0).expand(B, -1, -1)
        else:
            cache_len = 0
            total_len = L
            causal_mask = torch.tril(torch.ones((input_ids.size(1), input_ids.size(1)), device=input_ids.device)).to(torch.bool)
            causal_mask = causal_mask.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        
        
        kv_idx = torch.arange(total_len, device=input_ids.device)
        q_idx = torch.arange(L, device=input_ids.device).unsqueeze(1)
        causal_qk = (kv_idx.unsqueeze(0) <= (cache_len + q_idx)).to(torch.bool)  # (L, total_len)

        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)  # expected (B, total_len)
            # key padding mask (B, 1, total_len) broadcasted over L
            key_padding = attention_mask.unsqueeze(1)  # (B, 1, total_len)
            # valid_query: which of current queries are padding? (B, L, 1)
            valid_query = attention_mask[:, -L:].unsqueeze(-1)  # (B, L, 1)
            # combine -> (B, L, total_len)
            final_mask = key_padding & causal_qk.unsqueeze(0)
            # make sure padded queries don't attend (set mask False where query is padding)
            final_mask = torch.where(valid_query, final_mask, torch.zeros_like(final_mask))

            # pad_rows indicates which of the current queries are padding (B, L)
            pad_rows = (attention_mask[:, -L:] == 0)
            eye_total = torch.zeros((L, total_len), dtype=torch.bool, device=input_ids.device)
            q_idx_local = torch.arange(L, device=input_ids.device)
            key_pos = cache_len + q_idx_local
            eye_total[q_idx_local, key_pos] = True
            # expand to batch and replace rows where query is padding so that padded queries only attend to themselves
            final_mask = torch.where(pad_rows.unsqueeze(-1), eye_total.unsqueeze(0), final_mask)
        else:
            final_mask = causal_qk.unsqueeze(0).expand(B, -1, -1)  # (B, L, total_len)
        
        if position_ids is None:
            if past_key_values is not None and len(past_key_values.cache) > 0 and past_key_values.cache[0][0] != None:
                cache_len = past_key_values.cache[0][0].size(2)
                position_ids = torch.arange(cache_len, cache_len + L, device=input_ids.device).unsqueeze(0).expand(B, -1)
            else:
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=final_mask, position_embeddings=self.rotary_emb(position_ids), past_key_values=past_key_values)
        x = self.norm(x)
        return self.lm_head(x)
    
    @torch.inference_mode()
    def generate(self, *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = -1,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None):

        B, L = input_ids.shape
        device = input_ids.device
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        cache = KVCache(self.layers.__len__())
        
        for step in range(max_new_tokens if max_new_tokens > 0 else 1000):
            if step == 0:
                current_input = input_ids
                current_mask = attention_mask
            else:
                current_input = next_token
                current_mask = attention_mask
            
            logits = self(
                input_ids=current_input,
                attention_mask=current_mask,
                past_key_values=cache
            )[:, -1, :] / temperature

            all_inf = (logits == float('-inf')).all(dim=-1)
            if all_inf.any():
                logits[all_inf] = 0.0
            
            if top_k > 0:
                topk = torch.topk(logits, min(top_k, logits.size(-1)))
                mask = logits < topk.values[:, -1].unsqueeze(1)
                logits = logits.masked_fill(mask, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                next_token[finished] = eos_token_id

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            next_attention_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)
            next_attention_mask[finished] = 0
            attention_mask = torch.cat([attention_mask, next_attention_mask], dim=-1)

            if eos_token_id is not None:
                newly_finished = (next_token.view(-1) == eos_token_id)
                finished = finished | newly_finished
                if finished.all():
                    break

        return input_ids
