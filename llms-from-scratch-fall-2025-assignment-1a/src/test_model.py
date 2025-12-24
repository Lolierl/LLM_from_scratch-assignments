from einops import rearrange
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Int, Bool
from .model import *

def run_embedding(
    vocab_size: int,
    d_model: int,
    weight: Float[Tensor, "vocab_size d_model"],
    token_ids: Int[Tensor, "batch_size seq_len"],
) -> Float[Tensor, "batch_size d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weight (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "batch_size seq_len"]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "batch_size d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embed = Embedding(vocab_size, d_model)
    # Note that you should name the weight parameter of the Embedding module as "weight".
    # Otherwise load_state_dict will fail.
    embed.load_state_dict({"weight": weight})
    return embed(token_ids)


def run_linear(
    d_in: int,
    d_out: int,
    weight: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "batch_size d_in"],
) -> Float[Tensor, "batch_size d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        d_in (int): The size of the input dimension
        d_out (int): The size of the output dimension
        weight (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "batch_size d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "batch_size d_out"]: The transformed output of your linear module.
    """

    linear = Linear(d_in, d_out)
    linear.load_state_dict({"weight": weight}) 
    out_features = linear(in_features)
    return out_features


def run_rmsnorm(
    d_model: int,
    eps: float,
    weight: Float[Tensor, "d_model"],
    in_features: Float[Tensor, "batch_size d_model"],
) -> Float[Tensor, "batch_size d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weight (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "batch_size d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor, "batch_size d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    
    rmsnorm = RMSNorm(d_model, eps)
    rmsnorm.load_state_dict({"weight": weight})
    return rmsnorm(in_features)

def run_swiglu(
    d_model: int,
    d_ff: int,
    gate_proj_weight: Float[Tensor, "d_ff d_model"],
    up_proj_weight: Float[Tensor, "d_ff d_model"],
    down_proj_weight: Float[Tensor, "d_model d_ff"],
    in_features: Float[Tensor, "batch_size d_model"],
) -> Float[Tensor, "batch_size d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        gate_proj_weight (Float[Tensor, "d_ff d_model"]): Stored weights for gate_proj
        up_proj_weight (Float[Tensor, "d_ff d_model"]): Stored weights for up_proj
        down_proj_weight (Float[Tensor, "d_model d_ff"]): Stored weights for down_proj
        in_features (Float[Tensor, "batch_size d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "batch_size d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    swiglu = MLP(d_model, d_ff)

    # Note that you should include three linear modules in your MLP,
    # named "gate_proj", "up_proj", and "down_proj", each containing its
    # own parameter called "weight". Otherwise load_state_dict will fail.
    swiglu.load_state_dict(
        {
            "gate_proj.weight": gate_proj_weight,
            "up_proj.weight":   up_proj_weight,
            "down_proj.weight": down_proj_weight,
        }
    )
    return swiglu(in_features)


def run_rope(
    head_dim: int,
    theta: float,
    token_positions: Int[Tensor, "batch_size sequence_length"],
) -> Tuple[Float[Tensor, "batch_size sequence_length head_dim/2"],
            Float[Tensor, "batch_size sequence_length head_dim/2"]]:
    """
    Run RoPE for a given input tensor.

    Args:
        head_dim (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        token_positions (Int[Tensor, "batch_size sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Tuple[Float[Tensor, "batch_size sequence_length head_dim/2"],
              Float[Tensor, "batch_size sequence_length head_dim/2"]]: 
              Tuple of two tensors containing the cosine and sine values 
              of the RoPE embeddings.
    """
    rope = RotaryEmbedding(head_dim, theta)
    return rope(token_positions)


def run_multihead_self_attention_with_rope(
    d_model: int,
    d_head: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    layer_idx: int,
    in_features: Float[torch.Tensor, "batch seq_len d_model"],
    attention_mask: Bool[torch.Tensor, "batch seq_len seq_len"],
    position_embeddings: Tuple[
        Float[torch.Tensor, "batch seq_len d_head/2"],
        Float[torch.Tensor, "batch seq_len d_head/2"]
    ],
    q_proj_weight: Float[torch.Tensor, "num_attention_heads*d_head d_model"],
    k_proj_weight: Float[torch.Tensor, "num_key_value_heads*d_head d_model"],
    v_proj_weight: Float[torch.Tensor, "num_key_value_heads*d_head d_model"],
    o_proj_weight: Float[torch.Tensor, "d_model num_attention_heads*d_head"],
    q_norm_weight: Float[torch.Tensor, "d_head"],
    k_norm_weight: Float[torch.Tensor, "d_head"],
) -> Float[torch.Tensor, "batch seq_len d_model"]:
    attention_module = MHA(
        embed_dim=d_model,
        head_dim=d_head,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        layer_idx=layer_idx
    )
    
    state_dict = {
        "q_proj.weight": q_proj_weight,
        "k_proj.weight": k_proj_weight,
        "v_proj.weight": v_proj_weight,
        "o_proj.weight": o_proj_weight,
        "q_norm.weight": q_norm_weight,
        "k_norm.weight": k_norm_weight,
    }
    attention_module.load_state_dict(state_dict, strict=False)
    output = attention_module(
        x=in_features,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings
    )

    return output

    
def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0][f"layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weight=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        output
    )


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0][f"token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weight=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(
        output
    )


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        gate_proj_weight=w1_weight,
        up_proj_weight=w3_weight,
        down_proj_weight=w2_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_singlehead_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    q, k, v = (x.unsqueeze(-3) for x in (q, k, v))
    
    # q.shape:    (bsz, num_head_q,  len_q,  d_head)
    # k.shape:    (bsz, num_head_kv, len_kv, d_head)
    # v.shape:    (bsz, num_head_kv, len_kv, d_head)
    # mask.shape: (bsz, len_q, len_kv)
    # In this test, all num_head = 1.

    actual_output = scaled_dot_product_attention(q=q, k=k, v=v, attention_mask=mask)
    
    # expected output shape: (bsz, num_head_q, len_q, d_head)
    # where num_head_q = 1 in this test
    
    actual_output = actual_output.squeeze(-3)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = mask[:2]

    # q.shape:    (bsz, num_head_q,  len_q,  d_head)
    # k.shape:    (bsz, num_head_kv, len_kv, d_head)
    # v.shape:    (bsz, num_head_kv, len_kv, d_head)
    # mask.shape: (bsz, len_q, len_kv)
    # In this test, num_head_q = num_head_kv > 1.

    actual_output = scaled_dot_product_attention(q=q, k=k, v=v, attention_mask=mask)

    # expected output shape: (bsz, num_head_q, len_q, d_head)

    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_grouped_query_attention(numpy_snapshot, q, k, v, mask):
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = mask[:2]
    k = k[:, 0, :, :].unsqueeze(1)
    v = v[:, 0, :, :].unsqueeze(1)

    # q.shape:    (bsz, num_head_q,  len_q,  d_head)
    # k.shape:    (bsz, num_head_kv, len_kv, d_head)
    # v.shape:    (bsz, num_head_kv, len_kv, d_head)
    # mask.shape: (bsz, len_q,  len_kv)
    # In this test, num_head_q != num_head_kv.

    actual_output = scaled_dot_product_attention(q=q, k=k, v=v, attention_mask=mask)

    # expected output shape: (bsz, num_head_q, len_q, d_head)

    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention_with_rope(
    numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    d_head = d_model // n_heads
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    pos_ids = pos_ids.expand(in_embeddings.shape[:-2] + (pos_ids.shape[-1],))

    seq = in_embeddings.shape[-2]
    mask = torch.tril(torch.ones(seq, seq)).bool()
    mask = mask.view(1, seq, seq)
    mask = mask.expand(in_embeddings.shape[0], -1, -1)

    position_embeddings = run_rope(head_dim=d_head, theta=theta, token_positions=pos_ids)

    # generate q_norm_weight and k_norm_weight
    q_norm_weight = torch.randn(d_head)
    k_norm_weight = torch.randn(d_head)

    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        d_head=d_head,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        layer_idx=0,
        in_features=in_embeddings,
        attention_mask=mask,
        position_embeddings=position_embeddings,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]

    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weight=reference_weights, in_features=in_embeddings)

    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rope(numpy_snapshot, d_model, theta, pos_ids):
    output = run_rope(
        d_model, theta=theta, token_positions=pos_ids.unsqueeze(0)
    )
    output = {
        "cos_tri": output[0].squeeze(0),
        "sin_tri": output[1].squeeze(0),
    }
    numpy_snapshot.assert_match(output, atol=1e-6)
