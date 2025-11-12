from transformers import PretrainedConfig


class CustomModelConfig(PretrainedConfig):
    model_type = "custom_model"

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 512,
        inter_dim: int = 2048,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        n_layers: int = 16,
        rope_theta: float = 10000.0,
        pad_token_id: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.inter_dim = inter_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.rope_theta = rope_theta
