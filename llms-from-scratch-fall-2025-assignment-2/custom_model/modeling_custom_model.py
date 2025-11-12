from __future__ import annotations

from typing import Optional, Any

import torch
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_custom_model import CustomModelConfig
from .model import ModelArgs, Transformer


class CustomModelForCausalLM(PreTrainedModel):
    config_class = CustomModelConfig
    base_model_prefix = "transformer"

    def __init__(self, config: CustomModelConfig):
        super().__init__(config)
        self.transformer = Transformer(ModelArgs(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            inter_dim=config.inter_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            n_layers=config.n_layers,
            rope_theta=config.rope_theta,
        ))
        # Tie embeddings and lm_head weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.transformer.embed = value

    def get_output_embeddings(self) -> nn.Module:
        return self.transformer.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.transformer.lm_head = new_embeddings

    def tie_weights(self):
        self.transformer.lm_head.weight = self.transformer.embed.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        return_dict: Optional[bool] = True,
    ) -> CausalLMOutputWithPast:
        logits = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        return self.transformer.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
