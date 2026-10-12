from __future__ import annotations

import torch
from torch import nn


class ContextAdapter(nn.Module):
    """
    Lightweight gated context-fusion adapter for a frozen NLLB encoder.

    The adapter projects a semantic context embedding into the NLLB
    encoder hidden dimension and adds it as a gated residual.

    The gate is initialized close to zero so that the untrained adapter
    behaves approximately like the original NLLB encoder.
    """

    def __init__(
        self,
        context_dim: int = 384,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Initialize the gate so that the initial context contribution
        # is very small. This preserves the pretrained NLLB behavior
        # before adapter training.
        gate_linear = self.gate[0]

        nn.init.xavier_uniform_(gate_linear.weight)

        nn.init.constant_(
            gate_linear.bias,
            -4.0,
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse semantic context into NLLB encoder hidden states.

        Args:
            encoder_hidden_states:
                Tensor of shape [batch, sequence_length, hidden_dim].

            context_embedding:
                Tensor of shape [batch, context_dim].

        Returns:
            Fused encoder hidden states with the same shape as the
            original encoder hidden states.
        """

        projected_context = self.context_projection(
            context_embedding
        )

        context_sequence = projected_context.unsqueeze(1).expand(
            -1,
            encoder_hidden_states.size(1),
            -1,
        )

        fusion_input = torch.cat(
            [
                encoder_hidden_states,
                context_sequence,
            ],
            dim=-1,
        )

        gate = self.gate(fusion_input)

        fused = (
            encoder_hidden_states
            + gate * context_sequence
        )

        return fused