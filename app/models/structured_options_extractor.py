from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class StructuredOptionsExtractor(BaseFeaturesExtractor):
    """Encode market history and the option surface as structured token sets.

    The observation keeps the candidate order aligned with the action mapping.
    This avoids forcing a plain MLP to rediscover temporal and contract structure
    from one very large flattened vector.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 752):
        super().__init__(observation_space, features_dim)

        market_shape = observation_space.spaces["market"].shape
        candidate_shape = observation_space.spaces["candidates"].shape
        market_features = int(market_shape[-1])
        candidate_features = int(candidate_shape[-1])
        candidate_count = int(candidate_shape[-2])

        self.market_norm = nn.LayerNorm(market_features)
        self.market_gru = nn.GRU(
            input_size=market_features,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.market_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        self.candidate_in = nn.Sequential(
            nn.Linear(candidate_features, 48),
            nn.GELU(),
            nn.LayerNorm(48),
        )
        self.candidate_pos = nn.Parameter(
            torch.zeros(1, candidate_count, 48)
        )
        nn.init.normal_(self.candidate_pos, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=48,
            nhead=4,
            dim_feedforward=96,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )
        self.candidate_token = nn.Sequential(
            nn.Linear(48, 4),
            nn.Tanh(),
        )
        self.candidate_pool = nn.Sequential(
            nn.Linear(96, 96),
            nn.GELU(),
            nn.LayerNorm(96),
        )

        self.portfolio_head = nn.Sequential(
            nn.Linear(observation_space.spaces["portfolio"].shape[0], 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )
        self.context_head = nn.Sequential(
            nn.Linear(observation_space.spaces["context"].shape[0], 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )
        self.position_head = nn.Sequential(
            nn.Linear(observation_space.spaces["position"].shape[0], 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )

        self._features_dim = 128 + (candidate_count * 4) + 96 + 32 + 32 + 32

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        market = self.market_norm(observations["market"])
        market_seq, _ = self.market_gru(market)
        market_repr = self.market_head(market_seq[:, -1, :])

        candidates = self.candidate_in(observations["candidates"])
        candidates = candidates + self.candidate_pos
        candidates = self.candidate_transformer(candidates)

        # Preserve one compact representation per candidate so the downstream
        # policy head keeps a stable relationship between candidate slots and
        # discrete open actions.
        candidate_tokens = self.candidate_token(candidates).flatten(start_dim=1)
        candidate_mean = candidates.mean(dim=1)
        candidate_max = candidates.amax(dim=1)
        candidate_pool = self.candidate_pool(
            torch.cat([candidate_mean, candidate_max], dim=1)
        )

        portfolio = self.portfolio_head(observations["portfolio"])
        context = self.context_head(observations["context"])
        position = self.position_head(observations["position"])

        return torch.cat(
            [
                market_repr,
                candidate_tokens,
                candidate_pool,
                portfolio,
                context,
                position,
            ],
            dim=1,
        )
