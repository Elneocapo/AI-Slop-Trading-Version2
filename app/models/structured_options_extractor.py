from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class StructuredOptionsExtractor(BaseFeaturesExtractor):
    """Fast structured encoder for hourly NVDA options decisions.

    Design goals:
    - preserve the 60-bar temporal sequence instead of flattening it;
    - encode all 108 option candidates with shared weights;
    - condition every candidate on the current market representation;
    - keep candidate-slot order aligned with the Discrete action mapping;
    - remain cheap enough for CPU PPO training.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 696):
        super().__init__(observation_space, features_dim)

        market_shape = observation_space.spaces["market"].shape
        candidate_shape = observation_space.spaces["candidates"].shape
        market_features = int(market_shape[-1])
        candidate_features = int(candidate_shape[-1])
        candidate_count = int(candidate_shape[-2])

        # Temporal encoder: convolution is substantially cheaper than recurrent
        # attention on CPU while still learning local and multi-hour patterns.
        self.market_norm = nn.LayerNorm(market_features)
        self.market_conv = nn.Sequential(
            nn.Conv1d(market_features, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.market_pool = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        # Shared contract encoder. Every candidate is processed by the same
        # weights, preventing 108 independent parameter sets from memorizing
        # particular strikes/DTE slots.
        self.candidate_base = nn.Sequential(
            nn.Linear(candidate_features, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.market_to_candidate = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
        )
        self.candidate_fusion = nn.Sequential(
            nn.Linear(32 + 32, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )
        self.candidate_token = nn.Sequential(
            nn.Linear(32, 4),
            nn.Tanh(),
        )
        self.candidate_pool = nn.Sequential(
            nn.Linear(32 * 2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

        self.portfolio_head = nn.Sequential(
            nn.Linear(observation_space.spaces["portfolio"].shape[0], 24),
            nn.GELU(),
            nn.LayerNorm(24),
        )
        self.context_head = nn.Sequential(
            nn.Linear(observation_space.spaces["context"].shape[0], 24),
            nn.GELU(),
            nn.LayerNorm(24),
        )
        self.position_head = nn.Sequential(
            nn.Linear(observation_space.spaces["position"].shape[0], 24),
            nn.GELU(),
            nn.LayerNorm(24),
        )

        self._features_dim = 128 + (candidate_count * 4) + 64 + 24 + 24 + 24

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        market = self.market_norm(observations["market"])
        # [batch, time, features] -> [batch, features, time]
        market = market.transpose(1, 2)
        market_seq = self.market_conv(market)
        market_mean = market_seq.mean(dim=2)
        market_max = market_seq.amax(dim=2)
        market_repr = self.market_pool(
            torch.cat([market_mean, market_max], dim=1)
        )

        candidates = self.candidate_base(observations["candidates"])
        market_context = self.market_to_candidate(market_repr).unsqueeze(1)
        market_context = market_context.expand(-1, candidates.shape[1], -1)
        fused = self.candidate_fusion(
            torch.cat([candidates, market_context], dim=-1)
        )

        # Four compact values per candidate become the policy's candidate-aware
        # representation while retaining the exact 108-slot action alignment.
        candidate_tokens = self.candidate_token(fused).flatten(start_dim=1)
        candidate_mean = fused.mean(dim=1)
        candidate_max = fused.amax(dim=1)
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
