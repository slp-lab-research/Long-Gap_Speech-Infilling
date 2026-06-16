from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    num_layers: int
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def lora_config_from_args(args) -> LoRAConfig:
    targets = tuple(part.strip() for part in str(args.lora_target_modules).split(",") if part.strip())
    return LoRAConfig(
        num_layers=int(args.lora_layers),
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=targets,
    )


def apply_lora_to_last_llm_layers(
    model,
    num_layers: int,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str],
) -> int:
    layers = list(getattr(model.llm.model, "layers", []))
    if not layers or num_layers <= 0 or rank <= 0:
        return 0
    target_names = set(target_modules)
    injected = 0
    for layer in layers[-num_layers:]:
        injected += _inject_lora_modules(layer, target_names, rank, alpha, dropout)
    return injected


def _inject_lora_modules(module: nn.Module, target_names: set[str], rank: int, alpha: float, dropout: float) -> int:
    injected = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            injected += 1
        else:
            injected += _inject_lora_modules(child, target_names, rank, alpha, dropout)
    return injected


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
