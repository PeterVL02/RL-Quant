"""Q-network for the Mario capture-the-flag problem, sized from cfg.json."""

import json
from pathlib import Path
from typing import TypedDict

from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential, Tanh


class ModelConfig(TypedDict):
    """Network sizes read from cfg.json."""

    input_size: int
    output_size: int


config: ModelConfig = json.loads((Path(__file__).parent / "cfg.json").read_text())

INPUT_SIZE = config["input_size"]
OUTPUT_SIZE = config["output_size"]


class MarioModel(Module):
    """MLP mapping a batch of states to one Q-value per action."""

    def __init__(self) -> None:
        """Builds the network with sizes from cfg.json."""
        super().__init__()
        self.model = Sequential(
            Linear(INPUT_SIZE, 128),
            ReLU(),
            Linear(128, 128),
            Tanh(),
            Linear(128, 64),
            ReLU(),
            Linear(64, OUTPUT_SIZE),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Maps states (batch, INPUT_SIZE) to Q-values (batch, OUTPUT_SIZE)."""
        return self.model(x)
