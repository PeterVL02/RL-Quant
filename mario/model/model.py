"""Q-network for the Mario capture-the-flag problem, sized from cfg.json."""
import json
from typing import TypedDict

from torch import Tensor
from torch.nn import Module, Linear, ReLU, Sequential


class ModelConfig(TypedDict):
    input_size: int
    output_size: int


config: ModelConfig = json.load(open("mario/model/cfg.json"))

INPUT_SIZE = config["input_size"]
OUTPUT_SIZE = config["output_size"]


class MarioModel(Module):
    """MLP mapping a batch of states to one Q-value per action."""

    def __init__(self) -> None:
        """Builds the network with sizes from cfg.json."""
        super(MarioModel, self).__init__()
        self.model = Sequential(
            Linear(INPUT_SIZE, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, OUTPUT_SIZE)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Maps states (batch, INPUT_SIZE) to Q-values (batch, OUTPUT_SIZE)."""
        return self.model(x)
