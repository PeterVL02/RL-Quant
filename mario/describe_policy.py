"Train a model and attempt to quantitatively explain the actions"

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from torch import Tensor
from torch.nn import Module

from mario.model.model import MarioModel
from mario.train import DAYS, MarioEnv, build_state
from models.qtrainer import QTrainer

PLOT_PATH = Path(__file__).with_name("policy.png")
OPTIMAL_DAY1_CUTOFF = (math.sqrt(3) - 1) / (2 * math.sqrt(3) - 1)  # ~0.297
DOUBLE_COLOR = "#3b75af"


def train(n_games: int = 100_000, n_steps: int = 12_000) -> Module:
    """Trains a model and returns it. Also prints the learned probability of success."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = QTrainer(
        MarioModel(),
        MarioEnv(n_games, device, exploring_starts=True),
        gamma=1.0,
        eps_decay_steps=8_000,
        target_sync=250,
        per_beta_steps=3_000,
    )
    trainer.train(n_steps, progress_bar=True)
    trainer.env = MarioEnv(10**7, device)
    p = trainer.evaluate()
    print(f"learned P(success) = {p:.4f}")
    return trainer.q


def evaluate(model: Module, n_games: int = 10**7) -> float:
    """Evaluates a model and returns the learned probability of success."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = QTrainer(
        model,
        MarioEnv(n_games, device),
        gamma=1.0,
        eps_decay_steps=8_000,
        target_sync=250,
        per_beta_steps=3_000,
    )
    p = trainer.evaluate()
    print(f"learned P(success) = {p:.4f}")
    return p


@torch.no_grad()
def compute_double_probs(
    model: Module, step: float = 0.01, temperature: float = 0.01
) -> tuple[Tensor, Tensor, Tensor]:
    """Runs the model on every (day, distance left, draw) point of a grid.

    Returns (dists, draws, p_double): dists (n,) from step to 1, draws (n + 1,) from 0 to 1, and
    p_double (DAYS, n, n + 1) = softmax(Q / temperature)[double]. The greedy policy doubles where
    p_double > 0.5. The Q-values are success probabilities that differ by a few hundredths at
    most, so temperature 1 would leave p_double near 0.5 everywhere; a small temperature spreads
    it out. Day 1 always starts at distance 1, so on day 1 only the last row of dists can
    actually happen.
    """
    device = next(model.parameters()).device
    n = round(1 / step)
    dists = torch.linspace(step, 1, n, device=device)
    draws = torch.linspace(0, 1, n + 1, device=device)
    days_left = torch.arange(DAYS, 0, -1, device=device, dtype=torch.float)  # day 1, 2, 3
    grid = torch.cartesian_prod(days_left, dists, draws)  # (days_left, dist, draw) per row
    q = model(build_state(grid[:, 1], grid[:, 2], grid[:, 0]))
    p_double = torch.softmax(q / temperature, 1)[:, 1].view(DAYS, n, n + 1)
    return dists.cpu(), draws.cpu(), p_double.cpu()


def find_double_ranges(draws: Tensor, doubles: Tensor, dist: float) -> str:
    """Draw ranges where the model doubles, e.g. "0.38-0.60, 0.70-0.99", or "never".

    Only draws < dist are considered: from dist upwards Mario arrives without doubling, so the
    action there does not matter.
    """
    on = (doubles & (draws < dist)).int()
    edges = torch.diff(
        on, prepend=torch.zeros(1, dtype=on.dtype), append=torch.zeros(1, dtype=on.dtype)
    )
    starts = (edges == 1).nonzero().flatten()
    ends = (edges == -1).nonzero().flatten() - 1
    ranges = [f"{draws[a]:.2f}-{draws[b]:.2f}" for a, b in zip(starts, ends, strict=True)]
    return ", ".join(ranges) or "never"


def find_optimal_range(day: int, dist: float) -> str:
    """Draws where the optimal policy doubles (derived by hand), as text for the table."""
    if day == 1:
        return f"{OPTIMAL_DAY1_CUTOFF:.3f}-{dist:.2f}"
    if day == 2:
        return f"{dist / 3:.3f}-{dist:.2f}"
    # On the last day doubling never hurts; it is only needed from dist / 2 (below that both fail).
    return f"{dist / 2:.3f}-{dist:.2f}"


def describe_policy(
    model: Module, step: float = 0.01, temperature: float = 0.01, path: Path = PLOT_PATH
) -> None:
    """Prints where the model doubles next to where the optimal policy does, and saves a plot.

    Day 1 always has distance 1, so its panel is a line of softmax(Q / temperature)[double]
    against the draw. In the day 2 and 3 panels each pixel is the double colour with that value as
    its opacity: solid where doubling clearly wins, white where the normal move clearly wins, faint
    where the model is undecided. The black line is where the greedy policy switches (0.5).
    """
    dists, draws, p_double = compute_double_probs(model, step, temperature)
    doubles = p_double > 0.5
    print("Draws at which Mario doubles (only draws < dist left; above that he arrives anyway)")
    print(f"{'day':>3}  {'dist left':>9}  {'optimal':>11}  learned")
    for day in range(1, DAYS + 1):
        for dist in [1.0] if day == 1 else [k / 10 for k in range(1, 11)]:
            i = round(dist / step) - 1
            learned = find_double_ranges(draws, doubles[day - 1, i], dist)
            print(f"{day:>3}  {dist:>9.1f}  {find_optimal_range(day, dist):>11}  {learned}")

    fig, axes = plt.subplots(1, DAYS, figsize=(5 * DAYS, 4.5), layout="constrained")
    p_label = f"softmax(Q / {temperature:g})[double]"
    ax = axes[0]
    ax.plot(draws, p_double[0, -1], color=DOUBLE_COLOR, label="model (distance 1)")
    ax.axhline(0.5, color="k", lw=0.8, label="greedy policy switches")
    ax.axvline(OPTIMAL_DAY1_CUTOFF, color="r", ls="--", label="optimal: double above draw = 0.297")
    ax.set(title="Day 1", xlabel="today's draw", ylabel=p_label, xlim=(0, 1), ylim=(-0.02, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    r = dists.numpy()
    rgba = torch.empty(*p_double.shape, 4)
    rgba[..., :3] = torch.tensor(to_rgb(DOUBLE_COLOR))
    rgba[..., 3] = p_double
    extent = (-step / 2, 1 + step / 2, step / 2, 1 + step / 2)  # pixel edges around the centres
    for day, ax in enumerate(axes[1:], start=2):
        ax.imshow(rgba[day - 1], origin="lower", extent=extent, aspect="auto", interpolation="none")
        ax.contour(draws, dists, p_double[day - 1], levels=[0.5], colors="k", linewidths=0.8)
        ax.plot(r, r, ":k", label="draw = dist (enough without doubling)")
        if day == 2:
            ax.plot(r / 3, r, "--r", label="optimal: double above draw = dist / 3")
        else:
            ax.plot(r / 2, r, "--r", label="must double above draw = dist / 2")
        ax.set(title=f"Day {day}", xlabel="today's draw", xlim=(0, 1), ylim=(0, 1 + step / 2))
        ax.legend(loc="lower right", fontsize=8)
    axes[1].set_ylabel("distance left")
    cmap = LinearSegmentedColormap.from_list("p_double", ["white", DOUBLE_COLOR])
    fig.colorbar(ScalarMappable(cmap=cmap), ax=axes[1:], label=p_label)
    fig.savefig(path, dpi=120)
    print(f"plot saved to {path}")


if __name__ == "__main__":
    describe_policy(train(n_games=200_000, n_steps=16_000), step=0.002)
