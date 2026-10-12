"""Trains MarioModel with QTrainer on the capture-the-flag problem (see problem.md)."""

import torch
from torch import Tensor

from mario.model.model import INPUT_SIZE, OUTPUT_SIZE, MarioModel
from models.qtrainer import QTrainer, State, Transition

DAYS: int = 3


def build_state(dist: Tensor, draw: Tensor, days_left: Tensor) -> State:
    """Stacks per-game fields into the (n, INPUT_SIZE) model input.

    Columns: distance left, its square root, its square, today's draw, days left / DAYS.
    Distance left is always > 0 in a state, since games that reach the princess end.
    """
    state = torch.stack([dist, dist.sqrt(), dist**2, draw, days_left / DAYS], 1)
    assert state.shape[1] == INPUT_SIZE, "update input_size in mario/model/cfg.json"
    return state


class MarioEnv:
    """n Mario games stepped together. Action 0 = normal move, 1 = double move.

    State comes from build_state (distance left and features of it, today's draw, days left).
    Once the double is used there are no decisions left, so the remaining days are played out in
    that same step, which means every state the agent sees still has the double available.
    Reward is 1 when Mario reaches the princess, otherwise 0. With gamma=1, Q is the probability
    of success.

    With exploring_starts (for training only; evaluate() needs the real start), new games start
    from these states instead of (distance 1, day 1):
    - grid_share of them take the next state of a shuffled grid of every state at resolution
      step. The grid is swept in order and reshuffled once it has all been used.
    - the rest start from a random state: uniform day, distance and draw.
    Day 1 always has distance 1 in both: it is the only day-1 state a real game can have.
    """

    n_actions = OUTPUT_SIZE

    def __init__(
        self,
        n: int,
        device: torch.device,
        exploring_starts: bool = False,
        grid_share: float = 0.5,
        step: float = 0.01,
    ) -> None:
        """Sets up n games on device. Call reset() before step()."""
        self.n, self.device = n, device
        self.exploring_starts, self.grid_share = exploring_starts, grid_share
        self.grid = self._build_grid(step) if exploring_starts else None
        if self.grid is not None:
            # Upcoming grid indices: enough shuffled sweeps back to back to serve all n rows.
            self.sweeps = n // len(self.grid) + 2
            self.order = self._shuffle_sweeps()
            self.cursor = torch.zeros((), dtype=torch.long, device=device)  # next index in order

    def _build_grid(self, step: float) -> Tensor:
        """Every state at resolution step as (dist, draw, days_left) rows.

        Draws run 0 to 1 - step (a draw is never 1) and distances step to 1. Day 1 only has
        distance 1, so there are 1/step day-1 states and (1/step)^2 for each later day.
        """
        k, device = round(1 / step), self.device
        draws = torch.arange(k, device=device) * step
        dists = torch.arange(1, k + 1, device=device) * step
        ones = torch.ones(1, device=device)
        day_1 = torch.cartesian_prod(ones, draws, ones * DAYS)
        later_days = torch.cartesian_prod(dists, draws, torch.arange(1.0, DAYS, device=device))
        return torch.cat([day_1, later_days])

    def _shuffle_sweeps(self) -> Tensor:
        """self.sweeps independent shuffles of the grid indices, back to back."""
        assert self.grid is not None
        keys = torch.rand(self.sweeps, len(self.grid), device=self.device)
        return keys.argsort(1).flatten()

    def _take_grid_states(self, take: Tensor) -> Tensor:
        """Next grid states in shuffled order for the rows where take is True (others get junk).

        Runs entirely on the device, without waiting on the GPU. Each sweep visits every grid
        state once; finished sweeps are dropped and fresh shuffles appended behind the rest.
        """
        assert self.grid is not None
        n_grid = len(self.grid)
        pos = self.cursor + take.cumsum(0) - 1  # consecutive positions for the taking rows
        idx = self.order[pos.clamp(0, len(self.order) - 1)]
        self.cursor = self.cursor + take.sum()
        done_sweeps = self.cursor // n_grid
        shift = done_sweeps * n_grid + torch.arange(len(self.order), device=self.device)
        self.order = torch.cat([self.order, self._shuffle_sweeps()])[shift]
        self.cursor = self.cursor - done_sweeps * n_grid
        return self.grid[idx]

    def _observe(self) -> State:
        """Current state of every game."""
        return build_state(self.dist, self.draw, self.days_left)

    def _start_games(self, rows: Tensor) -> None:
        """Puts a fresh game in the rows where the bool mask rows is True."""
        n, device = self.n, self.device
        days_left = torch.full((n,), float(DAYS), device=device)
        if self.exploring_starts:
            days_left = torch.randint(1, DAYS + 1, (n,), device=device).float()
        dist = torch.where(days_left == DAYS, 1.0, 1 - torch.rand(n, device=device))  # (0, 1]
        start = torch.stack([dist, torch.rand(n, device=device), days_left], 1)
        if self.grid is not None:
            use_grid = rows & (torch.rand(n, device=device) < self.grid_share)
            start = torch.where(use_grid[:, None], self._take_grid_states(use_grid), start)
        self.dist = torch.where(rows, start[:, 0], self.dist)
        self.draw = torch.where(rows, start[:, 1], self.draw)
        self.days_left = torch.where(rows, start[:, 2], self.days_left)

    def reset(self) -> State:
        """Starts a fresh game in every row."""
        self.dist = torch.empty(self.n, device=self.device)
        self.draw = torch.empty(self.n, device=self.device)
        self.days_left = torch.empty(self.n, device=self.device)
        self._start_games(torch.ones(self.n, dtype=torch.bool, device=self.device))
        return self._observe()

    def step(self, action: Tensor) -> Transition:
        """Moves every game one day forward, finishing it if Mario doubles.

        Finished games are reset automatically.
        """
        double = action == 1
        self.dist -= self.draw * (1 + double)
        self.days_left -= 1
        future = torch.rand(self.n, DAYS - 1, device=self.device)
        future *= torch.arange(DAYS - 1, device=self.device) < self.days_left[:, None]
        self.dist -= future.sum(1) * double
        reward = (self.dist <= 0).float()
        done = (self.dist <= 0) | (self.days_left == 0) | double
        self.draw = torch.rand(self.n, device=self.device)
        self._start_games(done)
        return Transition(reward, self._observe(), done)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = QTrainer(
        MarioModel(),
        MarioEnv(100_000, device, exploring_starts=True),
        gamma=1.0,
        eps_decay_steps=8_000,
        target_sync=250,
        per_beta_steps=3_000,
    )
    trainer.train(12_000, log_every=500, progress_bar=True)
    trainer.env = MarioEnv(10**7, device)
    p = trainer.evaluate()
    # print(f"learned P(success) {p:.4f}  exact {EXACT:.4f}")
    # assert abs(p - EXACT) < 0.01, p
    print(f"learned P(success) = {p:.4f}")
