"""Generic DQN trainer for any nn.Module that maps a batch of states to one Q-value per action.

Everything stays on the env's device: the env steps N games at once and the replay buffer is a
preallocated ring of tensors, so there are no per-game Python loops or host<->device copies.
"""

import copy
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential
from tqdm import tqdm

type State = Tensor  # (n_envs, state_dim) float


@dataclass
class Transition:
    """Result of one batched env step."""

    reward: Tensor  # (n_envs,) float
    next_state: State
    done: Tensor  # (n_envs,) bool


@dataclass
class Experience:
    """A batch of (state, action, transition) rows, as stored in and sampled from the buffer."""

    state: State
    action: Tensor  # (n_envs,) int64
    transition: Transition


class Env(Protocol):
    """A batch of games stepped together.

    step() must auto-reset finished games: rows where done is True hold the next game's start
    state in next_state (they are never bootstrapped from, so that is safe).
    """

    n_actions: int
    device: torch.device

    def reset(self) -> State:
        """Starts a fresh game in every row and returns the start states."""
        ...

    def step(self, action: Tensor) -> Transition:
        """Applies one action (n_envs,) int64 per game and returns the batched transition."""
        ...


class ReplayBuffer:
    """Ring buffer with proportional prioritized replay (Schaul et al. 2015). alpha=0 is uniform.

    Priority = (1 - reward_weight) * |TD error| + reward_weight * |reward - mean buffer reward|,
    so rare rewards in either direction are replayed more. reward_weight=0 is plain PER.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        alpha: float = 0.6,
        reward_weight: float = 0.0,
        prio_eps: float = 1e-6,
    ) -> None:
        """Data tensors are allocated lazily on the first add(), once their shapes are known."""
        self.capacity, self.device = capacity, device
        self.alpha, self.reward_weight, self.prio_eps = alpha, reward_weight, prio_eps
        self.reward_sum = torch.zeros(
            (), dtype=torch.float64, device=device
        )  # running sum for the mean
        self.data: list[Tensor] | None = None
        self.ptr = self.size = 0
        self.prio = torch.zeros(capacity, device=device)  # stored already raised to alpha
        self.max_prio = torch.ones(
            (), device=device
        )  # new experiences get this, so each is seen once

    @staticmethod
    def _unpack_fields(e: Experience) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Flattens an Experience into (state, action, reward, next_state, done)."""
        t = e.transition
        return e.state, e.action, t.reward, t.next_state, t.done

    def add(self, exp: Experience) -> None:
        """Writes a batch of experiences, overwriting the oldest rows once full, at max priority."""
        fields = self._unpack_fields(exp)
        if self.data is None:
            self.data = [
                torch.zeros((self.capacity, *f.shape[1:]), dtype=f.dtype, device=self.device)
                for f in fields
            ]
            self.rewards = self.data[2]
        n = fields[0].shape[0]
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.capacity
        self.reward_sum += (
            exp.transition.reward.sum() - self.rewards[idx].sum()
        )  # overwritten rows leave
        for buf, f in zip(self.data, fields, strict=True):
            buf[idx] = f
        self.prio[idx] = self.max_prio
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, k: int, beta: float) -> tuple[Experience, Tensor, Tensor]:
        """Returns (experiences, buffer indices, importance-sampling weights scaled to max 1)."""
        assert self.data is not None, "sample() before add()"
        if self.alpha == 0:
            idx = torch.randint(self.size, (k,), device=self.device)
            weights = torch.ones(k, device=self.device)
        else:
            p = self.prio[: self.size]
            idx = torch.multinomial(p, k, replacement=True)
            weights = (self.size * p[idx] / p.sum()) ** -beta
            weights = weights / weights.max()
        s, a, r, s2, d = (buf[idx] for buf in self.data)
        return Experience(s, a, Transition(r, s2, d)), idx, weights

    def update_priorities(self, idx: Tensor, td_error: Tensor) -> None:
        """Re-scores the rows at idx from their latest TD errors (and rewards if reward_weight)."""
        if self.alpha == 0:
            return
        score = td_error.abs()
        if self.reward_weight:
            mean = (self.reward_sum / self.size).float()
            score = (1 - self.reward_weight) * score + self.reward_weight * (
                self.rewards[idx] - mean
            ).abs()
        new = (score + self.prio_eps) ** self.alpha
        self.prio[idx] = new
        self.max_prio = torch.maximum(self.max_prio, new.max())


class QTrainer:
    """DQN with a target network, epsilon-greedy exploration and prioritized replay."""

    def __init__(
        self,
        model: Module,
        env: Env,
        *,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10_000,
        target_sync: int = 1_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_steps: int = 100_000,
        per_reward_weight: float = 0.0,
    ) -> None:
        """Builds the online and target networks, the optimizer and the replay buffer.

        per_alpha=0 disables prioritized replay. beta anneals to 1 over per_beta_steps.
        per_reward_weight in [0, 1] blends reward extremeness into the priority (see ReplayBuffer).
        """
        self.env = env
        self.device = env.device
        self.q = model.to(self.device)
        self.target = copy.deepcopy(self.q).eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(
            buffer_size, self.device, alpha=per_alpha, reward_weight=per_reward_weight
        )
        self.gamma, self.batch_size, self.target_sync = gamma, batch_size, target_sync
        self.eps_start, self.eps_end, self.eps_decay_steps = (
            eps_start,
            eps_end,
            eps_decay_steps,
        )
        self.beta_start, self.beta_steps = per_beta_start, per_beta_steps
        self.steps = 0

    def compute_epsilon(self) -> float:
        """Exploration rate, decayed linearly from eps_start to eps_end over eps_decay_steps."""
        frac = min(self.steps / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def compute_beta(self) -> float:
        """Importance-sampling exponent, annealed linearly from per_beta_start to 1."""
        frac = min(self.steps / self.beta_steps, 1.0)
        return self.beta_start + frac * (1.0 - self.beta_start)

    @torch.no_grad()
    def act(self, state: State, eps: float = 0.0) -> Tensor:
        """Picks one action per game: greedy, except a random one with probability eps."""
        greedy = self.q(state).argmax(1)
        if eps <= 0:
            return greedy
        explore = torch.rand(greedy.shape, device=self.device) < eps
        return torch.where(explore, torch.randint_like(greedy, self.env.n_actions), greedy)

    def learn(self) -> None:
        """Takes one gradient step on a prioritized batch and refreshes those rows' priorities."""
        if self.buffer.size < self.batch_size:
            return
        e, idx, weights = self.buffer.sample(self.batch_size, self.compute_beta())
        t = e.transition
        q = self.q(e.state).gather(1, e.action[:, None]).squeeze(1)
        with torch.no_grad():
            target = t.reward + self.gamma * ~t.done * self.target(t.next_state).max(1).values
        loss = (weights * F.smooth_l1_loss(q, target, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.buffer.update_priorities(idx, (target - q).detach())

    def train(self, steps: int, log_every: int = 0, progress_bar: bool = False) -> None:
        """Runs `steps` batched env steps (each one advances every game in the batch).

        Prints the mean return of episodes finished in the last log_every steps; 0 disables logging.
        progress_bar shows a tqdm progress bar, with the log lines printed above it.
        """
        state = self.env.reset()
        ep_return = torch.zeros(state.shape[0], device=self.device)
        done_sum = torch.zeros((), device=self.device)
        done_n = torch.zeros((), device=self.device)
        for step in tqdm(range(steps), unit="step", disable=not progress_bar):
            action = self.act(state, self.compute_epsilon())
            t = self.env.step(action)
            self.buffer.add(Experience(state, action, t))
            self.learn()
            self.steps += 1
            if self.steps % self.target_sync == 0:
                self.target.load_state_dict(self.q.state_dict())
            ep_return += t.reward
            done_sum += (ep_return * t.done).sum()
            done_n += t.done.sum()
            ep_return *= ~t.done
            state = t.next_state
            if log_every and (step + 1) % log_every == 0:
                avg = (done_sum / done_n.clamp(min=1)).item()  # the only host sync
                tqdm.write(
                    f"step {step + 1}  avg return {avg:.4f}  eps {self.compute_epsilon():.3f}"
                )
                done_sum.zero_()
                done_n.zero_()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean greedy return over each game's first episode (one episode per game in the batch)."""
        state = self.env.reset()
        total = torch.zeros(state.shape[0], device=self.device)
        finished = torch.zeros(state.shape[0], dtype=torch.bool, device=self.device)
        while not finished.all():
            t = self.env.step(self.act(state))
            total += t.reward * ~finished
            finished |= t.done
            state = t.next_state
        return total.mean().item()


if __name__ == "__main__":
    # Self-check: one-step game, reward 1 for picking the action equal to the state bit.
    class MatchEnv:
        """One-step game: reward 1 for picking the action equal to the state bit."""

        n_actions = 2

        def __init__(self, n: int, device: torch.device) -> None:
            """Sets up n games on device."""
            self.n, self.device = n, device

        def reset(self) -> State:
            """Draws a fresh random bit per game and returns it as the state."""
            self.bit = torch.randint(2, (self.n,), device=self.device)
            return self.bit[:, None].float()

        def step(self, action: Tensor) -> Transition:
            """Rewards matching actions; every game ends and auto-resets."""
            reward = (action == self.bit).float()
            done = torch.ones(self.n, dtype=torch.bool, device=self.device)
            return Transition(reward, self.reset(), done)  # every game ends, so all auto-reset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PER: a row with a huge TD error dominates sampling and gets the smallest IS weight.
    buf = ReplayBuffer(100, device)
    env = MatchEnv(100, device)
    s = env.reset()
    buf.add(
        Experience(
            s,
            torch.zeros(100, dtype=torch.int64, device=device),
            env.step(s[:, 0].long()),
        )
    )
    buf.update_priorities(torch.arange(100, device=device), torch.full((100,), 0.01, device=device))
    buf.update_priorities(torch.tensor([7], device=device), torch.tensor([1e4], device=device))
    _, idx, w = buf.sample(
        2_000, beta=1.0
    )  # expected share of row 7: 251 / (251 + 99 * 0.063) ~ 0.976
    assert (idx == 7).float().mean() > 0.95 and w[idx == 7].max() < w[idx != 7].min()

    # Reward extremeness: with TD errors all zero, the one rare reward is replayed far more per row.
    buf = ReplayBuffer(100, device, reward_weight=1.0)
    reward = torch.zeros(100, device=device)
    reward[7] = 1.0
    buf.add(
        Experience(
            s,
            torch.zeros(100, dtype=torch.int64, device=device),
            Transition(reward, s, torch.ones(100, dtype=torch.bool, device=device)),
        )
    )
    buf.update_priorities(torch.arange(100, device=device), torch.zeros(100, device=device))
    _, idx, _ = buf.sample(5_000, beta=1.0)  # row 7 ~16x the per-row rate of the others
    assert (idx == 7).sum() > 10 * (idx != 7).sum() / 99

    for alpha in (0.6, 0.0):
        net = Sequential(Linear(1, 16), ReLU(), Linear(16, 2))
        trainer = QTrainer(
            net,
            MatchEnv(1024, device),
            eps_decay_steps=200,
            target_sync=20,
            per_alpha=alpha,
            per_beta_steps=400,
        )
        trainer.train(400)
        score = trainer.evaluate()
        assert score == 1.0, (alpha, score)
    print("ok", device)
