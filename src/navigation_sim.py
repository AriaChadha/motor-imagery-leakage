"""
navigation_sim.py — headless SimPy navigation model

Separate from simulation_06.py, which keeps the PyQt5 demo GUI. This module is
what the pipeline imports: no Qt dependency, no hardcoded accuracy table.

The key difference from the earlier version: measured decoding accuracy is the
probability that a decoded command matches the intended one. Command noise is
1 - accuracy. Previously the noise rate was fixed at 10% and the accuracy
argument was accepted but never used, so every group navigated identically and
the navigation results carried no information about decoding performance.

Metrics recorded per trial:
    reached_goal   did the chair reach the goal within the step budget
    n_steps        steps taken (rises when commands are wrong)
    n_collisions   attempted moves into an obstacle or wall
    n_overrides    commands replaced by the safety layer (blocking rate)
    n_correct      decoded commands matching intent
    latencies      per-command processing delay in seconds
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import simpy

ROWS, COLS = 10, 10
N_OBS      = 12
START      = (0, 0)
GOAL       = (9, 9)
LAT_MIN    = 0.150
LAT_MAX    = 0.400
MAX_STEPS  = 200

COMMANDS = {0: "up", 1: "down", 2: "left", 3: "right"}
DELTA    = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


class Grid:
    """10x10 world with obstacles placed by a seeded RNG."""

    def __init__(self, seed: int = 42):
        self.grid = np.zeros((ROWS, COLS), int)
        self.grid[GOAL] = 2
        rng = np.random.default_rng(seed)
        forbidden = {START, GOAL}
        placed = 0
        while placed < N_OBS:
            r = int(rng.integers(0, ROWS))
            c = int(rng.integers(0, COLS))
            if (r, c) not in forbidden and self.grid[r, c] == 0:
                self.grid[r, c] = 1
                forbidden.add((r, c))
                placed += 1

    def free(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < ROWS and 0 <= c < COLS and self.grid[r, c] != 1

    def scan(self, pos: Tuple[int, int]) -> Dict[int, bool]:
        return {cmd: self.free((pos[0] + dr, pos[1] + dc))
                for cmd, (dr, dc) in DELTA.items()}

    def distance_field(self) -> np.ndarray:
        """BFS step count from every free cell to the goal; inf if unreachable."""
        d = np.full((ROWS, COLS), np.inf)
        d[GOAL] = 0.0
        frontier = [GOAL]
        while frontier:
            nxt = []
            for r, c in frontier:
                for dr, dc in DELTA.values():
                    n = (r + dr, c + dc)
                    if self.free(n) and not np.isfinite(d[n]):
                        d[n] = d[r, c] + 1
                        nxt.append(n)
            frontier = nxt
        return d


@dataclass
class TrialResult:
    trial_id: int
    group: str
    phase: str
    accuracy: float
    reached_goal: bool
    n_steps: int
    n_collisions: int
    n_overrides: int
    n_correct: int
    n_total: int
    latencies: List[float] = field(default_factory=list)
    path: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def cmd_accuracy(self) -> float:
        return self.n_correct / max(self.n_total, 1)

    @property
    def override_rate(self) -> float:
        """Share of commands replaced by the safety layer."""
        return self.n_overrides / max(self.n_total, 1)

    @property
    def mean_lat_ms(self) -> float:
        return 1000.0 * float(np.mean(self.latencies)) if self.latencies else 0.0

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("latencies")
        d.pop("path")
        d["cmd_accuracy"] = self.cmd_accuracy
        d["override_rate"] = self.override_rate
        d["mean_lat_ms"] = self.mean_lat_ms
        return d


def run_trial(group: str, accuracy: float, seed: int,
              phase: str = "Phase1") -> TrialResult:
    """One navigation trial driven by the measured decoding accuracy.

    accuracy is P(decoded command == intended command). A wrong decode picks
    uniformly among the three other commands.
    """
    accuracy = float(np.clip(accuracy, 0.25, 1.0))
    noise_prob = 1.0 - accuracy

    env = simpy.Environment()
    grid = Grid(seed=seed)
    rng = np.random.default_rng(seed)

    res = TrialResult(trial_id=seed, group=group, phase=phase,
                      accuracy=accuracy, reached_goal=False, n_steps=0,
                      n_collisions=0, n_overrides=0, n_correct=0, n_total=0)
    pos = [START[0], START[1]]
    res.path.append(tuple(pos))

    dist = grid.distance_field()

    def intended_cmd() -> int:
        """First step of a shortest path to the goal.

        A greedy "move down then right" rule deadlocks: the safety layer steers
        around an obstacle, the greedy rule immediately steers back, and the
        chair oscillates until the step budget runs out. Goal completion then
        reflects the obstacle layout rather than the decoding accuracy, which
        defeats the purpose of the simulation. Planning on the true distance
        field removes that failure mode, so any variation left in the
        navigation metrics comes from decoding errors.
        """
        best, best_d = 0, np.inf
        for cmd, (dr, dc) in DELTA.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if grid.free(nxt) and dist[nxt] < best_d:
                best, best_d = cmd, dist[nxt]
        return best

    def decode(cmd: int) -> int:
        if rng.random() < noise_prob:
            alts = [x for x in COMMANDS if x != cmd]
            return int(rng.choice(alts))
        return cmd

    def safety(cmd: int, scan: Dict[int, bool]) -> Tuple[int, bool]:
        """Replace a blocked command with the free alternative nearest the goal."""
        if scan.get(cmd, False):
            return cmd, False
        alts = [(dist[(pos[0] + DELTA[a][0], pos[1] + DELTA[a][1])], a)
                for a in DELTA if scan.get(a, False)]
        if not alts:
            return -1, True
        return min(alts)[1], True

    def wheelchair(env):
        step = 0
        while step < MAX_STEPS:
            yield env.timeout(0.25)                 # 4 Hz BCI update rate
            want = intended_cmd()
            got = decode(want)

            lat = float(rng.uniform(LAT_MIN, LAT_MAX))
            yield env.timeout(lat)
            res.latencies.append(lat)

            res.n_total += 1
            if got == want:
                res.n_correct += 1

            final, overridden = safety(got, grid.scan(tuple(pos)))
            if overridden:
                res.n_overrides += 1

            if final == -1:
                res.n_collisions += 1
            else:
                dr, dc = DELTA[final]
                nxt = (pos[0] + dr, pos[1] + dc)
                if grid.free(nxt):
                    pos[0], pos[1] = nxt
                else:
                    res.n_collisions += 1

            res.path.append(tuple(pos))
            res.n_steps += 1
            step += 1

            if tuple(pos) == GOAL:
                res.reached_goal = True
                return

    env.process(wheelchair(env))
    env.run()
    return res


def run_all(accuracies: Dict[str, float], phase: str = "Phase1",
            n_trials: int = 50, seed: int = 42,
            out_json: Optional[str] = None) -> Dict[str, List[TrialResult]]:
    """Run n_trials per group. Returns {group: [TrialResult, ...]}.

    Seeds are derived from the base seed and the trial index only, so the same
    grid layouts are used for every group and every phase. Any difference in
    the navigation metrics therefore comes from the decoding accuracy, not from
    a different world.
    """
    out: Dict[str, List[TrialResult]] = {}
    for group, acc in accuracies.items():
        trials = [run_trial(group, acc, seed=seed + t, phase=phase)
                  for t in range(n_trials)]
        out[group] = trials
        print(f"    {group:<10} acc={acc:.3f}  "
              f"goal={100.0 * np.mean([t.reached_goal for t in trials]):5.1f}%  "
              f"steps={np.mean([t.n_steps for t in trials]):6.1f}  "
              f"coll={np.mean([t.n_collisions for t in trials]):5.2f}  "
              f"block={100.0 * np.mean([t.override_rate for t in trials]):5.1f}%  "
              f"lat={np.mean([t.mean_lat_ms for t in trials]):5.0f}ms")

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump({g: [t.summary() for t in ts] for g, ts in out.items()},
                      f, indent=2, default=float)
    return out


if __name__ == "__main__":
    run_all({"healthy": 0.84, "stroke": 0.71}, phase="Phase1", n_trials=10)