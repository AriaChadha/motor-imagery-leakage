"""
simulation_06.py
================
SimPy wheelchair navigation driven by REAL classifier accuracy.

Control paradigm
----------------
The decoders are binary (left hand vs right hand), so the simulation uses
shared control (Millan et al. 2010; Rebsamen et al. 2010): at each control
cycle an autonomous planner proposes the two best moves from the current
cell, and the binary BCI command selects between them. Two classes in, two
options out. A four-command interface is not simulated because a two-class
decoder cannot produce one without fabricating commands.

Accuracy source
---------------
`accuracy` must be passed in from the classifier results. There are no
default accuracies: hardcoded values would let the simulation report
navigation outcomes unrelated to the decoding experiment.

Safety subsystem
----------------
safety_enabled=True  : blocked directions are removed before ranking, so
                       collisions are impossible and the metric of interest
                       is the command-blocking rate.
safety_enabled=False : all four neighbours are ranked, so a misdecode can
                       select an obstacle and collide.
Running both conditions is what quantifies what the safety layer buys.
"""

import sys
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import simpy

ROWS, COLS = 10, 10
N_OBS      = 12
START      = (0, 0)
GOAL       = (9, 9)
LAT_MIN    = 0.150
LAT_MAX    = 0.400
CYCLE_SEC  = 0.25            # 4 Hz BCI update rate
MAX_STEPS  = 200
GROUPS     = ("healthy", "stroke")
GROUP_COLORS = {"healthy": "#2196F3", "stroke": "#F44336"}

DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


# ----------------------------------------------------------------------
# Grid world with a real planner
# ----------------------------------------------------------------------

class Grid:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        for _ in range(200):
            g = np.zeros((ROWS, COLS), int)
            placed = 0
            while placed < N_OBS:
                r, c = int(rng.integers(0, ROWS)), int(rng.integers(0, COLS))
                if (r, c) in (START, GOAL) or g[r, c] == 1:
                    continue
                g[r, c] = 1
                placed += 1
            self.grid = g
            self.dist = self._bfs_from_goal()
            if np.isfinite(self.dist[START]):
                return
        raise RuntimeError("Could not generate a solvable grid.")

    def _bfs_from_goal(self) -> np.ndarray:
        d = np.full((ROWS, COLS), np.inf)
        d[GOAL] = 0
        q = deque([GOAL])
        while q:
            r, c = q.popleft()
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < ROWS and 0 <= nc < COLS
                        and self.grid[nr, nc] != 1 and np.isinf(d[nr, nc])):
                    d[nr, nc] = d[r, c] + 1
                    q.append((nr, nc))
        return d

    def free(self, pos) -> bool:
        r, c = pos
        return 0 <= r < ROWS and 0 <= c < COLS and self.grid[r, c] != 1

    def in_bounds(self, pos) -> bool:
        r, c = pos
        return 0 <= r < ROWS and 0 <= c < COLS

    def optimal_steps(self) -> float:
        return float(self.dist[START])

    def rank_options(self, pos, safety: bool) -> List[Tuple[int, Tuple[int, int]]]:
        """Neighbour moves ordered by distance-to-goal, best first.

        With safety on, obstacles are removed. With safety off they remain
        in the ranking and can therefore be selected by a misdecode.
        """
        opts = []
        for cmd, (dr, dc) in DELTA.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not self.in_bounds(nxt):
                continue
            if safety and not self.free(nxt):
                continue
            d = self.dist[nxt] if self.free(nxt) else np.inf
            opts.append((d, cmd, nxt))
        opts.sort(key=lambda t: (t[0], t[1]))
        return [(cmd, nxt) for _, cmd, nxt in opts]


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

@dataclass
class TrialResult:
    trial_id: int
    group: str
    phase: str
    safety_enabled: bool
    reached_goal: bool = False
    n_steps: int = 0
    n_collisions: int = 0
    n_commands: int = 0
    n_correct_decodes: int = 0
    n_blocked: int = 0           # cycles where the safety layer removed options
    n_no_option: int = 0         # cycles with nothing available at all
    optimal_steps: float = 0.0
    latencies: List[float] = field(default_factory=list)
    path: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def decode_accuracy(self) -> float:
        return self.n_correct_decodes / max(self.n_commands, 1)

    @property
    def block_rate(self) -> float:
        return self.n_blocked / max(self.n_commands, 1)

    @property
    def mean_lat_ms(self) -> float:
        return 1000.0 * float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def path_efficiency(self) -> float:
        """Optimal steps divided by steps taken. 1.0 is a perfect route."""
        return self.optimal_steps / self.n_steps if self.n_steps else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path")
        d.update(decode_accuracy=self.decode_accuracy,
                 block_rate=self.block_rate,
                 mean_lat_ms=self.mean_lat_ms,
                 path_efficiency=self.path_efficiency)
        return d


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------

def run_simpy_trial(group: str, accuracy: float, seed: int,
                    phase: str = "Phase1", safety_enabled: bool = True,
                    callback=None) -> TrialResult:
    """One navigation trial.

    accuracy is the measured binary decoding accuracy for this group and
    phase. It sets the probability that the BCI selects the planner's
    preferred option rather than the alternative.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")

    env = simpy.Environment()
    grid = Grid(seed=seed)
    rng = np.random.default_rng(seed)

    res = TrialResult(trial_id=seed, group=group, phase=phase,
                      safety_enabled=safety_enabled,
                      optimal_steps=grid.optimal_steps())
    pos = [START[0], START[1]]
    res.path.append(START)

    def control_loop(env):
        for step in range(MAX_STEPS):
            yield env.timeout(CYCLE_SEC)

            all_opts = grid.rank_options(tuple(pos), safety=False)
            opts = grid.rank_options(tuple(pos), safety=safety_enabled)
            if safety_enabled and len(opts) < len(all_opts):
                res.n_blocked += 1
            if not opts:
                res.n_no_option += 1
                break

            lat = float(rng.uniform(LAT_MIN, LAT_MAX))
            yield env.timeout(lat)
            res.latencies.append(lat)

            # Shared control: planner offers the two best moves, the binary
            # BCI picks one. With a single option there is nothing to decode.
            res.n_commands += 1
            if len(opts) == 1:
                chosen, correct = opts[0], True
            elif rng.random() < accuracy:
                chosen, correct = opts[0], True
            else:
                chosen, correct = opts[1], False
            res.n_correct_decodes += int(correct)

            cmd, nxt = chosen
            if grid.free(nxt):
                pos[0], pos[1] = nxt
            else:
                res.n_collisions += 1      # only reachable with safety off

            res.path.append(tuple(pos))
            res.n_steps += 1

            if callback:
                callback({"type": "step", "pos": tuple(pos),
                          "path": list(res.path), "grid": grid.grid,
                          "collisions": res.n_collisions,
                          "latency_ms": lat * 1000.0,
                          "cmd": NAMES[cmd],
                          "accuracy": res.decode_accuracy,
                          "step": res.n_steps})

            if tuple(pos) == GOAL:
                res.reached_goal = True
                return

    env.process(control_loop(env))
    env.run()
    return res


def run_headless(group: str, accuracy: float, phase: str = "Phase1",
                 n_trials: int = 20, safety_enabled: bool = True,
                 seed0: int = 42, verbose: bool = True) -> List[TrialResult]:
    """Run n_trials for one group. accuracy must come from the classifier."""
    results = []
    if verbose:
        print(f"  [{group}] {phase} accuracy={accuracy:.3f} "
              f"safety={'on' if safety_enabled else 'off'} "
              f"trials={n_trials}")
    for t in range(n_trials):
        results.append(run_simpy_trial(group, accuracy, seed0 + t, phase,
                                       safety_enabled))
    if verbose:
        print(f"    goal {summarize(results)['goal_rate']:.0%}  "
              f"collisions/trial {summarize(results)['collisions_mean']:.2f}  "
              f"blocked {summarize(results)['block_rate']:.1%}")
    return results


def summarize(results: List[TrialResult]) -> Dict[str, float]:
    """Aggregate metrics across trials, for the results tables."""
    if not results:
        return {}
    lat = [l for r in results for l in r.latencies]
    return {
        "n_trials": len(results),
        "goal_rate": float(np.mean([r.reached_goal for r in results])),
        "steps_mean": float(np.mean([r.n_steps for r in results])),
        "collisions_mean": float(np.mean([r.n_collisions for r in results])),
        "collisions_total": int(sum(r.n_collisions for r in results)),
        "decode_accuracy": float(np.mean([r.decode_accuracy for r in results])),
        "block_rate": float(np.mean([r.block_rate for r in results])),
        "path_efficiency": float(np.mean([r.path_efficiency for r in results])),
        "latency_ms_mean": float(np.mean(lat) * 1000.0) if lat else 0.0,
        "latency_ms_p95": float(np.percentile(lat, 95) * 1000.0) if lat else 0.0,
    }


def run_all(accuracies: Dict[str, float], phase: str = "Phase1",
            n_trials: int = 20, out_json: Optional[str] = None) -> Dict:
    """Run every group under both safety conditions.

    accuracies maps group name to measured decoding accuracy, e.g.
    {"healthy": 0.612, "stroke": 0.554}.
    """
    out = {}
    for group, acc in accuracies.items():
        out[group] = {}
        for safety in (True, False):
            key = "safety_on" if safety else "safety_off"
            rs = run_headless(group, acc, phase, n_trials, safety)
            out[group][key] = summarize(rs)

    if out_json:
        with open(out_json, "w") as f:
            json.dump({"phase": phase, "accuracies": accuracies,
                       "results": out}, f, indent=2)
        print(f"  wrote {out_json}")
    return out


# ----------------------------------------------------------------------
# Optional GUI
# ----------------------------------------------------------------------

def launch_gui(accuracies: Optional[Dict[str, float]] = None):
    """Live dashboard. Requires measured accuracies; none are invented."""
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                                     QHBoxLayout, QVBoxLayout, QLabel,
                                     QPushButton, QComboBox, QSpinBox,
                                     QGroupBox, QGridLayout, QProgressBar,
                                     QTextEdit, QSplitter, QSizePolicy)
        from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal
        from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen
        import matplotlib
        matplotlib.use("Qt5Agg")
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
        from matplotlib.figure import Figure
        import matplotlib.ticker as mticker
    except ImportError:
        print("PyQt5 and matplotlib required: pip install PyQt5 matplotlib")
        return

    if not accuracies:
        print("launch_gui needs measured accuracies, e.g. "
              "launch_gui({'healthy': 0.61, 'stroke': 0.55})")
        return

    class GridWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.grid_data = Grid(42).grid
            self.path, self.pos = [START], START
            self.setMinimumSize(320, 320)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def update_state(self, grid_data, path, pos):
            self.grid_data, self.path, self.pos = grid_data, path, pos
            self.update()

        def paintEvent(self, _):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            w, h = self.width() // COLS, self.height() // ROWS
            for r in range(ROWS):
                for c in range(COLS):
                    col = QColor("#424242") if self.grid_data[r, c] == 1 else \
                          QColor("#00C853") if (r, c) == GOAL else QColor("#F5F5F5")
                    p.fillRect(c * w, r * h, w - 1, h - 1, col)
            p.setPen(QPen(QColor("#1565C0"), 2))
            for i in range(1, len(self.path)):
                r0, c0 = self.path[i - 1]
                r1, c1 = self.path[i]
                p.drawLine(c0 * w + w // 2, r0 * h + h // 2,
                           c1 * w + w // 2, r1 * h + h // 2)
            r, c = self.pos
            p.setBrush(QBrush(QColor("#E53935")))
            p.setPen(Qt.NoPen)
            p.drawEllipse(c * w + w // 6, r * h + h // 6, 2 * w // 3, 2 * h // 3)

    class Worker(QObject):
        step = pyqtSignal(dict)
        trial_done = pyqtSignal(object)
        finished = pyqtSignal()

        def __init__(self, jobs, n_trials, phase, safety):
            super().__init__()
            self.jobs, self.n_trials = jobs, n_trials
            self.phase, self.safety = phase, safety
            self._stop = False

        def run(self):
            for group, acc in self.jobs:
                for t in range(self.n_trials):
                    if self._stop:
                        break
                    r = run_simpy_trial(group, acc, 42 + t, self.phase,
                                        self.safety, self.step.emit)
                    self.trial_done.emit(r)
            self.finished.emit()

        def stop(self):
            self._stop = True

    class Win(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("EEG Wheelchair BCI Simulation")
            self.setMinimumSize(1280, 780)
            self.results = {g: [] for g in GROUPS}
            self.thread = self.worker = None
            self._build()

        def _build(self):
            central = QWidget()
            self.setCentralWidget(central)
            split = QSplitter(Qt.Horizontal)
            QHBoxLayout(central).addWidget(split)

            left = QWidget()
            ll = QVBoxLayout(left)
            title = QLabel("EEG Wheelchair BCI Simulation")
            title.setStyleSheet("font-size:16px;font-weight:bold;color:#1a237e;")
            ll.addWidget(title)

            ctrl = QGroupBox("Controls")
            cg = QGridLayout(ctrl)
            cg.addWidget(QLabel("Group:"), 0, 0)
            self.grp = QComboBox()
            self.grp.addItems(list(accuracies) + ["All"])
            cg.addWidget(self.grp, 0, 1)
            cg.addWidget(QLabel("Phase:"), 1, 0)
            self.phase = QComboBox()
            self.phase.addItems(["Phase1", "Phase2"])
            cg.addWidget(self.phase, 1, 1)
            cg.addWidget(QLabel("Safety:"), 2, 0)
            self.safety = QComboBox()
            self.safety.addItems(["on", "off"])
            cg.addWidget(self.safety, 2, 1)
            cg.addWidget(QLabel("Trials:"), 3, 0)
            self.ntr = QSpinBox()
            self.ntr.setRange(1, 50)
            self.ntr.setValue(10)
            cg.addWidget(self.ntr, 3, 1)
            self.run_btn = QPushButton("Run")
            self.run_btn.setStyleSheet("background:#1565C0;color:white;"
                                       "font-weight:bold;padding:8px;")
            self.run_btn.clicked.connect(self._run)
            cg.addWidget(self.run_btn, 4, 0, 1, 2)
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet("background:#C62828;color:white;"
                                        "font-weight:bold;padding:8px;")
            self.stop_btn.clicked.connect(self._stop)
            cg.addWidget(self.stop_btn, 5, 0, 1, 2)
            ll.addWidget(ctrl)

            gb = QGroupBox("Grid World")
            gl = QVBoxLayout(gb)
            self.gw = GridWidget()
            gl.addWidget(self.gw)
            self.status = QLabel("Ready")
            gl.addWidget(self.status)
            self.prog = QProgressBar()
            self.prog.setRange(0, MAX_STEPS)
            gl.addWidget(self.prog)
            ll.addWidget(gb)

            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(130)
            self.log.setStyleSheet("font-family:monospace;font-size:11px;")
            ll.addWidget(self.log)
            split.addWidget(left)

            right = QWidget()
            rl = QVBoxLayout(right)
            self.canvas = FigCanvas(Figure(figsize=(9, 7), facecolor="#fafafa"))
            rl.addWidget(self.canvas)
            self.axes = self.canvas.figure.subplots(2, 2)
            self.canvas.figure.subplots_adjust(hspace=.42, wspace=.35)
            split.addWidget(right)
            split.setSizes([480, 800])
            self.mticker = mticker

        def _run(self):
            sel = self.grp.currentText()
            jobs = [(g, accuracies[g]) for g in accuracies] if sel == "All" \
                else [(sel, accuracies[sel])]
            self.results = {g: [] for g in GROUPS}
            self.log.clear()
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            self.worker = Worker(jobs, self.ntr.value(),
                                 self.phase.currentText(),
                                 self.safety.currentText() == "on")
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.step.connect(self._on_step, Qt.QueuedConnection)
            self.worker.trial_done.connect(self._on_trial, Qt.QueuedConnection)
            self.worker.finished.connect(self._on_finish, Qt.QueuedConnection)
            self.thread.start()

        def _stop(self):
            if self.worker:
                self.worker.stop()

        def _on_step(self, info):
            self.gw.update_state(info["grid"], info["path"], info["pos"])
            self.prog.setValue(info["step"])
            self.status.setText(
                f"step {info['step']}  cmd {info['cmd']}  "
                f"decode {info['accuracy']:.0%}  "
                f"lat {info['latency_ms']:.0f}ms  "
                f"coll {info['collisions']}")

        def _on_trial(self, r):
            self.results.setdefault(r.group, []).append(r)
            self.log.append(
                f"[{r.group}] {'GOAL' if r.reached_goal else 'MISS'} "
                f"steps={r.n_steps} coll={r.n_collisions} "
                f"blocked={r.block_rate:.0%} eff={r.path_efficiency:.2f}")
            self._charts()

        def _on_finish(self):
            self.thread.quit()
            self.thread.wait()
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log.append("Complete.")

        def _charts(self):
            names = [g for g in GROUPS if self.results.get(g)]
            if not names:
                return
            for ax in self.axes.flat:
                ax.clear()
            cols = [GROUP_COLORS[g] for g in names]

            ax = self.axes[0, 0]
            ax.bar(names, [np.mean([r.decode_accuracy for r in self.results[g]])
                           for g in names], color=cols, alpha=.85)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(self.mticker.PercentFormatter(1.0))
            ax.set_title("Decode accuracy", fontsize=9, fontweight="bold")

            ax = self.axes[0, 1]
            for g in names:
                ax.plot(range(1, len(self.results[g]) + 1),
                        [r.n_collisions for r in self.results[g]], "o-",
                        color=GROUP_COLORS[g], label=g)
            ax.set_title("Collisions per trial", fontsize=9, fontweight="bold")
            ax.legend(fontsize=7)

            ax = self.axes[1, 0]
            ax.boxplot([[1000 * l for r in self.results[g] for l in r.latencies]
                        for g in names], labels=names)
            ax.axhline(250, color="red", ls="--", lw=1)
            ax.set_title("Latency (ms)", fontsize=9, fontweight="bold")

            ax = self.axes[1, 1]
            ax.bar(names, [100 * np.mean([r.reached_goal for r in self.results[g]])
                           for g in names], color=cols, alpha=.85)
            ax.set_ylim(0, 115)
            ax.set_title("Goal completion (%)", fontsize=9, fontweight="bold")

            for ax in self.axes.flat:
                ax.grid(alpha=.3)
            self.canvas.draw()

    app = QApplication.instance() or QApplication(sys.argv)
    w = Win()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    print("Demo accuracies only. Real runs take them from the classifier.")
    demo = {"healthy": 0.60, "stroke": 0.55}
    run_all(demo, phase="demo", n_trials=20)