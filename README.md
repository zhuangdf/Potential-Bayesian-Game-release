# Potential Bayesian Game Trajectory Planning

Trajectory-planning demos accompanying **“Fast and Scalable Game-Theoretic Trajectory Planning with Intentional Uncertainties”**.

The project models uncertain driving intentions as discrete vehicle types and optimizes their probability-weighted tracking, control, and collision costs. A type node represents one possible intention of a physical vehicle, so the number of optimization agents can exceed the number of actual vehicles.

## Methods and entry points

| Entry point | Method | Scenarios |
| --- | --- | --- |
| [`pb_game.py`](pb_game.py) | Main graph-based distributed potential Bayesian game solver. It builds local trajectory approximations, coordinates neighboring type nodes with ADMM, and solves local LQR subproblems. | Intersection (`C`), merge (`M`) |
| [`pb_game_DiLQR.py`](pb_game_DiLQR.py) | Distributed iLQR comparison implementation. Each agent maintains dual variables over all type-node pairs and controls, giving a larger coordination problem than the neighbor-based representation in the main solver. | Intersection (`C`), merge (`M`) |
| [`pb_game_contingency.py`](pb_game_contingency.py) | Contingency planning with alternative trajectory branches. A quadratic penalty encourages the ego branches to agree during the first five time steps before diverging for different hypotheses. | Dedicated two-hypothesis demo |

All three implementations provide serial and multiprocessing solvers. In the two command-line demos, a multiprocessing worker corresponds to a type node.

## Installation

Run the following commands from the repository root. Python 3.12 on Ubuntu has been used for validation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If creating the environment fails because `ensurepip` is unavailable, install the venv package matching your Python version, for example:

```bash
sudo apt update
sudo apt install python3.12-venv
```

The core dependencies are NumPy, SciPy, SymPy, Numba, and Matplotlib. SciPy is also needed by Numba's matrix operations.

Video export additionally requires FFmpeg:

```bash
sudo apt install ffmpeg
```

The examples below use `--no-video` so they can run without FFmpeg. On a server without a graphical display, prefix a command with `MPLBACKEND=Agg`.

## Main solver: `pb_game.py`

```bash
# Two-car merge
python pb_game.py M --no-video

# Three-car intersection
python pb_game.py C --no-video

# Serial solver
python pb_game.py C --solver single --no-video

# Run serial and multiprocessing solvers from separate initializations
python pb_game.py M --solver both --no-plot

# Short smoke run; this does not establish convergence
python pb_game.py C --solver both --max-iter 3 --admm-iter 3 --no-plot

python pb_game.py --help
```

Both `C` and `M` select the appropriate reference trajectories, tracking weights, and collision-distance parameters. Lowercase `c` and `m` are also accepted. Omitting the scene selects `M`.

## Distributed iLQR: `pb_game_DiLQR.py`

This entry point accepts the same command-line options as the main solver:

```bash
python pb_game_DiLQR.py M --no-video
python pb_game_DiLQR.py C --no-video
python pb_game_DiLQR.py C --solver both --max-iter 3 --no-plot
python pb_game_DiLQR.py --help
```

The default iteration counts differ between the two methods:

| Option | Meaning | `pb_game.py` default | `pb_game_DiLQR.py` default |
| --- | --- | --- | --- |
| `scene` | `C`: intersection; `M`: merge | `M` | `M` |
| `--solver` | `single`, `multi`, or `both` | `multi` | `multi` |
| `--max-iter` | Maximum outer iterations | `2000` | `500` |
| `--admm-iter` | ADMM iterations per outer iteration | `3` | `2` |
| `--tolerance` | Stop when the absolute change in total cost is below this value | `0.1` | `0.1` |
| `--no-video` | Save the static image without exporting an MP4 | Off | Off |
| `--no-plot` | Skip image and video output | Off | Off |

With `--solver both`, the displayed/exported trajectory is the final multiprocessing result. First-use Numba compilation contributes to the printed runtime, so these timings alone should not be treated as a controlled benchmark.

## Contingency demo: `pb_game_contingency.py`

```bash
python pb_game_contingency.py
```

This demo currently uses settings in its `if __name__ == "__main__":` block rather than command-line `C/M` arguments:

- Two physical vehicles and two lane-intention hypotheses.
- Lane targets: `[0.5, 0]`; the other vehicle is assigned the opposite lane target in each hypothesis.
- A single reference speed of `0.5 m/s` for the other vehicle, with hypothesis probabilities `[0.5, 0.5]`.
- Two branches per vehicle, giving four trajectories in total.
- 25 time steps at `0.1 s` intervals.

It runs both serial and multiprocessing solvers, then saves the **serial** result to `result.jpg`. This entry point does not export a video. Its first-five-step branch agreement is a soft penalty, not a hard equality constraint.

## Outputs and trajectory format

The ordinary intersection and merge demos use 100 time steps at `0.1 s` intervals. The default configurations expand to 21 type nodes for `C` and 11 for `M`.

| Output | Description |
| --- | --- |
| Terminal logs | Iteration costs and solver runtimes |
| `result.jpg` | Static trajectory plot, with one vehicle pentagon at the final state of each branch |
| `result.mp4` | Animation from the intersection/merge demos, unless `--no-video` or `--no-plot` is set |

Outputs are written to the **current working directory**. Reusing the same directory overwrites files with the same names.

In intersection/merge plots, colors identify physical vehicles and line opacity represents the relative type probability within each vehicle. Endpoint pentagons do not encode probability. In contingency plots, colors identify the two vehicles and line styles distinguish branches.

Trajectories have shape `(N, T, 6)`:

```text
[x, y, yaw, speed, steering, acceleration]
```

Positions are in meters, angles in radians, speed in meters per second, and acceleration in meters per second squared. `N` is the number of type/branch nodes, not necessarily the number of physical vehicles.

To save numerical results from a command-line demo through Python:

```python
import numpy as np
from pb_game import main

trajectory = main(["M", "--no-plot"])
np.save("trajectory.npy", trajectory)
```

## Source layout

```text
pb_game.py                          Main distributed solver and CLI
pb_game_DiLQR.py                    Distributed iLQR comparison and CLI
pb_game_contingency.py              Two-hypothesis contingency demo
scenarios.py                       Scene construction and plotting
Utilities/pb_Utilities.py           Main solver kernels
Utilities/pb_DiLQR_Utilities.py      Distributed iLQR kernels
Utilities/pb_contingency_Utilities.py  Contingency kernels
requirements.txt                   Python dependencies
```

The collision model uses two circles per vehicle and probability-weighted soft distance penalties. Solver termination or finite trajectories alone do not guarantee collision-free motion or a global optimum.
