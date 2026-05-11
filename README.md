# 2-Link Robot Arm Motion Planner

A motion planning agent for a 2-link planar robot arm navigating a 2D workspace with obstacles. Built for CS450 (Intro to AI) at SDSU Georgia.

## Motion Planning in Robotics

Motion planning is the problem of finding a collision-free path for a robot from a start configuration to a goal configuration. The key idea is to work in **configuration space (C-space)** rather than physical space: instead of reasoning about the robot's body in 2D/3D, we map each possible robot configuration (here, the two joint angles θ₁, θ₂) to a single point. Obstacles in physical space become regions in C-space, and path planning reduces to finding a curve through the free (non-colliding) region of that space.

For a 2-link planar arm, C-space is 2D — one axis per joint angle — so it can be discretized into a bitmap grid. Each cell is marked **free** or **in-collision** by checking whether the arm's links intersect any obstacle at that configuration. A graph search (A\*) then finds the shortest collision-free path through the grid.

This approach is the foundation of more advanced planners like PRM (Probabilistic Roadmap) and RRT (Rapidly-exploring Random Trees), which handle higher-dimensional C-spaces by sampling rather than discretization.

## Project Structure

```
.
├── demo.ipynb              # Main notebook — walkthrough
└── src/
    ├── workspace.py        # Workspace, robot, and obstacle definitions
    ├── cspace.py           # C-space bitmap construction and index mapping
    ├── path_planning.py    # Grid A* planner
    ├── astar.py            # Generic A* implementation
    └── visualization.py    # Workspace + C-space rendering and interactive planner
```

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Vectorized forward kinematics, C-space bitmap construction |
| `matplotlib` | Workspace and C-space visualization |
| `scipy` | `scipy.ndimage.label` for connected-component analysis |
| `ipympl` | Interactive `%matplotlib widget` backend (local Jupyter only) |

## Run Locally

**Requirements:** Python 3.10+

```bash
git clone git@github.com:guramrekh/motion-planning.git
cd motion-planning

python -m venv .venv
source ./.venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

jupyter notebook
```

Then open `demo.ipynb` from the Jupyter file browser.
