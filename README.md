# Price of Anarchy in Multi-Agent Pathfinding (PoA-MAPF)

This repository provides a compact simulation framework for studying selfish routing and system efficiency in congestion games.

## Implemented Core

- **Price of Anarchy (PoA)** computation:
  - Worst Nash Equilibrium (iterative best response)
  - Social optimum (min-cost max-flow on convex marginal costs)
- **Braess's paradox** utilities:
  - Classic Braess graph constructor
  - Edge-removal detector for paradoxical edges
- **Congestion-aware shortest paths** (Dijkstra with load-dependent edge latency)
- **Limited information mode** for decentralized agents (`limited_information=True`, noisy edge-load beliefs)
- **Adversarial agents** via `adversarial_fraction`
- **Frank-Wolfe baseline** for polynomial edge costs (`alpha * x^power + beta`, default power=4)
- **Barabási–Albert graph generation** for scale-free experiments
- **Pigovian toll utility** (`pigovian_toll`) for mechanism design experiments

## Quick Start

```bash
python main.py
python -m unittest -v
```

## GitHub Pages Dashboard

A minimal dark-mode research dashboard is available at `docs/index.html`, powered by:
- **Native HTML5 Canvas** for plots
- **Styled HTML/Unicode math text** for equations
- `docs/results.json` for simulation output data

To host on GitHub Pages (`github.io`):
1. Go to repository **Settings → Pages**
2. Set source to **Deploy from a branch**
3. Choose branch (e.g. `main`) and folder **`/docs`**
4. Save; GitHub will publish the site URL

For this repository, the expected project Pages URL is:

`https://arnavd371.github.io/Price-of-Anarchy-in-Multi-Agent-Pathfinding-PoA-MAPF-/`

Dashboard page:

`https://arnavd371.github.io/Price-of-Anarchy-in-Multi-Agent-Pathfinding-PoA-MAPF-/docs/`

## Main API

Located in `src/poa_mapf/core.py`.

Key functions:
- `iterative_best_response(...)`
- `min_cost_max_flow_social_optimum(...)`
- `price_of_anarchy(...)`
- `frank_wolfe_equilibrium(...)`
- `detect_braess_edges(...)`
- `barabasi_albert_graph(...)`

Notes:
- `build_braess_graph(..., demand_scale=N)` scales variable-edge latency to model the standard normalized Braess demand setting for `N` atomic agents.
