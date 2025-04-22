# Model Cortical Folding

This is a final project for CSE 397.

This repository compares mechanical buckling and Turing reaction-diffusion models of cortical folding:

- **mechanical_model**: Implements the Euler–Bernoulli buckling formulation.
- **turing_model**: Implements a two-species reaction–diffusion system.
- **comparison**: Quantifies similarities between pattern outputs.
- **utils**: Helper functions for numerics and plotting.
- **notebooks**: Interactive demonstrations.

## Installation

```bash
pip install -r requirements.txt
```
## Example
Run simulations from Python scripts:

```python
from cortical_folding.mechanical_model import simulate_buckling
from cortical_folding.turing_model import simulate_turing
buckling = simulate_buckling(params, domain)
turing = simulate_turing(params, domain, init_cond, steps)
```
Open `notebooks/mechanical_vs_turing.ipynb` to see side-by-side visualizations.
