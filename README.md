# Model Cortical Folding

This is a final project for CSE 397.
This repository implements and compares two biophysical models of cortical folding:
1. **Mechanical Buckling**  
2. **Turing Reaction–Diffusion**  

## Structure

- **project_root/**
  - **notebooks/**
    - `buckling_folding.ipynb` &mdash; simulate 2D mechanical buckling  
    - `turing_folding.ipynb` &mdash; simulate 2D Turing pattern  
    - `comparison.ipynb` &mdash; sensitivity analysis & symbolic regression  
  - **model/**
    - `__init__.py`  
    - `buckling.py`  
    - `turing.py`  
  - **utils/**
    - `__init__.py`  
    - `math_utils.py`  
    - `plot_utils.py`  
  - `environment.yml` &mdash; conda environment spec  
  - `README.md`

## Installation
1. **Clone** the repository  
  ```bash
  git clone https://github.com/JiachenWang512/Cortical_Folding
  ```
2.	**Create and activate** the Conda environment
   ```bash
   conda env create -f environment.yml -n cortical_folding
   conda activate cortical_folding
   ```

## Quick start
Run `notebooks/buckling_folding.ipynb` to see 1D, 2D simulation and visuals.
