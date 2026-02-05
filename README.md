# Robust-Dynamic-Game-SLS
Implementation of algorithm proposed in [Robustly Constrained Dynamic Games for Uncertain Dynamics](https://arxiv.org/abs/2509.16826). 
If you are interested, please take a look at the [video](https://drive.google.com/file/d/1K_BZBQ4TzrMYfRazLCbyDwR1ElJraAgF/view?usp=drive_link) introducing this work. 

## Setup
- `conda create -n robust-dyna-game-sls python=3.10`
- `conda activate robust-dyna-game-sls`
- `pip install -r requirements.txt`
- `pip install -e .`

## Introduction to Repo Layout

The dyn directory contains the definitions of dynamics used in the experiments.

The expe directory contains code for running experiments. 

The solver directory contains implementation of our method. 