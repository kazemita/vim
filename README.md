# Learned VIM multipliers: code and results

Repository: https://github.com/kazemita/vim

Code accompanying the manuscript *Learning the Lagrange multiplier of the variational iteration method: a neural-ODE
quasi-Newton framework for nonlinear Volterra equations*.

```bash
git clone https://github.com/kazemita/vim.git
cd vim
pip install torch numpy scipy matplotlib
```

Trained generators (`*.pt`) and raw results (`*.json`) are in `results/`. Copy them next to the scripts, or run the scripts from
`results/`, to rebuild the tables and figures without retraining (`python make_figures.py`).

Everything runs on a CPU (PyTorch, double precision). Each script prints progress and writes JSON results; `make_figures.py`
turns the JSON files into every table (`paper/tables/*.tex`) and figure (`paper/figures/*`) of the experiments section.

## Core modules
| file | content |
|---|---|
| `vim_node.py` | scalar benchmarks (env `KERNEL=osc|gauss`, `KAPPA`): Gregory quadrature, problem family, `LambdaNet` generator, derivative-form cell `vim_step` (Prop. 1), integral-form cell `vim_step_int` (Prop. 2), Picard/textbook/Newton baselines |
| `neural_pop.py` | E/I population model (env `NP_T=30 NP_SG=6`): delayed gamma kernels, Nyström marching reference, Newton, integral-form graph cell, Padé multiplier, `GraphLambda` (from scratch), `GraphLambdaPade` (Padé-anchored), Anderson |

## Experiments (section numbers refer to the manuscript)
| script | produces | section |
|---|---|---|
| `train_lambda.py --m M --phys 1 --seed S [--unroll K]` | `lam_<tag>.pt`, `eval_<tag>.json` | 4.1 benchmarks, 4.2 ablations |
| `validate_theory.py [--skip-e3]` | `validate_theory.json` (E4: ε_θ, n*, rates) | 4.2 |
| `validate_e3.py` | E3 (untrained generators, integral form) in `validate_theory.json` | 4.2 |
| `train_np.py --init scratch|pade` | `lam_np_*.pt`, `eval_np_*.json` | 4.3 |
| `eval_hybrid.py` | `eval_np_hybrid.json` (learned→Padé schedule) | 4.3 |
| `train_part2.py <variant> <problem>` | `part2_<variant>_p<k>.json` (Mode B) | 4.4 |
| `make_figures.py`, `fig_schematic.py`, `fig_network_example.py` | tables and figures | all |

Typical CPU times (one core): scalar generator 6–23 min; network generator about 24 min; Mode B 10 s (frozen multiplier) to 2 min (online).

## Reproduce
```bash
KERNEL=osc   KAPPA=3.0 python train_lambda.py --m 4 --phys 1 --seed 0
KERNEL=gauss KAPPA=3.0 python train_lambda.py --m 4 --phys 1 --seed 0
for m in 0 1 2 8; do KERNEL=osc KAPPA=3.0 python train_lambda.py --m $m --phys 1 --baselines 0; done
KERNEL=osc KAPPA=3.0 python validate_theory.py && KERNEL=osc KAPPA=3.0 python validate_e3.py
NP_T=30 NP_SG=6 python train_np.py --init scratch --steps 250 --batch 8 --unroll 5
NP_T=30 NP_SG=6 python train_np.py --init pade    --steps 200 --batch 8 --unroll 6
NP_T=30 NP_SG=6 python eval_hybrid.py
for p in 0 1 2 3 4 5 6 7 8 9; do for v in picard local learned_frozen_phys; do KERNEL=osc KAPPA=3.0 python train_part2.py $v $p; done; done
python make_figures.py
```

## Citation
If you use this code, please cite the paper (reference to be completed on publication) and the repository:

```bibtex
@misc{vimcode,
  author       = {{kazemita}},
  title        = {Learned {L}agrange multipliers for the variational iteration method: code, trained multipliers and results},
  year         = {2026},
  howpublished = {\url{https://github.com/kazemita/vim}}
}
```
