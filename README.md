# camppca_paper

Analyses and figures for the camppca manuscript. The method itself is
[camppca](https://github.com/quantitative-mri-and-in-vivo-histology/camppca), pinned here as a
submodule.

## Install

You need Docker, FSL 6.0.7 and ANTs 2.6.2 on the host. eddy needs an NVIDIA GPU. The panel's HCP
scans need HCP AWS credentials. NORDIC on the EDDEN complex data needs MATLAB.

```bash
git clone --recurse-submodules git@github.com:quantitative-mri-and-in-vivo-histology/camppca_paper.git
cd camppca_paper
cp .devcontainer/.env.example .devcontainer/.env   # set DATA_PATH, FSLDIR, ANTSDIR
```

Then open the folder in VS Code and choose "Reopen in Container". Without VS Code:

```bash
docker compose -f .devcontainer/docker-compose.yml -f .devcontainer/gpu/docker-compose.gpu.yml up -d --build
docker exec -it camppca_paper_devcontainer-devcontainer-1 /usr/local/bin/post-create.sh
```

The data root `DATA_PATH` appears as `data/` inside the container. All commands below run from
the repository root inside the container.

## Layout

```
scripts/datasets/<dataset>/   download (fetch.py), checked against pinned hashes
scripts/analyses/<dataset>/   numbered stages, run in order by run.py, each skipping finished work
scripts/figures/main|supp/    one script per figure, written to data/results/main|supp/<name>.pdf
src/camppca_paper/            shared code
```

## Reproducing the figures

| Figures | Data | Commands |
|---|---|---|
| `simulation`, `simulation_noise_draws`, `simulation_matrix_geometry` | subject phantom (OpenNeuro ds004666) | `datasets/subject_phantom/fetch.py`, `prepare.py`, then the figure scripts |
| `exvivo`, `exvivo_maps` | ex vivo mouse brain (Zenodo 10200469) | `datasets/exvivo_mouse/fetch.py`, `analyses/exvivo_mouse/run.py` |
| `invivo_panel`, `regional_correlation_transfer`, `correlation_convergence` | 28 configurations from OpenNeuro and HCP | `datasets/panel/fetch.py`, `analyses/panel/run.py` |
| `invivo_repeat_harmony`, `invivo_reprod_harmony`, `invivo_repeat_harmony_correction`, `invivo_maps` | ON-Harmony (OpenNeuro ds004712) | `datasets/on_harmony/fetch.py`, `datasets/segmentation_priors/fetch.py`, `analyses/on_harmony/run.py` |
| `invivo_repeat_edden`, `invivo_repeat_edden_complex`, `invivo_maps` | EDDEN (OpenNeuro ds004666) | `datasets/edden/fetch.py`, `datasets/segmentation_priors/fetch.py`, `analyses/edden/nordic2.m` in MATLAB, `analyses/edden/run.py` |

All paths are under `scripts/`. Each script's docstring says what it reads and writes.

## Licence

MIT. NORDIC_Raw is a submodule under its own licence.
