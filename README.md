# Speculative Decoding for Robot Diffusion Policies

**Read the paper: [A draft model for a robot diffusion policy, revisited (PDF)](PAPER.pdf)**

> 1st place (solo). PyData × Cursor Boston Hackathon at Moderna HQ, May 13 2026.

I tried porting the draft/verifier idea behind speculative decoding (Medusa, EAGLE) to `lerobot/diffusion_pusht`. A tiny MLP proposes actions and the 262.7M-parameter diffusion policy checks them. The hackathon version claimed 2.02x, and the May writeup 1.6x, at a 10 Hz control rate. This repo now holds a second version that re-runs the idea. The draft never sees the episodes it's scored on, every strategy is scored on the same frames, and everything also runs closed-loop in the PushT simulator. The verifier checkpoint was trained on every pusht episode, so only the draft is held out.

What the second version found:

- **The original rule saves nothing.** It needs the verifier's action to decide whether to skip the verifier, so it pays for every call. It worked as a model cascade.
- **Gates that decide without the verifier do skip calls,** but they more than quadruple the action error on the way to 2x.
- **Fewer denoising steps get the speedup with no draft.** DDPM or DDIM at 10 or 5 steps instead of 20 runs 2x to 3.9x faster and stays within noise of 20 steps in open loop. In the simulator they succeed about as often as 20 steps (48% to 64% over 50 rollouts, within noise), while the original rule cuts success from 48% to 22%.
- **Executing more actions per plan costs accuracy.** 15 actions instead of 8 adds about 27% error after the first plan of each episode.
- **The speculative-decoding idea that does carry over works inside the denoising chain,** and De Bortoli et al. (2025) already did it on PushT.

## What's here

| file | what it does |
|---|---|
| `PAPER.pdf` | the writeup |
| `spec_eval.py` | open-loop evaluation: held-out episodes for the draft, chunked serving loop, gates charged for every verifier call, BCa episode-bootstrap intervals |
| `closed_loop.py` | runs every strategy in the gym-pusht simulator from the same 50 starting positions. The paper's run was stopped after 11 of 14 strategies |
| `stats.py` | BCa, Wilson, exact McNemar and Newcombe intervals |
| `make_figures.py` | rebuilds every figure in the paper from the two results files |
| `test_*.py` | tests for the serving loop, gates, paid-access guard, lockstep rollouts and statistics (no models needed) |
| `results/` | `v2_openloop.json` and `v2_closed_partial.json`, the numbers behind the paper, and `environment.txt`, the exact packages they came from |
| `compat.py` | two import patches lerobot 0.4 needs on Python 3.14 |
| `notebook.py` | the original hackathon notebook, kept as submitted. Its speedup, 10 Hz, "Pareto-improving" and "first port" claims don't hold (paper, section 1) |
| `async_speedup.py` | a June follow-up to the notebook. It picks which calls to skip from a first pass that already ran the verifier, and it evaluates on training episodes, so its speedup is an upper bound |
| `media/` | the original notebook's figures, with its old numbers |

## Running it

```bash
uv sync --python 3.12 --extra dev --extra figures
uv run pytest -q
uv run spec_eval.py --out results/repro_openloop.json
uv run closed_loop.py --from results/repro_openloop.json --out results/repro_closed.json
uv run plotly_get_chrome -y   # kaleido needs Chrome to export figures
uv run make_figures.py --open results/repro_openloop.json --closed results/repro_closed.json
```

The defaults match the paper: 4 diffusion samples per boundary open-loop, 50 rollouts per strategy closed-loop. The commands write to new files so the committed results stay put, and figures land in `blog/figures_v2/`. Only uv works; there's no pip-installable package.

On Linux you also need a few system packages: `ffmpeg` (the dataset's videos are AV1, decoded through torchcodec), `build-essential` and `linux-libc-dev` (one of lerobot's dependencies builds from source), and `libgl1 libglib2.0-0` (OpenCV, used by the simulator). The locked torch build targets CUDA 12.8 and needs a recent NVIDIA driver.

`spec_eval.py --max-episodes 2` is a quick smoke test. `spec_eval.py --from-cache results/v2_openloop.cache.pkl` recomputes every table from the plan cache a full run writes, with no diffusion. Both scripts pick CUDA, then Apple's MPS, then CPU. On an M-series laptop the open-loop run takes about 70 minutes and the closed-loop run about an hour. The first run downloads `lerobot/pusht` and the `lerobot/diffusion_pusht` checkpoint from Hugging Face.

The paper's results came from Python 3.14.2 with torch 2.10.0 and lerobot 0.4.4 on Apple MPS, with the checkpoint at Hugging Face revision `84a7c23`. `results/environment.txt` lists every package from that run. `pyproject.toml` and `uv.lock` pin the same top-level packages for Python 3.12, where the `compat.py` patches don't run; some transitive versions differ. Other hardware won't reproduce the numbers exactly: the DDPM step noise comes from the device's random generator, so CUDA draws different noise than MPS, and timing is machine-specific. On the same machine with the same settings, the error and call counts repeat exactly.

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Cai et al. *Medusa.* 2024. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Li et al. *EAGLE.* 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- Kwon et al. *vLLM / PagedAttention.* SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Chi et al. *Diffusion Policy.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Prasad et al. *Consistency Policy.* RSS 2024. [arXiv:2405.07503](https://arxiv.org/abs/2405.07503)
- Wang et al. *One-Step Diffusion Policy.* 2024. [arXiv:2410.21257](https://arxiv.org/abs/2410.21257)

## Hackathon

The notebook was the artifact submitted to the **PyData x Cursor Boston Hackathon** at Moderna HQ on May 13, 2026, and won **1st place (solo) among 13 winner-eligible submissions.** Final judging board: [cursorboston.com/events/cursor-boston-pydata-2026](https://www.cursorboston.com/events/cursor-boston-pydata-2026). Original submission PR: [rogerSuperBuilderAlpha/cursor-boston#855](https://github.com/rogerSuperBuilderAlpha/cursor-boston/pull/855). The first version of the paper is in the repo history.
