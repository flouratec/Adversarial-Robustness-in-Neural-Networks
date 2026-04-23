# Adversarial Robustness in Neural Networks

A GitHub-ready academic project that evaluates how a neural network behaves under adversarial attack and how adversarial training improves robustness.

## Project summary

This repository presents a clear research story:

1. Train a clean baseline model.
2. Attack it with FGSM and PGD.
3. Analyze the failure.
4. Apply adversarial training.
5. Publish the results through a simple interactive dashboard.

## Headline results

| Model | Clean | FGSM | PGD |
|---|---:|---:|---:|
| Baseline MLP | 96.67% | 21.11% | 0.28% |
| Adversarially trained MLP | 96.39% | 58.89% | 30.28% |

## What is included

- `manuscript/` - master’s-style rewritten research manuscript in Markdown and DOCX.
- `docs/` - GitHub Pages dashboard.
- `src/` - experiment script.
- `data/` - saved metrics.
- `slides/` - presentation files.
- `guides/` - practical setup and publishing instructions.
- `legacy_material/` - original paper sources kept for reference.

## Launch the dashboard on GitHub Pages

1. Create a new GitHub repository.
2. Upload all repository files.
3. Open **Settings > Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select branch **main** and folder **/docs**.
6. Click **Save**.
7. Wait for the deployment link.
8. Open your live dashboard.

## Local editing workflow

- Replace figures inside `docs/assets/` if you generate new experiments.
- Update headline metrics inside `docs/data/dashboard-data.js`.
- Edit the manuscript in `manuscript/adversarial_robustness_masters_style.md` and regenerate the DOCX if needed.

## Recommended repository name

`adversarial-robustness-thesis-project`

## Quick start for researchers

```bash
git clone <your-repo-url>
cd adversarial-robustness-thesis-project
python src/run_experiments.py
```

## Suggested next improvements

- Replace the local digits proxy with MNIST.
- Add a CNN baseline.
- Add TRADES as a second defense.
- Extend the dashboard with new metrics and figures.
