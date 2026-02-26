# TPU Permeability Model

Predict and optimize permeability of TPU membranes using experimental Franz Cell data.

## Live Demo

**[Launch App](https://permeability-model.streamlit.app)** — no installation required.

## Features

- **Membrane Data**: Browse experimental Franz Cell permeability data for Sparsa and Carbosil TPU formulations
- **Permeability Calculator**: Predict permeability for any composition using trained models
- **Optimal Composition**: Find the composition that maximizes or minimizes permeability
- **Bayesian Optimization**: Suggest the next composition to test based on model uncertainty

## Models

### Gaussian Process Regression (GPR)

A Gaussian Process with a Matérn 5/2 kernel (ν = 2.5), scaled by a constant amplitude term. The Matérn kernel is preferred over RBF for scientific data because it enforces finite differentiability, which is more physically realistic than the infinitely smooth RBF assumption.

**Noise handling:** The `alpha` parameter fixes the noise variance on the diagonal of the kernel matrix rather than using a free `WhiteKernel`. Noise is estimated directly from experimental replicates as α = mean(Δ²/2) across repeated membrane tests, where Δ is the difference in log₁₀(P) between two runs of the same composition. This pins the noise to a physically meaningful value and prevents the optimizer from absorbing structured variation into spurious noise terms. When no replicates are available, alpha defaults to near-zero (essentially noiseless).

Predictions are made in log₁₀(P) space. The GP is fit with 10 random restarts of the kernel hyperparameter optimizer to avoid local optima on the small dataset. The posterior standard deviation provides calibrated uncertainty estimates, which are used directly by the Bayesian optimizer to balance exploration and exploitation.

### Neural Network (Ensemble)

An ensemble of 7 small feedforward networks, each with one hidden layer (6 neurons) trained entirely in NumPy. Ensemble averaging over multiple random seeds substantially reduces seed-dependent variance, which is a major source of instability when training on small datasets.

**Architecture:** Inputs and outputs are normalized before training (z-score). The hidden layer uses sigmoid activations; the output layer is linear. Weights are initialized with He initialization.

**Optimizer:** Adam (lr = 1e-3, β₁ = 0.9, β₂ = 0.999) with L2 weight decay (λ = 1e-3). Adam converges reliably on the tight output ranges and sparse gradients typical of this dataset, outperforming plain SGD. Each network trains for 8,000 epochs. The final prediction is the mean across all 7 ensemble members.

## Permeants

- Phenol
- M-Cresol

## License

MIT License
