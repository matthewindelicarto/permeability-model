# TPU Permeability Model

Predict and optimize permeability of TPU membranes using experimental Franz Cell data.

## Live Demo

**[Launch App](https://permeability-model.streamlit.app)** - Free web app, no installation required.

## Features

- **TPU Membranes**: Browse experimental Franz Cell permeability data for Sparsa and Carbosil formulations
- **Permeability Calculator**: Predict permeability for any composition using three models
- **Optimal Composition**: Find the composition that maximizes or minimizes permeability

## Models

### Regression
Polynomial ridge regression (degree 2). The four membrane composition fractions are expanded into all pairwise interaction terms (e.g. Sparsa1 × Carbosil1), giving the model the ability to capture non-linear blending effects. Ridge regularization (L2 penalty) prevents overfitting on the small dataset by shrinking large coefficients toward zero. Predictions are made in log-permeability space and converted back to cm/s.

### Neural Network
A small feedforward network with one hidden layer (8 neurons) trained entirely in NumPy. Inputs and outputs are normalized before training. The hidden layer uses sigmoid activations; the output layer is linear. Weights are updated with gradient descent (learning rate 0.01, 5000 epochs) using mean-squared error loss. The network learns non-linear composition–permeability relationships directly from the data.

### RBF Interpolation
Gaussian radial basis function interpolation. Each training point acts as a basis function centered at that composition. The shape parameter (epsilon) is set automatically as the inverse of the median pairwise distance between training points. Interpolation weights are found by solving a linear system. A small regularization term (1e-6) is added to the kernel matrix for numerical stability. RBF interpolation passes through all training points exactly (modulo regularization), making it more sensitive to individual data points than the other two models.

## Permeants

- Phenol
- M-Cresol

## License

MIT License
