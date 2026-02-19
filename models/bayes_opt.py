import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class BayesianOptimiser:
    """
    Multi-objective Bayesian Optimisation for membrane composition selection.

    The goal: given two trained GPs (one per molecule), find the next composition
    to test in the lab that will most efficiently guide us toward the true minimum
    permeability — balancing exploitation (test near the current best guess) and
    exploration (test where we are most uncertain).

    Acquisition function: Expected Improvement (EI)
    ------------------------------------------------
    EI(x) = E[ max(f_best - f(x), 0) ]

    For a GP with mean mu(x) and std sigma(x), this has a closed form:
        z    = (f_best - mu(x)) / sigma(x)
        EI   = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)

    where Phi is the normal CDF and phi is the normal PDF.

    We use a *combined* EI across both molecules: the acquisition function is
    the sum of normalised EI for Phenol + M-Cresol, so the next point will be
    informative for both simultaneously.
    """

    def __init__(self, gp_ph, gp_mc, y_ph, y_mc, xi=0.01):
        """
        Parameters
        ----------
        gp_ph, gp_mc : fitted GaussianProcessModel instances
        y_ph, y_mc   : training log-permeability arrays (used to find f_best)
        xi           : exploration parameter (default 0.01).
                       Larger xi = more exploration of uncertain regions.
                       Smaller xi = more exploitation of the current best area.
        """
        self.gp_ph = gp_ph
        self.gp_mc = gp_mc
        # f_best = best (lowest) log-permeability seen so far
        self.f_best_ph = y_ph.min()
        self.f_best_mc = y_mc.min()
        # ranges for normalisation
        self.range_ph = y_ph.max() - y_ph.min() + 1e-12
        self.range_mc = y_mc.max() - y_mc.min() + 1e-12
        self.xi = xi

    def _ei_single(self, x_2d, gp, f_best):
        """Expected Improvement for one GP at a batch of points."""
        mu, sigma = gp.model.predict(x_2d, return_std=True)
        sigma = np.maximum(sigma, 1e-9)      # avoid division by zero
        z  = (f_best - mu - self.xi) / sigma
        ei = (f_best - mu - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)             # EI is always non-negative
        return ei, mu, sigma

    def acquisition(self, x):
        """
        Combined (normalised) EI for both molecules at composition x.
        Returns a scalar — higher is better (more worth testing).
        """
        x_2d = np.atleast_2d(x)
        ei_ph, _, _ = self._ei_single(x_2d, self.gp_ph, self.f_best_ph)
        ei_mc, _, _ = self._ei_single(x_2d, self.gp_mc, self.f_best_mc)
        # normalise so neither molecule dominates
        return float(ei_ph[0] / self.range_ph + ei_mc[0] / self.range_mc)

    def suggest_next(self, n_restarts=40):
        """
        Maximise the acquisition function over the 3-simplex to find the
        single best composition to test next.

        Returns
        -------
        dict with keys:
            x_opt       : np.array shape (4,), the suggested composition fractions
            acq_value   : the acquisition function value at x_opt
            mu_ph, mu_mc: predicted log-permeability (mean)
            std_ph, std_mc: GP uncertainty (1 sigma, log-units)
            ci95_ph, ci95_mc: (lo, hi) 95% confidence intervals in cm/s
        """
        constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * 4

        # We minimise the *negative* EI (scipy minimises)
        def neg_acq(x):
            return -self.acquisition(x)

        # Structured + random starts to cover the full simplex
        fixed = [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5],
            [0, 0.5, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5],
            [0.25, 0.25, 0.25, 0.25],
            [0, 0.3, 0.7, 0], [0, 0.4, 0.6, 0], [0, 0.6, 0.4, 0],
            [0.1, 0.2, 0.7, 0], [0.2, 0.3, 0.5, 0],
        ]
        np.random.seed(42)
        random_starts = [np.random.dirichlet(np.ones(4)) for _ in range(n_restarts)]
        all_starts = [np.array(s, dtype=float) / np.sum(s) for s in fixed] + random_starts

        best_val = -np.inf
        best_x = None
        for x0 in all_starts:
            res = minimize(
                neg_acq, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x

        best_x = np.maximum(best_x, 0)
        best_x /= best_x.sum()

        x_2d = best_x.reshape(1, -1)
        _, mu_ph, std_ph = self._ei_single(x_2d, self.gp_ph, self.f_best_ph)
        _, mu_mc, std_mc = self._ei_single(x_2d, self.gp_mc, self.f_best_mc)

        return {
            "x_opt":      best_x,
            "acq_value":  best_val,
            "mu_ph":      float(mu_ph[0]),
            "mu_mc":      float(mu_mc[0]),
            "std_ph":     float(std_ph[0]),
            "std_mc":     float(std_mc[0]),
            "ci95_ph":    (10 ** (mu_ph[0] - 2 * std_ph[0]),
                           10 ** (mu_ph[0] + 2 * std_ph[0])),
            "ci95_mc":    (10 ** (mu_mc[0] - 2 * std_mc[0]),
                           10 ** (mu_mc[0] + 2 * std_mc[0])),
            "pred_ph":    10 ** float(mu_ph[0]),
            "pred_mc":    10 ** float(mu_mc[0]),
        }

    def acquisition_surface(self, n=60):
        """
        Evaluate the acquisition function on a 2-D grid (S2 vs C1, S1=C2=0)
        for visualisation. Returns (S2_grid, C1_grid, EI_grid).
        """
        s2 = np.linspace(0, 1, n)
        c1 = np.linspace(0, 1, n)
        SS, CC = np.meshgrid(s2, c1)
        mask = (SS + CC) <= 1.0
        X_grid = np.column_stack([
            np.zeros(SS.size), SS.ravel(), CC.ravel(), np.zeros(SS.size)
        ])
        ei_ph, _, _ = self._ei_single(X_grid, self.gp_ph, self.f_best_ph)
        ei_mc, _, _ = self._ei_single(X_grid, self.gp_mc, self.f_best_mc)
        ei_combined = (ei_ph / self.range_ph + ei_mc / self.range_mc).reshape(SS.shape)
        ei_combined[~mask] = np.nan
        return SS * 100, CC * 100, ei_combined
