from typing import Any, Dict
import math
import numpy as np


# ============================================================
# Noise Utilities
# ============================================================

def gumbel_noise(scale: float) -> float:
    """
    Draw a single Gumbel noise sample with the given scale.
    """
    return np.random.gumbel(scale=scale)


# ============================================================
# Exponential Mechanism (via Gumbel trick)
# ============================================================

def exp_mech(
    candidates_with_scores: Dict[Any, float],
    eps: float,
    sensitivity: float,
    private: bool = True,
):
    """
    Selects an element using the Exponential Mechanism.

    If private=False, returns the exact maximizer.
    """
    if private:
        noise_scale = 2 * sensitivity / eps

        noisy_scores = {
            candidate: score + gumbel_noise(noise_scale)
            for candidate, score in candidates_with_scores.items()
        }
    else:
        noisy_scores = candidates_with_scores

    return max(noisy_scores, key=noisy_scores.get)


# ============================================================
# Privacy Composition Utilities
# ============================================================

def get_best_eps_0(
    eps_target: float,
    delta_target: float,
    k: int,
    decomposable: bool = True,
) -> float:
    """
    Computes candidate ε₀ values under different composition bounds
    and returns the largest valid one.

    Includes:
        1. Basic Composition
        2. Advanced Composition
        3. Gupta Bound (if decomposable=True)
    """

    # 1. Basic Composition
    eps_basic = eps_target / k

    # 2. Advanced Composition
    log_term = math.log(1.0 / delta_target)
    term1 = (2 * log_term) / k
    eps_adv = math.sqrt(term1 + (eps_target / k)) - math.sqrt(term1)

    # 3. Gupta Bound (for decomposable objectives)
    if decomposable:
        eps_gupta = math.log(
            1 + eps_target / (4 + log_term)
        )
    else:
        eps_gupta = -1

    return max(eps_basic, eps_adv, eps_gupta)