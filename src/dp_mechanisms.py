from typing import Any, Dict
import math
import numpy as np


# ============================================================
# Noise Utilities
# ============================================================

def gumbel_noise(scale: float) -> float:

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
    eps_basic = eps_target / k

    log_term = math.log(1.0 / delta_target)
    term1 = (2 * log_term) / k
    eps_adv = math.sqrt(term1 + (eps_target / k)) - math.sqrt(term1)

    if decomposable:
        eps_gupta = math.log(
            1 + eps_target / (4 + log_term)
        )
    else:
        eps_gupta = -1

    return max(eps_basic, eps_adv, eps_gupta)