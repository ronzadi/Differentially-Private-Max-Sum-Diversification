from typing import Any, Dict
import numpy as np
import math

def gumbel_noise(scale):
    return np.random.gumbel(scale=scale)


def exp_mech(candidates_with_scores: Dict[Any, float], eps, sensitivity, private=True):

    if private:
        gumbel_scores = {
            candidate: score + gumbel_noise(2 * sensitivity / eps)
            for candidate, score in candidates_with_scores.items()
        }
    else:
        gumbel_scores = candidates_with_scores

    arg_max = max(gumbel_scores, key=gumbel_scores.get)

    return arg_max


def get_best_eps_0(eps_target, delta_target, k, decomposable=True):
    """
    Calculates Basic, Advanced, and Gupta bounds and selects the maximum epsilon_0.
    """
    # 1. Basic Composition
    eps_basic = eps_target / k

    # 2. Advanced Composition
    term1 = (2 * math.log(1.0 / delta_target)) / k
    eps_adv = math.sqrt(term1 + (eps_target / k)) - math.sqrt(term1)

    # 3. Gupta Bound (For decomposable objectives)
    eps_gupta = math.log(1 + eps_target/(3 + math.log(1.0 / delta_target))) if decomposable else -1

    return max(eps_basic, eps_adv, eps_gupta)