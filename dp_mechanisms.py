from typing import Any, Dict
import heapq
import numpy as np


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

