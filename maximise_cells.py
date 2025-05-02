from scipy.optimize import minimize
import numpy as np
from typing import List, Tuple


def neg_likelihood(params, a, b, c, d):
    p, q = params
    if not (0 < p < 1 and 0 < q < 1):
        return np.inf  # stay in domain
    return -((p ** a) * ((1 - p) ** b) * (q ** c) * ((1 - q) ** d))


def find_max(params):
    a, b, c, d = params
    initial_guess = np.array([0.5, 0.5])
    bounds = [(1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5)]
    result = minimize(neg_likelihood, initial_guess, args=(a, b, c, d), bounds=bounds)
    return result.x[0], result.x[1], -result.fun


def process_combinations(combinations: List[List[int]]) -> List[Tuple]:
    """Process combinations and return sorted results as (combination, p, q, max_value)."""
    results = []
    for combo in combinations:
        p_opt, q_opt, max_value = find_max(combo)
        results.append((combo, p_opt, q_opt, max_value))

    # Sort by max_value (index 3) in descending order
    return sorted(results, key=lambda x: x[3], reverse=True)


def display_results(name: str, results: List[Tuple]):
    print(f"Results for {name}:")
    for combo, p, q, max_val in results:
        print(f"Combination {combo}: p={p:.4f}, q={q:.4f}, max_value={max_val:.6f}")


# Combinations
val_comb_c = [
    [2, 4, 0, 0], [3, 3, 0, 0], [3, 1, 2, 0], [1, 1, 1, 3],
    [2, 2, 2, 0], [1, 1, 2, 2], [2, 0, 3, 1], [0, 0, 2, 4]
]

val_comb_d = [
    [1, 5, 0, 0], [2, 2, 2, 0], [2, 2, 2, 0], [2, 0, 3, 1],
    [0, 4, 1, 1], [1, 1, 3, 1], [0, 2, 2, 2], [0, 0, 3, 3]
]


# Main execution
if __name__ == "__main__":
    results_c = process_combinations(val_comb_c)
    results_d = process_combinations(val_comb_d)

    display_results("character c", results_c)
    print('\n')
    display_results("character d", results_d)