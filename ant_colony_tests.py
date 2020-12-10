#!/usr/bin/env python3
"""ant_colony_tests.py: Solution quality test driver for ant colony TSP algorithm"""
__author__ = "Jessica Lynch"

import tsp
from time import perf_counter_ns


def main():
    # Assign timer function to variable
    clock = perf_counter_ns

    # Determine max run time for each algorithm
    MAX_MATRIX_SIZE = 50
    MAX_COST = 100
    NUM_TRIALS_PER_N = 50

    # Init table variables
    params = ["Ants", "Max steps", "Phero", "Decay", "% Exact", "Min cost", "Avg. cost"]
    cols_per_func = 7  # Update if additional data columns are added
    col_width_full = 13 * cols_per_func
    col_width_med = col_width_full // cols_per_func
    col_width_small = 10

    # Print function name and test info
    print(f"{'':>{col_width_small}}", end="")
    print(f"{'Ant Colony':>{col_width_full}}", end="")
    print("Trials per N x N matrix:", NUM_TRIALS_PER_N)
    print("Graph type: circular Euclidean")
    print("Graph radius:", MAX_COST)

    # Print data columns (header second row)
    print(f"{'N':>{col_width_small}}", end="")
    for param in params:
        print(f"{param:>{col_width_med}}", end="")
    print("\n")

    # Test algorithm with N x N matrices
    # for N in range(2, MAX_MATRIX_SIZE + 1):
    N = 9
    while True:
        # Print current N value
        print(f"{N:>{col_width_small}}", end="")

        # Generate N x N matrix
        matrix, min_path, min_cost = tsp.generate_circular_cost_matrix(N, MAX_COST)

        phero_factor = 1
        decay_factor = 1
        ants = 100
        steps = 20

        correct = 0
        total_cost = 0
        for _ in range(NUM_TRIALS_PER_N):
            path, cost = tsp.tsp_ant_colony(matrix, ants, phero_factor, decay_factor, steps)
            total_cost += cost
            if path == min_path or path == list(reversed(min_path)):
                correct += 1
        percent_correct = (correct / NUM_TRIALS_PER_N) * 100
        average_cost = total_cost / NUM_TRIALS_PER_N

        print(f"{ants:>{col_width_med}}", end="")
        print(f"{steps:>{col_width_med}}", end="")
        print(f"{phero_factor:>{col_width_med}}", end="")
        print(f"{decay_factor:>{col_width_med}}", end="")
        print(f"{round(percent_correct, 1):>{col_width_med}}", end="")
        print(f"{round(min_cost, 2):>{col_width_med}}", end="")
        print(f"{round(average_cost, 2):>{col_width_med}}", end="")
        print()


if __name__ == "__main__":
    main()
