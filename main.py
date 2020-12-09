#!/usr/bin/env python3
"""main.py: demo driver for TSP module functions"""
__author__ = "Jessica Lynch"

import tsp


def main():
    # Verify algorithms with each graph type
    MAX_COST = 100
    for graph_type in ["circular", "euclidean", "random"]:
        verified = tsp.verify_exact_algorithms(graph_type, MAX_COST, print_results=True)
        if not verified:
            print("Something went wrong...")
            exit()
    print()

    # Write and read matrix from file
    NUM_V = 5
    MAX_COST = 9
    m = tsp.generate_random_cost_matrix(NUM_V, MAX_COST)
    tsp.write_matrix_to_file(m, "matrix.txt")
    m = tsp.read_matrix_from_file("matrix.txt")
    tsp.matrix_print(m)


if __name__ == "__main__":
    main()
