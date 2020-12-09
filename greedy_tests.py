#!/usr/bin/env python3
"""greedy_tests.py: performance test driver for "greedy" TSP algorithm"""
__author__ = "Jessica Lynch"

import tsp
import sys
from os.path import dirname, join
from time import perf_counter_ns, time_ns

# # Route print statements to file
# current_dir = dirname(__file__)
# file_name = "greedy_performance_" + str(time_ns()) + ".txt"
# file_path = join(current_dir, file_name)
# sys.stdout = open(file_path, "w")


def main():
    # Assign timer function to variable
    clock = perf_counter_ns

    # Determine max run time for each algorithm
    one_second = 1000000000  # 1 second in nanoseconds
    # MAX_RUN_TIME = one_second * 60 * 10
    MAX_RUN_TIME = one_second  # small value for testing
    MAX_COST = 100

    # Init table variables
    t_str = "Time"
    dr_str = "DR"
    expected_str = "expected"
    na_str = "--"
    cols_per_func = 3  # Update if additional data columns are added
    col_width_full = 39
    col_width_med = col_width_full // cols_per_func
    col_width_small = 10

    TEST_FUNC = tsp.tsp_greedy

    # Print function name (header top row)
    print(f"{'':>{col_width_small}}", end="")
    print(f"{TEST_FUNC.__name__:>{col_width_full}}", end="")
    print()

    # Print data columns (header second row)
    print(f"{'N':>{col_width_small}}", end="")
    print(f"{t_str:>{col_width_med}}", end="")
    print(f"{dr_str:>{col_width_med}}", end="")
    print(f"{expected_str:>{col_width_med}}", end="")
    print("\n")

    # Init flag to track when all functions are complete
    timed_out = False

    # Test alg with increasing N x N matrices
    N = 2
    while N < sys.maxsize and not timed_out:
        # Print current N value
        print(f"{N:>{col_width_small}}", end="")

        # Generate a N x N matrix
        matrix, min_path, min_cost = tsp.generate_circular_cost_matrix(N, MAX_COST)

        # Start clock
        t0 = clock()

        # Run algorithm
        TEST_FUNC(matrix)

        # Stop clock and calculate time
        t1 = clock() - t0

        # Convert nanoseconds to milliseconds or seconds if possible
        # and print time taken for current function
        tsp.print_time(t1, col_width_med)

        # Print expected and actual doubling ratio
        if N >= 4:
            expected_dr = ((N - 1) * (N / 2) + 1) / ((N//2 - 1) * (N//2 / 2) + 1)
            doubling_ratio = t1 / previous_result
            print(f"{doubling_ratio:>{col_width_med}.2f}", end="")
            print(f"{expected_dr:>{col_width_med}.2f}", end="")
        else:
            print(f"{'':>{col_width_med * 2}}", end="")
        print()

        # Store time result from current run
        previous_result = t1

        # Quit once the function takes too long
        if t1 > MAX_RUN_TIME:
            timed_out = True
            print("Function timed out")

        # Double the matrix size
        N *= 2


if __name__ == "__main__":
    main()
