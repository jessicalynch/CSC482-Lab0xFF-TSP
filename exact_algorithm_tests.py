#!/usr/bin/env python3

"""exact_algorithm_tests.py: performance test driver for exact TSP algorithms"""
__author__ = "Jessica Lynch"

import tsp
import sys
from os.path import dirname, join
import math
from time import perf_counter_ns, time_ns

# # Route print statements to file
# current_dir = dirname(__file__)
# file_name = "performance_" + str(time_ns()) + ".txt"
# file_path = join(current_dir, file_name)
# sys.stdout = open(file_path, "w")


def main():
    # Assign timer function to variable
    clock = perf_counter_ns

    # Determine max run time for each algorithm
    one_second = 1000000000  # 1 second in nanoseconds
    MAX_RUN_TIME = one_second * 60 * 30
    # MAX_RUN_TIME = one_second  # small value for testing
    MAX_NUMBER = 100
    MAX_COST = 100

    # Build list with functions to test
    tsp_funcs = [tsp.tsp_brute_iterative, tsp.tsp_brute_recur, tsp.tsp_dynamic]
    num_funcs = len(tsp_funcs)
    timed_out_funcs = [False] * num_funcs

    # Verify functions are accurate
    verify = tsp.verify_exact_algorithms("circular", MAX_COST, False)
    if verify:
        print("<Functions verified>\n")
    else:
        print("<Inconsistent function results>")
        exit()

    # Init table variables
    t_str = "Time"
    dr_str = "DR"
    expected_str = "expected"
    na_str = "--"
    cols_per_func = 3  # Update if additional data columns are added
    col_width_full = 39
    col_width_med = col_width_full // cols_per_func
    col_width_small = 10

    # Print function names (header top row)
    print(f"{'':>{col_width_small}}", end="")
    for i in range(num_funcs):
        print(f"{tsp_funcs[i].__name__:>{col_width_full}}", end="")
    print()

    # Print data columns (header second row)
    print(f"{'N':>{col_width_small}}", end="")
    for i in range(num_funcs):
        print(f"{t_str:>{col_width_med}}", end="")
        print(f"{dr_str:>{col_width_med}}", end="")
        print(f"{expected_str:>{col_width_med}}", end="")
    print("\n")

    # Init flag to track when all functions are complete
    timed_out = False

    # Init list of dictionaries to store timing results
    results = []
    for _ in range(num_funcs):
        results.append({})

    # Init list to store doubling ratios
    doubling_ratio = [-1] * num_funcs

    # Start testing algorithms with increasing x values
    for N in range(2, MAX_NUMBER + 1):

        # Exit if all functions are complete
        if timed_out:
            print("All functions timed out")
            break

        # Otherwise assume complete and keep testing
        timed_out = True

        # Print current N value
        print(f"{N:>{col_width_small}}", end="")

        # Generate a N x N matrix
        matrix = tsp.generate_euclidean_cost_matrix(N, MAX_COST)

        # Loop through each function and test with current x value
        for i in range(num_funcs):

            # Skip to next function if current is timed out
            if timed_out_funcs[i]:
                # Print filler values to maintain table structure
                for _ in range(cols_per_func):
                    print(f"{na_str:>{col_width_med}}", end="")
                if i == (num_funcs - 1):
                    print()
                continue

            try:
                # Start clock
                t0 = clock()

                # Run algorithm
                tsp_funcs[i](matrix)

                # Stop clock and calculate time
                t1 = clock() - t0

                # Convert nanoseconds to milliseconds or seconds if possible
                # and print time taken for current function
                tsp.print_time(t1, col_width_med)

                # Calculate doubling ratio
                if N % 2 == 0 and N >= 4:
                    expected_dr = get_expected_dr(tsp_funcs[i].__name__, N)
                    doubling_ratio[i] = t1 / results[i].pop(N // 2)
                    print(f"{doubling_ratio[i]:>{col_width_med}.2f}", end="")
                    print(f"{expected_dr:>{col_width_med}.2f}", end="")
                else:
                    print(f"{'':>{col_width_med * 2}}", end="")

                # Store time result from current run
                results[i][N] = t1

                # Update flags
                if t1 < MAX_RUN_TIME:
                    # At least one function is still going
                    timed_out = False
                else:
                    # Current function is timed out
                    timed_out_funcs[i] = True
                    results[i].clear()

            except RecursionError:
                # Current function has reached maximum recursion depth,
                # set flag and clear timing results dictionary
                timed_out_funcs[i] = True
                results[i].clear()
            except ValueError as e:
                # Something went wrong
                print(e)
                exit()

        print()


def get_expected_dr(func_name, N):
    """Returns expected doubling ratio for each function"""
    if func_name == "tsp_dynamic":  # Exponential time
        return (2**N * N**2) / (2**(N//2) * (N//2)**2)
    elif func_name in ("tsp_brute_iterative", "tsp_brute_recur"):  # Factorial time
        return math.factorial(N - 1) / math.factorial((N//2) - 1)
    else:
        return -1


if __name__ == "__main__":
    main()
