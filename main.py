import tsp
import pprint
import sys
from os.path import dirname, join
from time import perf_counter_ns, time_ns

# # Route print statements to file
# current_dir = dirname(__file__)
# file_name = "verification_" + str(time_ns()) + ".txt"
# file_path = join(current_dir, file_name)
# sys.stdout = open(file_path, "w")


def main():

    n = 9
    max_cost = r = 99
    m = tsp.generate_random_cost_matrix(n, max_cost)
    # m = tsp.generate_euclidean_cost_matrix(n, max_cost)
    # m, min_cost = tsp.generate_circular_cost_matrix(n, r)
    # tsp.write_matrix_to_file(m, "matrix.txt")

    # m = tsp.read_matrix_from_file("matrix.txt")
    print(tsp.greedy(m))
    print(tsp.brute_iterative(m))
    print(tsp.brute_recur(m))
    print(tsp.dynamic_programming(m))

    # print("Verified:", tsp.verify_exact_algorithms())


if __name__ == "__main__":
    main()
