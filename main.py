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

    n = 8
    max_cost = r = 100
    # m = tsp.generate_random_cost_matrix(n, max_cost)
    m = tsp.generate_euclidean_cost_matrix(n, max_cost)
    # m, min_cost = tsp.generate_circular_cost_matrix(n, r)
    # tsp.write_matrix_to_file(m, "matrix.txt")

    tsp.matrix_print(m)
    # m = tsp.read_matrix_from_file("matrix.txt")
    # print(tsp.greedy(m))
    # print(tsp.brute_iterative(m))
    # print(tsp.dynamic_programming(m))

    brute_result = tsp.brute_recur(m)
    brute_cost = round(brute_result[1], 6)

    # How many times did the ant colony find the optimal solution?
    trials = 50
    num_ants = n * 10
    num_steps = 100  # quit if min_path hasn't changed in this many steps
    correct = 0
    total_cost = 0
    for x in range(trials):
        ant_result = tsp.ant_colony(m, num_ants, num_steps)
        ant_cost = round(ant_result[1], 6)
        total_cost += ant_cost
        if ant_cost == brute_cost:
            correct += 1

    percent_correct = round((correct/trials) * 100, 2)
    print("Min cost:", brute_cost)
    print("Avg. ant colony cost:", round(total_cost / trials, 6))
    print(f"After {trials} trials:")
    print(f"Ant colony found the optimal path {percent_correct}% of the time")

    # print("Verified:", tsp.verify_exact_algorithms())


if __name__ == "__main__":
    main()
