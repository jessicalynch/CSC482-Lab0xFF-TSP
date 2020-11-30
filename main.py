import tsp
import pprint


def main():

    n = 10
    max_cost = r = 9
    m = tsp.generate_random_cost_matrix(n, max_cost)
    # m = tsp.generate_euclidean_cost_matrix(n, max_cost)
    # m, min_distance = tsp.generate_circular_cost_matrix(n, r)
    tsp.write_matrix_to_file(m, "matrix.txt")

    m = tsp.read_matrix_from_file("matrix.txt")
    tsp.matrix_print(m)
    print(tsp.greedy(m))
    print(tsp.brute(m))

    g = tsp.matrix_to_graph(m)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(g)

    print("Verified:", tsp.verify_optimal_solutions())


if __name__ == "__main__":
    main()
