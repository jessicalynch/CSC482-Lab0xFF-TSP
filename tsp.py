import random
import itertools
import sys
import math


def generate_random_cost_matrix(num_v, max_cost):
    """Generates cost matrix for random undirected and connected graph"""
    # Generate the first half
    matrix = [[(random.randint(1, max_cost) if i < j else 0) for i in range(num_v)] for j in range(num_v)]

    # Return a symmetrical matrix
    return [[(matrix[i][j] if i > j else matrix[j][i]) for i in range(num_v)] for j in range(num_v)]


def generate_euclidean_cost_matrix(num_v, max_cost):
    """Generates cost matrix for random undirected and connected Euclidean graph"""
    # Generate random (x,y) coordinates
    vertices = [(random.randint(1, max_cost), random.randint(1, max_cost)) for _ in range(num_v)]

    return coordinates_to_cost_matrix(vertices)


def generate_circular_cost_matrix(num_v, r):
    """Generates random cost matrix for evenly spaced points around a circle"""

    # Determine number of slices around the circle
    theta = (2 * math.pi) / num_v

    # Calculate (x,y) coordinate at each slice
    vertices = [(r * math.cos(i * theta), r * math.sin(i * theta))
                for i in range(num_v)]

    min_cost = 0
    for i in range(len(vertices)):
        u = vertices[i]
        v = vertices[(i + 1) % num_v]
        min_cost += get_distance(u, v)

    # Put vertices in random order
    random.shuffle(vertices)

    # Return matrix and minimum distance
    # return coordinates_to_cost_matrix(vertices), min_distance
    return coordinates_to_cost_matrix(vertices), min_cost


def coordinates_to_cost_matrix(vertices):
    """Converts (x,y) coordinates to cost matrix"""
    matrix = []
    for i, u in enumerate(vertices):
        row = []
        for j, v in enumerate(vertices):
            if i != j:
                row.append(get_distance(u, v))
            else:
                row.append(0.0)
        matrix.append(row)
    return matrix


def get_distance(u, v):
    """Calculates the distance between two 2D points"""
    x1 = u[0]
    x2 = v[0]
    y1 = u[1]
    y2 = v[1]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def greedy(m):
    """Finds a path by always choosing the shortest edge"""

    # Init list of vertices
    n = len(m)
    vertices = list(range(n))

    # Start at vertex 0
    u = vertices.pop(0)
    path = [u]
    cost = 0

    # Visit every other vertex
    for _ in range(n - 1):
        # Build list of unvisited vertices
        row = enumerate(m[u])
        unvisited = [x for x in row if x[0] in vertices]

        # Find vertex with the least edge cost
        min_edge_cost = sys.maxsize
        for vertex, edge_cost in unvisited:
            if edge_cost < min_edge_cost:
                min_edge_cost = edge_cost
                v = vertex

        # Travel to the vertex with the least cost
        u = v
        vertices.remove(u)
        path.append(u)
        cost += min_edge_cost

    # Travel back to vertex 0
    path.append(0)
    cost += m[u][0]

    return path, cost


def brute_iterative(m):
    """Finds optimal solution to the TSP by checking every possibility"""
    # Get permutations for an n-1 length list,
    # and sandwich each permutation with zeros
    # (we can always start and end at vertex 0)
    n = len(m)
    perms = itertools.permutations(range(n - 1))
    perms = [[0] + [int(x) + 1 for x in perm] + [0] for perm in perms]

    # Check the cost of each permutation
    min_cost = sys.maxsize
    for p in perms:
        cost = 0
        # Calculate cost from 0 -> ... -> 0
        for i in range(1, n + 1):
            curr_v = p[i]
            prev_v = p[i - 1]
            cost += m[prev_v][curr_v]
        # Update min cost and min perm if shortest seen
        if cost < min_cost:
            min_p = p
            min_cost = cost

    return min_p, min_cost


def brute_recur(m):
    """Finds optimal solution to the TSP by checking every possibility"""

    # Start and end with the same node to complete the circuit
    start_node = end_node = 0

    # Build list of unvisited nodes
    tour_nodes = [x for x in range(len(m)) if x != start_node]

    # Recursively find the minimum cost path between tour nodes
    path, cost = brute_recur_helper(m, start_node, end_node, tour_nodes)

    # Add starting node to beginning of the best path
    path.insert(0, start_node)

    return path, cost


def brute_recur_helper(m, start_node, end_node, tour_nodes):
    """Recursive helper function for brute_recur"""

    # Base case: only one node left to visit
    if len(tour_nodes) == 1:
        last_tour_node = tour_nodes[0]

        # Make a path from last tour node -> end node
        path = tour_nodes + [end_node]

        # Calculate the cost from k -> last tour node -> end node
        cost = m[start_node][last_tour_node] + m[last_tour_node][end_node]

        return path, cost
    else:
        min_cost = sys.maxsize
        min_path = []

        # Determine all possible permutations
        for k in tour_nodes:

            # Use each k in tour_nodes as a starting node
            # for the recursive calls
            unvisited = [x for x in tour_nodes if x != k]
            tmp_path, tmp_cost = brute_recur_helper(m, k, end_node, unvisited)

            # Take the branch with the least cost
            best_cost_from_k = m[start_node][k] + tmp_cost
            if best_cost_from_k < min_cost:
                min_cost = best_cost_from_k
                min_path = [k] + tmp_path

        return min_path, min_cost


def dynamic_programming(m):
    """brute_recur with caching"""
    # Start and end with the same node to complete the circuit
    start_node = end_node = 0

    # Build list of unvisited nodes
    tour_nodes = [x for x in range(len(m)) if x != start_node]

    # Init cache table for all possibilities
    cache = []
    n = len(m)
    power_set_length = 2 ** n
    for i in range(power_set_length):
        cache.append([-1] * power_set_length)

    # Recursively find the minimum cost path between tour nodes
    path, cost = dynamic_programming_helper(m, start_node, end_node, tour_nodes, cache)

    # Add starting node to beginning of the best path
    path.insert(0, start_node)

    return path, cost


def dynamic_programming_helper(m, start_node, end_node, tour_nodes, cache):
    """Recursive helper function for dynamic_programming"""

    # Solution already in cache table
    if cache[start_node][list_to_index(tour_nodes)] != -1:
        return cache[start_node][list_to_index(tour_nodes)]
    else:
        # Base case: only one node left to visit
        if len(tour_nodes) == 1:
            last_tour_node = tour_nodes[0]

            # Make a path from last tour node -> end node
            path = tour_nodes + [end_node]

            # Calculate the cost from k -> last tour node -> end node
            cost = m[start_node][last_tour_node] + m[last_tour_node][end_node]

            # Save solution in cache
            cache[start_node][list_to_index(tour_nodes)] = path, cost

            return path, cost
        else:
            min_cost = sys.maxsize
            min_path = []

            # Determine all possible permutations
            for k in tour_nodes:

                # Use each k in tour_nodes as a starting node
                # for the recursive calls
                unvisited = [x for x in tour_nodes if x != k]
                tmp_path, tmp_cost = dynamic_programming_helper(m, k, end_node, unvisited, cache)

                # Save solution in cache
                cache[k][list_to_index(unvisited)] = tmp_path, tmp_cost

                # Take the branch with the least cost
                best_cost_from_k = m[start_node][k] + tmp_cost
                if best_cost_from_k < min_cost:
                    min_cost = best_cost_from_k
                    min_path = [k] + tmp_path

            return min_path, min_cost


def matrix_to_graph(m):
    """Converts adjacency matrix to dictionary"""
    g = {}
    for i in range(len(m)):
        g[i] = [(v, cost) for (v, cost) in enumerate(m[i]) if cost > 0]

    return g


def write_matrix_to_file(m, filename):
    """Writes adjacency matrix to txt file"""
    f = open(filename, "w")
    for row in m:
        for ele in row:
            f.write(str(ele) + " ")
        f.write("\n")
    f.close()


def read_matrix_from_file(filename):
    """Reads adjacency matrix from txt file"""
    f = open(filename, "r")
    first_value = (f.readline().split()[0])
    f.seek(0, 0)
    if "." in first_value:
        return [[float(x) for x in row.split()] for row in f.readlines()]
    else:
        return [[int(x) for x in row.split()] for row in f.readlines()]


def matrix_print(m):
    """Prints a matrix"""
    max_val = max([max(x) for x in m])
    min_val = min(m[0])
    padding = len(str(int(max_val)))
    if isinstance(min_val, float):
        for row in m:
            for ele in row:
                print(f"{ele:{padding + 2}.1f} ", end="")
            print()
    else:
        padding = len(str(max_val))
        for row in m:
            for ele in row:
                print(f"{ele:0{padding}} ", end="")
            print()


def verify_exact_algorithms():
    # Generate random circular cost matrices
    print("************ TESTING CIRCULAR EUCLIDEAN COST MATRICES ************", end="\n\n")
    for i in range(4, 10):
        matrix, min_cost = generate_circular_cost_matrix(i, 1000)
        brute_iter_path, brute_iter_cost = brute_iterative(matrix)
        brute_recur_path, brute_recur_cost = brute_recur(matrix)
        dynamic_path, dynamic_cost = dynamic_programming(matrix)
        print(f"Testing {i}x{i} matrix...", end="\n\n")
        matrix_print(matrix)
        print()
        print(f"Brute Iterative: \t{brute_iter_path} \tCost: {brute_iter_cost}")
        print(f"Brute Recursive: \t{brute_recur_path} \tCost: {brute_recur_cost}")
        print(f"Dynamic Programming: \t{dynamic_path} \tCost: {dynamic_cost}")
        print()

        # Compare solution with minimum distance around the circle
        precision = 8
        if round(brute_recur_cost, precision) != round(min_cost, precision) \
                or round(brute_iter_cost, precision) != round(min_cost, precision) \
                or round(dynamic_cost, precision) != round(min_cost, precision):
            return False

    print("************ TESTING RANDOM COST MATRICES ************", end="\n\n")
    for i in range(4, 10):
        matrix = generate_random_cost_matrix(i, 1000)
        brute_iter_path, brute_iter_cost = brute_iterative(matrix)
        brute_recur_path, brute_recur_cost = brute_recur(matrix)
        dynamic_path, dynamic_cost = dynamic_programming(matrix)
        print(f"Testing {i}x{i} matrix...", end="\n\n")
        matrix_print(matrix)
        print()
        print(f"Brute Iterative: \t{brute_iter_path} \tCost: {brute_iter_cost}")
        print(f"Brute Recursive: \t{brute_recur_path} \tCost: {brute_recur_cost}")
        print(f"Dynamic Programming: \t{dynamic_path} \tCost: {dynamic_cost}")
        print()

        # Compare each solution
        if brute_recur_cost != brute_iter_cost != dynamic_cost:
            return False
    return True


def list_to_index(L):
    """Converts a list to an integer value"""
    if min(L) < 0:
        raise ValueError("List must contain positive values only")
    return sum([2 ** (elem - 1) for elem in L])


def index_to_list(i):
    """Converts an integer value to a list"""
    if i < 0:
        raise ValueError("Index value must be positive")
    bits = []
    while i > 0:
        bits.append(i % 2)
        i //= 2
    return [i + 1 for i, j in enumerate(bits) if j == 1]
