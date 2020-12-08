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

    # Generate (x,y) coordinate at each slice
    coordinates = [(r * math.cos(i * theta), r * math.sin(i * theta))
                   for i in range(num_v)]

    # Use the first two points to determine
    # the min cost to travel around the circle
    min_cost = get_distance(coordinates[0], coordinates[1]) * num_v

    # Make an indexed list of coordinates
    vertices = list(enumerate(coordinates))

    # Put the vertex list in a new random order
    random.shuffle(vertices)

    # Use the new position of initial vertex numbers
    # to determine min path
    min_path = []
    for i in range(len(vertices)):
        closest_vertex = [y[0] for y in vertices].index(i)
        min_path.append(closest_vertex)
    zero_loc = min_path.index(0)
    min_path = min_path[zero_loc:] + min_path[:zero_loc] + [0]

    # Strip coordinates from indexed list
    coordinates = [i[1] for i in vertices]

    # Return cost matrix and minimum cost
    return coordinates_to_cost_matrix(coordinates), min_path, min_cost


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


def tsp_ant_colony(m, num_ants, max_unchanged_steps):
    """Ant colony TSP algorithm"""
    n = len(m)
    PHERO_FACTOR = 1.5
    DECAY_FACTOR = .5
    min_cost = sys.maxsize

    # Set phero matrix to initial probability based on edge cost
    phero = [[1 / j if int(j) != 0 else 0 for j in row] for row in m]

    steps_since_path_changed = 0
    while steps_since_path_changed < max_unchanged_steps:
        print("STEP:\t\t", steps_since_path_changed)
        # Init new pheromones matrix to zeros
        new_phero = [[0] * n for _ in range(n)]

        # Each ant embarks on a probabilistic path
        for ant in range(num_ants):
            vertices = list(range(n))

            # Ant starts at vertex 0
            u = vertices.pop(0)
            path = [u]
            cost = 0
            total_attraction = 0

            # And randomly picks remaining vertices,
            # while influenced by their attractiveness
            for _ in range(n - 1):

                # Build list of unvisited vertices
                row = enumerate(m[u])
                unvisited = [v for v in row if v[0] in vertices]
                attraction = [0] * n

                # Determine attraction level of each vertex
                for v in unvisited:
                    v_id = v[0]
                    v_cost = v[1]
                    v_attraction = (1 * phero[u][v_id]) / v_cost
                    attraction[v_id] = v_attraction
                    total_attraction += v_attraction

                # Generate random number between 0 and 1
                Q = random.random()
                cumulative_probability = 0

                # Ant makes the choice of where to go next
                for v in unvisited:
                    v_id = v[0]
                    v_probability = attraction[v_id] / total_attraction
                    cumulative_probability += v_probability
                    if Q < cumulative_probability:
                        break

                # And then goes there
                u = v_id
                path.append(u)
                vertices.remove(u)
                cost += v[1]

            # Ant completes the journey back home
            # and final path and cost are determined
            path.append(0)
            cost += m[u][0]

            # Check if this ant's path is the best so far
            if cost < min_cost:
                min_cost = cost
                min_path = path
                min_path_changed = True

            # Ant lays pheromones on its path
            for i in range(n):
                u = path[i]
                v = path[(i + 1) % n]
                # print(f"Laying pheromones from {u} to {v}")
                new_phero[u][v] += PHERO_FACTOR / cost

        # Decay previous pheromone values
        phero = [[x * DECAY_FACTOR for x in row] for row in phero]

        # Add new pheromones
        for i, row in enumerate(phero):
            for j, _ in enumerate(row):
                phero[i][j] += new_phero[i][j]

        if min_path_changed:
            steps_since_path_changed = 0
        else:
            steps_since_path_changed += 1
        min_path_changed = False

    return min_path, min_cost


def tsp_greedy(m):
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


def tsp_brute_iterative(m):
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


def tsp_brute_recur(m):
    """Finds optimal solution to the TSP by checking every possibility"""
    # Start and end with the same node to complete the circuit
    start_node = end_node = 0

    # Build list of unvisited nodes
    tour_nodes = [x for x in range(len(m)) if x != start_node]

    # Recursively find the minimum cost path between tour nodes
    path, cost = tsp_brute_recur_helper(m, start_node, end_node, tour_nodes)

    # Add starting node to beginning of the best path
    path.insert(0, start_node)

    return path, cost


def tsp_brute_recur_helper(m, start_node, end_node, tour_nodes):
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
            tmp_path, tmp_cost = tsp_brute_recur_helper(m, k, end_node, unvisited)

            # Take the branch with the least cost
            best_cost_from_k = m[start_node][k] + tmp_cost
            if best_cost_from_k < min_cost:
                min_cost = best_cost_from_k
                min_path = [k] + tmp_path

        return min_path, min_cost


def tsp_dynamic(m):
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
    path, cost = tsp_dynamic_helper(m, start_node, end_node, tour_nodes, cache)

    # Add starting node to beginning of the best path
    path.insert(0, start_node)

    return path, cost


def tsp_dynamic_helper(m, start_node, end_node, tour_nodes, cache):
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
                tmp_path, tmp_cost = tsp_dynamic_helper(m, k, end_node, unvisited, cache)

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


def verify_exact_algorithms(graph_type, max_value):
    # Make list of exact algorithms to test
    exact_algs = [tsp_brute_iterative, tsp_brute_recur, tsp_dynamic]

    # Add "greedy" if using a circular graph,
    # as it will always return the min path
    if graph_type == "circular":
        exact_algs.append(tsp_greedy)

    num_algs = len(exact_algs)
    results = [0] * num_algs
    largest_size = 10

    # Test random graphs of increasing size
    for i in range(4, largest_size):

        # Generate new i x i graph
        if graph_type == "circular":
            matrix, min_path, min_cost = generate_circular_cost_matrix(i, max_value)
        elif graph_type == "euclidean":
            matrix = generate_euclidean_cost_matrix(i, max_value)
        elif graph_type == "random":
            matrix = generate_random_cost_matrix(i, max_value)
        else:
            raise ValueError("Graph type not recognized")

        print(f"Testing {i}x{i} {graph_type.upper()} matrix...")
        matrix_print(matrix)
        print()

        # Print min path and min cost from each alg
        for x in range(num_algs):
            results[x] = exact_algs[x](matrix)
            alg_path = results[x][0]
            alg_cost = round(results[x][1], 2)
            print(f"{exact_algs[x].__name__:15}",
                  f"\t{str(alg_path):{largest_size * 3 + 2}}\t{alg_cost}")

        # Ensure all algorithms return the same path as each other
        # (or the reverse)
        for x in range(num_algs - 1):
            if results[x][0] != results[x + 1][0] \
                    and results[x][0] != list(reversed(results[x + 1][0])):
                return False
            # Check if min cost is accurate on circular graphs where it is known
            if graph_type == "circular":
                if round(results[x][1], 2) != round(min_cost, 2):
                    return False

        # If we made it this far, we know the results
        # are consistent between algorithms.
        # If using circular graphs, we can do one more test
        # to make sure the algorithms match the known min path
        if graph_type == "circular":
            if results[0][0] != min_path and results[0][0] != list(reversed(min_path)):
                return False

        print()
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
