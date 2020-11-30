import random
import itertools
import sys
import math


def generate_random_cost_matrix(num_v, max_cost):
    """Generates undirected connected graph matrix with edge costs"""
    # Generate the first half
    matrix = [[(random.randint(1, max_cost) if i < j else 0) for i in range(num_v)] for j in range(num_v)]

    # Return a symmetrical matrix
    return [[(matrix[i][j] if i > j else matrix[j][i]) for i in range(num_v)] for j in range(num_v)]


def generate_euclidean_cost_matrix(num_v, max_cost):
    """Generates cost matrix for random Euclidean graph"""
    # Generate random (x,y) coordinates
    vertices = [(random.randint(1, max_cost), random.randint(1, max_cost)) for _ in range(num_v)]

    return coordinates_to_cost_matrix(vertices)


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


def generate_circular_cost_matrix(num_v, r):
    """Generates cost matrix for evenly spaced points around a circle"""

    # Determine number of slices around the circle
    theta = (2 * math.pi) / num_v

    # Calculate (x,y) coordinate at each slice
    vertices = [(r * math.cos(i * theta), r * math.sin(i * theta))
                for i in range(num_v)]

    min_distance = 0
    for i in range(len(vertices)):
        u = vertices[i]
        v = vertices[(i + 1) % num_v]
        min_distance += get_distance(u, v)

    # Put vertices in random order
    random.shuffle(vertices)

    # Return matrix and minimum distance
    return coordinates_to_cost_matrix(vertices), min_distance


def get_distance(u, v):
    """Calculates the distance between two 2D points"""
    x1 = u[0]
    x2 = v[0]
    y1 = u[1]
    y2 = v[1]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def greedy_first_attempt(m):
    """Finds a path by always choosing the shortest edge"""
    # Convert data structure from matrix to dictionary
    g = matrix_to_graph(m)

    # Start at vertex 0
    v = 0
    path = [0]
    cost = 0

    # Visit every vertex once
    n = len(g)
    for _ in range(n-1):
        # Make a list of unvisited vertices
        unvisited = [x for x in g[v] if x[0] not in path]

        # Travel to the vertex with the least edge cost
        min_edge_cost = sys.maxsize
        for i in range(len(unvisited)):
            edge_cost = unvisited[i][1]
            if edge_cost < min_edge_cost:
                min_edge_cost = edge_cost
                v = unvisited[i][0]

        # Add the vertex to the path and track cost
        path.append(v)
        cost += min_edge_cost

    path.append(0)
    final_edge_cost = [x[1] for x in g[v] if x[0] == 0]
    cost += min(final_edge_cost)

    return path, cost


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
    for _ in range(n-1):
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


def get_list_permutations(n):
    """Returns all permutations of an n-length list"""
    perms = itertools.permutations(range(n))
    return perms


def brute(m):
    """Finds optimal solution to the TSP by checking every possibility"""
    # Get permutations for an n-1 length list,
    # and sandwich each permutation with zeros
    # (we can always start and end at vertex 0)
    n = len(m)
    perms = get_list_permutations(n-1)
    perms = [[0] + [int(x)+1 for x in perm] + [0] for perm in perms]

    # Check the cost of each permutation
    min_cost = sys.maxsize
    for p in perms:
        cost = 0
        # Calculate cost from 0 -> ... -> 0
        for i in range(1, n+1):
            curr_v = p[i]
            prev_v = p[i-1]
            cost += m[prev_v][curr_v]
        # Update min cost and min perm if shortest seen
        if cost < min_cost:
            min_p = p
            min_cost = cost

    return min_p, min_cost


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


def verify_optimal_solutions():
    # Generate random circular cost matrices
    for i in range(4, 10):
        matrix, min_distance = generate_circular_cost_matrix(i, 1000)
        brute_path, brute_cost = brute(matrix)

        # Compare solution with minimum distance around the circle
        if round(brute_cost, 6) != round(min_distance, 6):
            print(brute_cost, min_distance)
            return False
    return True
