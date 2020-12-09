# CSC482-Lab0xFF-TSP

## Working with cost matrices

Generate cost matrix for **evenly spaced (x,y) coordinates around a circle**
  * Vertex numbers in random order
  * Traveling around the circle will always be the shortest path
  
``` python
>>> n = 5
>>> radius = 10
>>> m, min_path, min_cost = tsp.generate_circular_cost_matrix(n, radius)
>>>
>>> tsp.matrix_print(m)
 0.0 11.8 11.8 19.0 19.0
11.8  0.0 19.0 11.8 19.0
11.8 19.0  0.0 19.0 11.8
19.0 11.8 19.0  0.0 11.8
19.0 19.0 11.8 11.8  0.0
>>>
>>> min_path
[0, 1, 3, 4, 2, 0]
>>>
>>> min_cost
58.77852522924731
```

Generate cost matrix for **random Euclidean graph**
``` python
>>> m = tsp.generate_euclidean_cost_matrix(n, max_cost)
>>> tsp.matrix_print(m)
0.0 1.4 2.8 4.1 4.1
1.4 0.0 1.4 3.6 3.0
2.8 1.4 0.0 3.6 2.2
4.1 3.6 3.6 0.0 5.8
4.1 3.0 2.2 5.8 0.0
```

Generate **random cost matrix** and **write to file**
``` python
>>> n = 5
>>> max_cost = 10
>>> m = tsp.generate_random_cost_matrix(n, max_cost)
>>> tsp.write_matrix_to_file(m, "matrix.txt")
```

**Read matrix from file** and **print matrix**
``` python
>>> m = tsp.read_matrix_from_file("matrix.txt")
>>> tsp.matrix_print(m)
0 2 4 8 3
2 0 2 4 6
4 2 0 7 7
8 4 7 0 3
3 6 7 3 0
```
Convert cost matrix to **dictionary graph strutcure**
``` python
>>> import pprint
>>> pp = pprint.PrettyPrinter(indent=4)
>>>
>>> g = tsp.matrix_to_graph(m)
>>> pp.pprint(g)
{   0: [(1, 2), (2, 4), (3, 8), (4, 3)],
    1: [(0, 2), (2, 2), (3, 4), (4, 6)],
    2: [(0, 4), (1, 2), (3, 7), (4, 7)],
    3: [(0, 8), (1, 4), (2, 7), (4, 3)],
    4: [(0, 3), (1, 6), (2, 7), (3, 3)]}

```

## TSP algorithms

Brute force (recursive and iterative)
``` python
>>> tsp.tsp_brute_recur(m)
([0, 2, 1, 3, 4, 0], 16)
>>>
>>> tsp.tsp_brute_iterative(m)
([0, 2, 1, 3, 4, 0], 16)
```

Bellman–Held–Karp dynamic programming algorithm
``` python
>>> tsp.tsp_dynamic(m)
([0, 2, 1, 3, 4, 0], 16)
```

Greedy (shortest next step) algorithm
``` python
>>> tsp.tsp_greedy(m)
([0, 1, 2, 3, 4, 0], 17)
```

Ant colony algorithm (one possible outcome)
``` python
>>> tsp.tsp_ant_colony(m, 10, 1, 1, 20)
([0, 4, 3, 1, 2, 0], 16)
```
