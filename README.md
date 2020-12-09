# CSC482-Lab0xFF-TSP

## Working with cost matrices
Generate random cost matrix:
```
>>> n = 5
>>> max_cost = 10
>>> m = tsp.generate_random_cost_matrix(n, max_cost)
```
Read and write from file:
```
>>> tsp.write_matrix_to_file(m, "matrix.txt")
>>> m = tsp.read_matrix_from_file("matrix.txt")
>>> tsp.matrix_print(m)
0 2 4 8 3
2 0 2 4 6
4 2 0 7 7
8 4 7 0 3
3 6 7 3 0
```
Convert to dictionary graph strutcure:
```
>>> g = tsp.matrix_to_graph(m)
>>> pp.pprint(g)
{   0: [(1, 2), (2, 4), (3, 8), (4, 3)],
    1: [(0, 2), (2, 2), (3, 4), (4, 6)],
    2: [(0, 4), (1, 2), (3, 7), (4, 7)],
    3: [(0, 8), (1, 4), (2, 7), (4, 3)],
    4: [(0, 3), (1, 6), (2, 7), (3, 3)]}

```

## TSP algorithms

Brute force (recursive and iterative):
```
>>> tsp.tsp_brute_recur(m)
([0, 2, 1, 3, 4, 0], 16)
>>>
>>> tsp.tsp_brute_iterative(m)
([0, 2, 1, 3, 4, 0], 16)
```

Bellman–Held–Karp dynamic programming algorithm:
```
>>> tsp.tsp_dynamic(m)
([0, 2, 1, 3, 4, 0], 16)
```

Greedy (shortest next step) algorithm:
```
>>> tsp.tsp_greedy(m)
([0, 1, 2, 3, 4, 0], 17)
```

Ant colony algorithm (possible outcome):
```
>>> tsp.tsp_ant_colony(m, 10, 1, 1, 20)
([0, 4, 3, 1, 2, 0], 16)
```
