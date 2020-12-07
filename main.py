import tsp
import pprint
import sys
from os.path import dirname, join
from time import perf_counter_ns, time_ns

# Route print statements to file
current_dir = dirname(__file__)
file_name = "verification_" + str(time_ns()) + ".txt"
file_path = join(current_dir, file_name)
sys.stdout = open(file_path, "w")


def main():

    print("Verified:", tsp.verify_exact_algorithms())


if __name__ == "__main__":
    main()
