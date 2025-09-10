#!/usr/bin/env python3
# from src.common import print_C

# from src.neighbors import generate_neighbors_adjacent, generate_neighbors_swap
from src.parser import parser
from src.serach import tabu_search

# from src.permutation_procesing import c_max


def main() -> None:
    data = parser(file_path="data/tai20_5.txt", instance_number=0)
    processing_times = data["processing_times"]
    best_pi, best_cmax = tabu_search(processing_times, max_iter=200, tabu_tenure=10)

    print("Najlepsza kolejność:", best_pi)
    print("Cmax:", best_cmax)

    print(data["info"])
    # neighbors = generate_neighbors_swap(pi)
    # print(f"neighbors {len(neighbors)} neighbors by swapping any two jobs.")
    # print(neighbors)  # print first neighbor
    # neighbors_adj = generate_neighbors_adjacent(pi)
    # print(f"Generated {len(neighbors_adj)} neighbors by swapping adjacent jobs.")
    # print(neighbors_adj)  # print first neighbor


if __name__ == "__main__":
    main()
