"""Entry point for quick manual parsing test."""

from parser import parse_taillard_data


def main() -> None:
    # Adjust the path to an existing Taillard instance file.
    instance = parse_taillard_data("data/JSPLIB/instances/ta01")
    print(instance)


if __name__ == "__main__":
    main()
