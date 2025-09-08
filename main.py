from src.parser import parser


def main() -> None:
    data = parser(file_path="data/tai20_5.txt", instance_number=0)
    print(data)


if __name__ == "__main__":
    main()
