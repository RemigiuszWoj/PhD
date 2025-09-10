def parser(file_path: str, instance_number: int = 0) -> dict:
    instances = []
    with open(file_path, "r") as f:
        lines = iter([line.strip() for line in f if line.strip()])

    for line in lines:
        if line.startswith("number of jobs"):
            header = next(lines).split()
            jobs, machines, seed, upper_bound, lower_bound = map(int, header)
            instance_info = {
                "jobs": jobs,
                "machines": machines,
                "seed": seed,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
            }

            next(lines)  # skip"processing times :"
            processing_times = [list(map(int, next(lines).split())) for _ in range(machines)]

            instances.append({"info": instance_info, "processing_times": processing_times})

    if instance_number < 0 or instance_number >= len(instances):
        raise IndexError(f"instance_number {instance_number} out of range")

    return instances[instance_number]
