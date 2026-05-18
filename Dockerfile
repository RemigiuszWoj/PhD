# Experiment runtime image: PFSP CPU/QPU experiments (ILS, SA, D-Wave QUBO).
FROM python:3.11-slim

# build-essential covers any dwave sub-dependency shipped only as an sdist.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first — this layer is cached across code-only commits.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Project code. data/ and results/ are provided as volumes at run time,
# so the image stays small and rebuilds are not triggered by new results.
COPY src/ ./src/
COPY main.py config.yaml ./

VOLUME ["/app/data", "/app/results"]

# Default command runs the experiment batch defined in config.yaml.
CMD ["python", "main.py"]
