def c_max(pi: list[int], processing_times: list[list[int]]) -> int:
    # number of machines
    m = len(processing_times)
    # number of jobs
    n = len(pi)

    # initialize completion time matrix C with zeros
    C = [[0 for _ in range(n)] for _ in range(m)]
    # first job on first machine
    C[0][0] = processing_times[0][pi[0]]

    # first machine, remaining jobs
    for j in range(1, n):
        C[0][j] = C[0][j - 1] + processing_times[0][pi[j]]

    # first job, remaining machines
    for i in range(1, m):
        C[i][0] = C[i - 1][0] + processing_times[i][pi[0]]

    # remaining jobs and machines
    for i in range(1, m):
        for j in range(1, n):
            # the job can start only after the previous job on the same machine
            # and after the same job on the previous machine
            C[i][j] = max(C[i - 1][j], C[i][j - 1]) + processing_times[i][pi[j]]

    # C_max is the completion time of the last job on the last machine
    return C[m - 1][n - 1]
