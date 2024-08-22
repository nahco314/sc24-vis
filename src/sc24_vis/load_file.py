from pathlib import Path


def load_input(
    input_path: Path, problem_num: int
) -> tuple[int, list[int], list[int], list[float], list[float]]:
    with open(input_path, "r") as file:
        for i in range(10):
            if i != problem_num:
                n = int(file.readline())
                for j in range(n):
                    file.readline()
            else:
                n = int(file.readline())
                x_island_sc = [-1] * n
                y_island_sc = [-1] * n
                t_sc = [-1.0] * n
                f_sc = [-1.0] * n
                for j in range(n):
                    x, y, t, f = file.readline().split()
                    x_island_sc[j] = int(x)
                    y_island_sc[j] = int(y)
                    t_sc[j] = float(t)
                    f_sc[j] = float(f)

                return n, x_island_sc, y_island_sc, t_sc, f_sc

    assert False


def load_program_output(program_output_path: Path, problem_num: int) -> list[float]:
    with open(program_output_path, "r") as file:
        while True:
            file.readline().strip()
            q = file.readline().strip()
            i_num = int(q.split()[-2])
            n = int(q.split()[-1])
            if i_num == problem_num:
                break
            else:
                for _ in range(n):
                    file.readline()

        cuts = []
        for _ in range(n):
            cuts.append(float(file.readline().strip()))

        return cuts
