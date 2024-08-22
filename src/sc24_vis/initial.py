from pathlib import Path

from sc24_vis.load_file import load_input, load_program_output

import numpy as np
import numpy.typing as npt

from sc24_vis.sim import find_fire_starts


def init(
    input_path: Path, program_output_path: Path, problem_num: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, list[int], list[int]]:
    n, x_island_sc, y_island_sc, t_sc, f_sc = load_input(input_path, problem_num)
    cuts = load_program_output(program_output_path, problem_num)

    t = np.zeros((1000, 1000), dtype=np.float64)
    f = np.zeros((1000, 1000), dtype=np.float64)
    cuts_na = np.zeros((1000, 1000), dtype=np.float64)

    for i in range(n):
        t[x_island_sc[i]][y_island_sc[i]] = t_sc[i]
        f[x_island_sc[i]][y_island_sc[i]] = f_sc[i]
        cuts_na[x_island_sc[i]][y_island_sc[i]] = cuts[i]

    fire_start_xs, fire_start_ys = find_fire_starts(f)

    return t, f, cuts_na, fire_start_xs, fire_start_ys
