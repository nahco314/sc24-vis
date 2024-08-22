from numba import jit
import numpy as np
import numpy.typing as npt


beta1 = 0.8
beta2 = 0.4
beta3 = 0.2
hc = 0.1
param_a = 2
param_r = 0.5
gamma = 0.1

param_l = 1000


@jit(nopython=True, cache=True)
def simulate_next(t: npt.NDArray, f: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    next_t = np.zeros((1000, 1000), dtype=np.float64)
    next_f = np.zeros((1000, 1000), dtype=np.float64)

    for x in range(1000):
        for y in range(1000):
            h = beta1 * f[x][y]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 1000 and 0 <= ny < 1000:
                    h += beta2 * f[nx][ny]
            for dx, dy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 1000 and 0 <= ny < 1000:
                    h += beta3 * f[nx][ny]
            phi = 0 if h < hc else np.tanh(param_a * (h - hc))
            phi *= t[x][y]

            next_t[x][y] = t[x][y] - phi
            next_f[x][y] = f[x][y] + phi - gamma * f[x][y]

    return next_t, next_f


@jit(nopython=True, cache=True)
def find_fire_starts(
    f: npt.NDArray,
) -> tuple[list[int], list[int]]:
    res_x = []
    res_y = []

    for x in range(param_l):
        for y in range(param_l):
            if f[x][y] > 0:
                res_x.append(x)
                res_y.append(y)

    return res_x, res_y


@jit(nopython=True, cache=True)
def zoom(
    t: npt.NDArray,
    f: npt.NDArray,
    fire_start_xs: list[int],
    fire_start_ys: list[int],
    zoom_xy: int,
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    res_lst = []

    for i in range(10):
        res_t = np.zeros((zoom_xy, zoom_xy), dtype=np.float64)
        res_f = np.zeros((zoom_xy, zoom_xy), dtype=np.float64)
        for x in range(zoom_xy):
            for y in range(zoom_xy):
                nx = fire_start_xs[i] - zoom_xy // 2 + x
                ny = fire_start_ys[i] - zoom_xy // 2 + y
                if 0 <= nx < param_l and 0 <= ny < param_l:
                    res_t[x][y] = t[nx][ny]
                    res_f[x][y] = f[nx][ny]
                else:
                    res_t[x][y] = 0
                    res_f[x][y] = 0

        res_lst.append((res_t, res_f))

    return res_lst
