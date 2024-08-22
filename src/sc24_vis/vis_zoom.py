import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2

from sc24_vis.initial import init
from sc24_vis.sim import simulate_next, zoom


def render(
    t: npt.NDArray,
    s: npt.NDArray,
    fire_start_xs: list[int],
    fire_start_ys: list[int],
    zoom_xy: int,
    zoom_magnification: int,
    cut_time=False,
    last_tss=None,
    last_fss=None,
):
    frame = np.ndarray(
        (zoom_xy * zoom_magnification * 3, zoom_xy * zoom_magnification * 4, 3),
        dtype=np.uint8,
    )
    frame.fill(0)

    tss = []
    fss = []

    zooms = zoom(t, s, fire_start_xs, fire_start_ys, zoom_xy)

    for fire_num in range(10):
        zoomed_ts, zoomed_fs = zooms[fire_num]
        t_sum = zoomed_ts.sum()

        tss.append(t_sum)
        fss.append(zoomed_fs.sum())

        inner_frame = np.ndarray((zoom_xy, zoom_xy, 3), dtype=np.uint8)
        inner_frame.fill(0)

        inner_frame[:, :, 2] = (zoomed_fs * 255).astype(np.uint8)
        inner_frame[:, :, 1] = (zoomed_ts * 255).astype(np.uint8)

        inner_frame = np.repeat(inner_frame, zoom_magnification, axis=0)
        inner_frame = np.repeat(inner_frame, zoom_magnification, axis=1)

        cv2.putText(
            inner_frame,
            f"{t_sum:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if cut_time:
            f_sum_r = last_fss[fire_num]
            t_before_sum = last_tss[fire_num]

            t_after_sum = t_sum

            t_delta_sum = t_before_sum - t_after_sum
            ratio = t_delta_sum / f_sum_r

            # cv2.putText(frame, f's={int(t_sum)}', (width // 2, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                inner_frame,
                f"{f_sum_r:.3f}, {t_delta_sum:.3f}, {ratio:.3f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        frame[
            zoom_xy * zoom_magnification * (fire_num % 3) : zoom_xy
            * zoom_magnification
            * (fire_num % 3 + 1),
            zoom_xy * zoom_magnification * (fire_num // 3) : zoom_xy
            * zoom_magnification
            * (fire_num // 3 + 1),
            :,
        ] = inner_frame

    return frame, tss, fss


def vis_zoom(
    input_path: Path,
    output_path: Path,
    zoom_xy: int = 40,
    zoom_magnification: int = 10,
    fps: int = 10,
    steps: int = 50,
    video_dir: Path = Path("./"),
    generate_mp4: bool = False,
) -> None:
    width, height = zoom_xy * zoom_magnification * 4, zoom_xy * zoom_magnification * 3
    video_output_file = video_dir / "zoomed-output_video.avi"

    fourcc = cv2.VideoWriter_fourcc(*"IYUV")
    video_writer = cv2.VideoWriter(
        str(video_output_file.absolute()), fourcc, fps, (width, height)
    )

    cut_timing = 10
    use_steps = steps

    t, f, cuts, fire_start_xs, fire_start_ys = init(input_path, output_path, 0)

    for i in range(use_steps + 1):
        frame, tss, fss = render(
            t, f, fire_start_xs, fire_start_ys, zoom_xy, zoom_magnification
        )

        cv2.putText(
            frame,
            f"t={i:>3}",
            (0, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"ss={sum(tss):.3f}",
            (width // 3, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        video_writer.write(frame)

        if i == 0:
            for _ in range(fps * 1):
                video_writer.write(frame)

        if i == cut_timing:
            for _ in range(fps * 1):
                video_writer.write(frame)

            f_sum_r = sum(fss)
            t_before_sum = sum(tss)

            t -= cuts

            frame, tss, fss = render(
                t,
                f,
                fire_start_xs,
                fire_start_ys,
                zoom_xy,
                zoom_magnification,
                True,
                tss,
                fss,
            )

            t_after_sum = sum(tss)

            t_delta_sum = t_before_sum - t_after_sum
            ratio = t_delta_sum / f_sum_r

            cv2.putText(
                frame,
                f"t={i:>3} (cut)",
                (0, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"ss={sum(tss):.3f}",
                (width // 3, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"f_sum={sum(fss):.3f} delta={t_delta_sum:.3f} r={ratio:.3f}",
                (0, height - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for _ in range(fps * 1):
                video_writer.write(frame)

        if i == use_steps:
            for _ in range(fps * 1):
                video_writer.write(frame)

        t, f = simulate_next(t, f)

        print(f"Generated time = {i:>3}")

    video_writer.release()

    print("Completed! `.avi` file is generated.")

    if generate_mp4:
        print("Converting `.avi` to `.mp4`...")

        subprocess.run(
            f"ffmpeg -loglevel warning -y -i {video_output_file.absolute()} -strict "
            f"-2 {(video_dir / "zoomed-output_video.mp4").absolute()}",
            shell=True,
            check=True,
        )

        print("Completed! `.mp4` file is generated.")
