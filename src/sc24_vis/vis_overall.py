import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2

from sc24_vis.initial import init
from sc24_vis.sim import simulate_next, param_l


def render(
    t: npt.NDArray, f: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, float, float]:
    t_sum = t.sum()
    f_sum = f.sum()

    frame = np.ndarray((param_l, param_l, 3), dtype=np.uint8)
    frame.fill(0)

    ts = (t * 255).astype(np.uint8)
    fs = (f * 255).astype(np.uint8)

    frame[:, :, 2] = fs
    frame[:, :, 1] = ts

    return frame, ts, fs, t_sum, f_sum


def vis_overall(
    input_path: Path,
    output_path: Path,
    fps: int = 30,
    steps: int = 500,
    video_dir: Path = Path("./"),
    generate_mp4: bool = False,
) -> None:
    width, height = param_l, param_l
    video_output_file = video_dir / "output_video.avi"

    fourcc = cv2.VideoWriter_fourcc(*"IYUV")
    video_writer = cv2.VideoWriter(
        str(video_output_file.absolute()), fourcc, fps, (width, height)
    )

    cut_timing = 10
    use_steps = steps

    t, f, cuts, _, _ = init(input_path, output_path, 0)

    frame, _, _, t_sum, f_sum = render(t, f)

    for i in range(use_steps + 1):
        frame, _, _, t_sum, f_sum = render(t, f)

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
            f"s={t_sum:.3f}",
            (width // 2, height - 30),
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

            f_sum_r = f_sum
            t_before_sum = t_sum

            t -= cuts

            frame, ts, fs, t_sum, f_sum = render(t, f)

            t_after_sum = t_sum

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
                f"s={t_sum:.3f}",
                (width // 2, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"f_sum={f_sum_r:.3f} delta={t_delta_sum:.3f} r={ratio:.3f}",
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
            f"-2 {(video_dir / "output_video.mp4").absolute()}",
            shell=True,
            check=True,
        )

        print("Completed! `.mp4` file is generated.")
