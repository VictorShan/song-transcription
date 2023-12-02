from pathlib import Path
import demucs.separate

def separate_vocals(
        filepath: list[Path],
        output_dir: Path,
        should_speedup: bool = False,
        shifts: int = 1,
        parallel: int = 1,
    ):
    """Separate vocals from a given audio file and save them in the output directory.

    Args:
        filepath (Path): Path to the audio file.
        output_dir (Path): Path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = [
        "-n", f"{'htdemucs' if should_speedup else 'htdemucs_ft'}",
        "-j", f"{parallel}",
        "--shifts", f"{shifts}",
        "--two-stems", "vocals",
        "--filename", f"{output_dir.absolute()}/" + "{track}/{stem}.{ext}",
        " ".join(f'{str(path)}' for path in filepath),
    ]
    demucs.separate.main(arguments)
