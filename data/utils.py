from pathlib import Path
from pyannote.audio import Pipeline
from .env import HF_TOKEN
from dataclasses import dataclass
import demucs.separate
import os
import torch

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

@dataclass
class VadCut:
    filepath: Path
    dataset: str
    duration_ms: int

@dataclass
class TimedAnnotation:
    start: int
    end: int
    label: str

    def duration(self) -> int:
        return self.end - self.start

def vad_cut(audio_filepath: Path, output_dir: Path, max_duration_sec: float, max_silence_sec: float) -> list[VadCut]:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("========== Audio File", audio_filepath)
    diarization = pipeline(audio_filepath)
    valid_segements = []
    for segment, label in diarization.itertracks(yield_label=False):
        if len(valid_segements) == 0:
            valid_segements.append(TimedAnnotation(segment.start, segment.end, label))
        else:
            last_segment = valid_segements[-1]
            segment_duration_sec = segment.end - segment.start
            if last_segment.duration() + segment_duration_sec < max_duration_sec \
                    and segment.start - last_segment.end < max_silence_sec:
                last_segment.end = segment.end
            else:
                valid_segements.append(TimedAnnotation(segment.start, segment.end, label))

    valid_cuts = []
    for segment in valid_segements:
        segment_filepath = Path(f"{output_dir}/{audio_filepath.stem}_{segment.start}_{segment.end}.wav")
        segment_filepath.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"ffmpeg -i {audio_filepath} -ss {segment.start} -to {segment.end} -c copy {segment_filepath}")
        valid_cuts.append(VadCut(segment_filepath, segment.label, segment.duration()*1000))
    print(f"Valid cuts: {len(valid_cuts)}")
    return valid_cuts





