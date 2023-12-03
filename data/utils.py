from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from .whisperX.vad import load_vad_model, VoiceActivitySegmentation, merge_chunks
import demucs.separate
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

def get_voice_activity_segments() -> VoiceActivitySegmentation:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_vad_model(device, vad_offset=0.5)

def vad_cut(
        pipeline: VoiceActivitySegmentation,
        audio_filepath: Path,
        dataset="unknown",
        max_duration_sec: float=10,
        onset: float = 0.5,
        offset: Optional[float] = None,
    ) -> list[VadCut]:

    segments = pipeline(audio_filepath)
    output = merge_chunks(segments, max_duration_sec, onset=onset, offset=offset)
    valid_cuts = []
    for segment in output:
        annotation = TimedAnnotation(segment["start"], segment["end"], segment["segments"])
        valid_cuts.append(VadCut(audio_filepath, dataset, annotation.duration()*1000))
    return valid_cuts





