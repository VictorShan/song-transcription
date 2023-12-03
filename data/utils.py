from pathlib import Path
from typing import Optional
from pyannote.audio import Pipeline, Model, Inference
from .env import HF_TOKEN
from dataclasses import dataclass
from .whisperX.vad import load_vad_model, VoiceActivitySegmentation, merge_chunks
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

def get_voice_activity_segments() -> VoiceActivitySegmentation:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_vad_model(device, vad_offset=0.5)

def vad_cut(
        pipeline: VoiceActivitySegmentation,
        audio_filepath: Path,
        output_dir: Path,
        max_duration_sec: float=10,
        onset: float = 0.5,
        offset: Optional[float] = None,
    ) -> list[VadCut]:

    segments = pipeline(audio_filepath)
    output = merge_chunks(segments, max_duration_sec, onset=onset, offset=offset)

    valid_segments = []
    for segment in output:
        print(f'{segment["end"] - segment["start"]:0.3f}', segment)
        valid_segments.append(TimedAnnotation(segment["start"], segment["end"], segment["segments"]))

    valid_cuts = []
    for segment in valid_segments:
        segment_filepath = Path(f"{output_dir}/{audio_filepath.stem}_{segment.start}_{segment.end}.wav")
        segment_filepath.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"ffmpeg -i {audio_filepath} -ss {segment.start} -to {segment.end} -c copy {segment_filepath}")
        valid_cuts.append(VadCut(segment_filepath, segment.label, segment.duration()*1000))
    print(f"Valid cuts: {len(valid_cuts)}")
    return valid_cuts





