#%%
from dataclasses import dataclass
from datasets import Dataset, Audio
from pathlib import Path
from tqdm import tqdm
import subprocess
import json
FILE_DIR = Path(__file__).parent

DATA_DIR = FILE_DIR/"data"
import sys
if str(FILE_DIR.parent) not in sys.path:
    sys.path.append(str(FILE_DIR.parent))
else:
    print("DATA_DIR already in sys.path", sys.path)
from utils import get_voice_activity_pipeline, vad_cut, VadCut

#%%
pipeline = get_voice_activity_pipeline()
#%%
@dataclass
class NUS_Song:
    singer: str
    song: str
    is_read: bool
    is_sing: bool

    @property
    def read_sing_dir(self) -> str:
        if self.is_read:
            return "read"
        elif self.is_sing:
            return "sing"
        else:
            raise Exception("Unknown read_sing_dir name")

    @property
    def audio_file(self) -> Path:
        return DATA_DIR/self.singer/self.read_sing_dir/f"{self.song}.wav"

    @property
    def phonemes(self) -> list[str]:
        phonemes = []
        with open(DATA_DIR/self.singer/self.read_sing_dir/f"{self.song}.txt", "r") as f:
            for line in f.readlines():
                start, end, phoneme = line.strip().split()
                phonemes.append({
                    "start": float(start),
                    "end": float(end),
                    "phoneme": phoneme.upper(),
                })
        return phonemes

    def get_segements(self) -> list:
        segments = []
        vad_cuts = vad_cut(
            pipeline=pipeline,
            audio_filepath=self.audio_file,
        )
        for cut in vad_cuts:
            segments.append(NUS_Segment(
                song=self,
                start=cut.annotation.start,
                end=cut.annotation.end,
            ))
        return segments

@dataclass
class NUS_Segment:
    song: NUS_Song
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def segment_filename(self) -> Path:
        path = FILE_DIR / f"segments/{self.song.singer}_{'read' if self.song.is_read else 'sing'}_{self.start}_{self.end}.wav"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            subprocess.run([
                "ffmpeg",
                "-ss", f"{self.start}",
                "-i", f"{self.song.audio_file}",
                "-t", f"{self.duration}",
                "-c", "copy",
                str(path),
                "-loglevel", "error",
                "-hide_banner"
            ])
        return path

    @property
    def phonemes(self) -> list[str]:
        phonemes = []
        for phoneme in self.song.phonemes:
            if self.start <= phoneme["start"] and phoneme["end"] <= self.end:
                phoneme = phoneme.copy()
                phoneme['start'] -= self.start
                phoneme['end'] -= self.start
                phonemes.append(phoneme)
        return phonemes

    @property
    def json_dict(self) -> dict:
        return {
            "song": self.song.song,
            "singer": self.song.singer,
            "is_read": self.song.is_read,
            "is_sing": self.song.is_sing,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "filename": str(self.segment_filename),
            "timestamped_phonemes": self.phonemes,
            "phonemes": [phoneme["phoneme"] for phoneme in self.phonemes],
        }
#%%
songs = []
for singer in DATA_DIR.iterdir():
    if singer.is_dir():
        for read_sing_dir in singer.iterdir():
            if read_sing_dir.is_dir():
                items = set()
                for file in read_sing_dir.iterdir():
                    if file.is_file():
                        items.add(file.stem)
                for item in items:
                    if read_sing_dir.name == "read":
                        is_read = True
                        is_sing = False
                    elif read_sing_dir.name == "sing":
                        is_read = False
                        is_sing = True
                    else:
                        raise Exception("Unknown read_sing_dir name")
                    song = NUS_Song(
                        singer=singer.name,
                        song=item,
                        is_read=is_read,
                        is_sing=is_sing,
                    )
                    songs.append(song)
#%%
with open(FILE_DIR/"nus.jsonl", "w") as f:
    for song in tqdm(songs, total=len(songs)):
        for segment in song.get_segements():
            f.write(json.dumps(segment.json_dict))
            f.write("\n")
    f.flush()
#%%
dataset = Dataset.from_json(str(FILE_DIR/"nus.jsonl"))
dataset = dataset.train_test_split(test_size=0.2, seed=42)
validation_test = dataset["test"].train_test_split(test_size=0.5, seed=42)
dataset["test"] = validation_test["test"]
dataset["validation"] = validation_test["train"]
dataset = dataset.cast_column("filename", Audio(sampling_rate=16000))
dataset = dataset.rename_column("filename", "audio")
#%%
dataset.save_to_disk(FILE_DIR/"nus_dataset")
# %% Cleanup segments directory
for file in (FILE_DIR/"segments").glob("*.wav"):
    file.unlink()
(FILE_DIR/"segments").rmdir()
#%%