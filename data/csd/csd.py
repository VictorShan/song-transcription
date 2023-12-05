#%% This directory should contain the english folder of the dataset.
from dataclasses import dataclass
import json
import pandas as pd
import subprocess
from tqdm import tqdm
import sys
from pathlib import Path
FILE_DIR = Path(__file__).parent
DATA_DIR = FILE_DIR.parent
if str(DATA_DIR) not in sys.path:
    sys.path.append(str(DATA_DIR))
else:
    print("DATA_DIR already in sys.path", sys.path)
# %%
metadata = pd.read_json(FILE_DIR/"english/metadata.json", orient="index")
metadata.head()

#%%
mapping = pd.read_csv(FILE_DIR/"phoneme-translation.csv")
csd2cmu = dict(zip(mapping["csd"], mapping["cmu"]))
csd2cmu

# %%
@dataclass
class CSD_Song:
    id: str
    song_name: str

    def get_csv(self) -> pd.DataFrame:
        return pd.read_csv(FILE_DIR / f"english/csv/{self.id}.csv")

    def get_lyrics(self) -> list[str]:
        lyric_lines = []
        with open(FILE_DIR / f"english/lyric/{self.id}.txt", "r") as f:
            lyric_lines = f.readlines()
        return [lyric.strip() for lyric in lyric_lines]

    def get_syllables(self) -> list[str]:
        syllables_lines = []
        with open(FILE_DIR / f"english/txt/{self.id}.txt", "r") as f:
            for syllables_line in f.readlines():
                syllables_line = syllables_line.strip()
                if syllables_line == "":
                    continue
                syllables_lines.append(syllables_line.split())
        return syllables_lines

    def get_csd_phonemes(self) -> list[str]:
        phonemes_lines = []
        for syllable_line in self.get_syllables():
            current_phoneme_line = []
            for syllable in syllable_line:
                for csd_phoneme in syllable.split("_"):
                    current_phoneme_line.append(csd_phoneme)
            phonemes_lines.append(current_phoneme_line)
        return phonemes_lines

    def get_cmu_phonemes(self) -> list[str]:
        cmu_phoneme_lines = []
        for csd_phoneme in self.get_csd_phonemes():
            cmu_phonemes = []
            print(csd_phoneme)
            for phoneme in csd_phoneme:
                cmu_phoneme = csd2cmu[phoneme]
                cmu_phonemes.append(cmu_phoneme)
            cmu_phoneme_lines.append(cmu_phonemes)
        return cmu_phoneme_lines

@dataclass
class CSD_Segment:
    song_id: str
    song_name: str
    start: float
    end: float
    filename: Path
    csd_phonemes: list[str]
    lyrics: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def cmu_phonemes(self) -> list[str]:
        cmu_phonemes = []
        for csd_phoneme in self.csd_phonemes:
            cmu_phoneme = csd2cmu[csd_phoneme]
            cmu_phonemes.append(cmu_phoneme)
        return cmu_phonemes

    @property
    def segment_filename(self) -> Path:
        path = self.filename.parent.parent.parent / f"segments/{self.song_id}_{self.start}_{self.end}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            subprocess.run([
                "ffmpeg",
                "-i", str(self.filename),
                "-ss", str(self.start),
                "-to", str(self.end),
                "-c", "copy", str(path),
                "-loglevel", "error",
                "-hide_banner"
            ])
        return path
#%%
def get_segments(song: CSD_Song) -> list[CSD_Segment]:
    segments = []
    csv = song.get_csv()
    phonemes_lines = song.get_csd_phonemes()
    syllables_lines = song.get_syllables()
    lyrics_lines = song.get_lyrics()
    i = 0
    for n, syllable_line in enumerate(syllables_lines):
        start = csv.iloc[i]["start"]
        end = csv.iloc[i+len(syllable_line)-1]["end"]
        segments.append(CSD_Segment(
            song.id,
            song.song_name,
            start,
            end,
            FILE_DIR / f"english/wav/{song.id}.wav",
            phonemes_lines[n],
            lyrics_lines[n]
        ))
        i += len(syllable_line)
    return segments

#%%
# song = CSD_Song("en001a", "Alphabet")
# segments = get_segments(song)
# segments

# %%
def create_csd_jsonl():
    id = 0
    with open(FILE_DIR/"csd.jsonl", "w") as f:
        for song_id, row in tqdm(metadata.iterrows(), total=len(metadata)):
            song = CSD_Song(song_id, row["songname"])
            segments = get_segments(song)
            for segment in segments:
                segment_dict = {
                    "id": id,
                    "song_id": segment.song_id,
                    "song_name": segment.song_name,
                    "start": segment.start,
                    "end": segment.end,
                    "filename": str(segment.segment_filename),
                    "phonemes": " ".join(segment.cmu_phonemes),
                    "lyrics": segment.lyrics,
                }
                id += 1
                f.write(f'{json.dumps(segment_dict)}\n')
        f.flush()

# %%
from datasets import Dataset, Audio
def create_csd_dataset():
    if not (FILE_DIR/"csd.jsonl").exists():
        create_csd_jsonl()
    dataset = Dataset.from_json(str(FILE_DIR/"csd.jsonl"))
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = dataset["test"].train_test_split(test_size=0.5, seed=42)
    dataset["test"] = test_valid["test"]
    dataset["validation"] = test_valid["train"]
    dataset = dataset.cast_column("filename", Audio(sampling_rate=16000))
    dataset = dataset.rename_column("filename", "audio")
    # dataset.save_to_disk(FILE_DIR/"csd_dataset")
    return dataset
# %% Cleanup segments directory
# for file in (FILE_DIR/"segments").glob("*.wav"):
#     file.unlink()
# (FILE_DIR/"segments").rmdir()
# %%
