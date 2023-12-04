#%%
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, Audio
import subprocess
import pandas as pd
import json
import sys
FILE_DIR = Path(__file__).parent
DATA_DIR = FILE_DIR.parent
if str(DATA_DIR) not in sys.path:
    sys.path.append(str(DATA_DIR))
else:
    print("DATA_DIR already in sys.path", sys.path)
from CMUdict.utils import CMUDict
#%%
cmudict = CMUDict()
#%%
@dataclass
class JamendoLyrics_Song:
    artist: str
    title: str
    genre: str
    language: str
    filename: str

    @property
    def audio_file(self) -> Path:
        return FILE_DIR/f"data/mp3/{self.filename}.mp3"

    @property
    def lyrics_file(self) -> Path:
        return FILE_DIR/f"data/annotations/lines/{self.filename}.csv"

    def get_segments(self, max_duration: float = 10.0):
        segments = []
        df = pd.read_csv(self.lyrics_file)
        current_segments = []
        current_lyrics = []
        for _, row in df.iterrows():
            previous_duration = 0.0
            if len(current_segments) > 0:
                previous_duration = current_segments[-1]['end_time'] - current_segments[0]['start_time']
            current_duration = row["end_time"] - row["start_time"]
            if previous_duration + current_duration >= max_duration:
                segments.append(JamendoLyics_Segment(
                    song=self,
                    start=current_segments[0]['start_time'],
                    end=current_segments[-1]['end_time'],
                    lyrics=" ".join(current_lyrics),
                ))
                current_segments = []
                current_lyrics = []
            current_segments.append(row)
            current_lyrics.append(row["lyrics_line"].strip())
        if len(current_segments) > 0:
            segments.append(JamendoLyics_Segment(
                song=self,
                start=current_segments[0]['start_time'],
                end=current_segments[-1]['end_time'],
                lyrics=" ".join(current_lyrics),
            ))
        return segments

@dataclass
class JamendoLyics_Segment:
    song: JamendoLyrics_Song
    start: float
    end: float
    lyrics: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def audio_file(self) -> Path:
        path = FILE_DIR/f"segments/{self.song.filename}_{self.start}_{self.end}.mp3"
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
        return self.song.audio_file

    @property
    def phonemes(self) -> list[str]:
        phonemes_list = []
        for word in self.lyrics.split():
            for phoneme in cmudict.get_phonemes(word):
                phonemes_list.append(phoneme)
        return phonemes_list

    @property
    def json_dict(self):
        return {
            "song": self.song.title,
            "artist": self.song.artist,
            "genre": self.song.genre,
            "language": self.song.language,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "filename": str(self.audio_file),
            "phonemes": self.phonemes,
            "lyrics": self.lyrics,
        }


#%%
df = pd.read_csv(FILE_DIR/"data/JamendoLyrics.csv")
df = df[df["Language"] == "English"]
df.head()

# %%
songs = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    songs.append(JamendoLyrics_Song(
        artist=row["Artist"],
        title=row["Title"],
        genre=row["Genre"],
        language=row["Language"],
        filename=row["Filepath"][:-4],
    ))
# %%
segments = []
for song in tqdm(songs):
    for segment in song.get_segments():
        segments.append(segment)
segments[0]
# %%
with open(FILE_DIR/"jamendolyrics.jsonl", "w") as f:
    for segment in tqdm(segments):
        f.write(json.dumps(segment.json_dict) + "\n")
print(cmudict.unknown)
#%%
dataset = Dataset.from_json(str(FILE_DIR/"jamendolyrics.jsonl"))
dataset = dataset.cast_column("filename", Audio(sampling_rate=16_000)).rename_column("filename", "audio")
dataset.save_to_disk(FILE_DIR/"jamendolyrics_dataset")

# %% Cleanup segments directory
for file in (FILE_DIR/"segments").glob("*.mp3"):
    file.unlink()
(FILE_DIR/"segments").rmdir()
#%%
