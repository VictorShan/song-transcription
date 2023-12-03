from dataclasses import dataclass
from pathlib import Path

FILE_DIR = Path(__file__).parent

@dataclass
class CMUDict:
    def __init__(self):
        self.map = {}

        ## Load CMUdict
        with open(FILE_DIR/"cmudict-0.7b.txt", "r") as f:
            for line in f:
                if line.startswith(";;;"):
                    continue
                word, phonemes = line.split("  ")
                self.map[word] = phonemes.split()


    def get_phonemes(self, word: str) -> list[str]:
        """Get phonemes for a given word.

        Args:
            word (str): The word to get phonemes for.

        Returns:
            list[str]: The phonemes for the given word.
        """
        return self.map[word.upper()]