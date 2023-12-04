from dataclasses import dataclass
from pathlib import Path

FILE_DIR = Path(__file__).parent

additional_words = {
    "WASTIN'": "W EY1 S T IH0 NG".split(),
    "LA": "L AA1".split(),
    "LALALALALA": "L AA1 L AA1 L AA1 L AA1".split(),
    "LALALALA": "L AA1 L AA1 L AA1 L AA1".split(),
    "GETTING'":  "G IH1 T AH0 N".split(),
    "PEYOTE": "P EY0 OW1 T IY0".split(),
    "KNIFES": "N AY1 F".split(),
    "BREATHIN'": "B R EH1 TH IH0 NG".split(),
    "HUHHH": "HH AH1 N".split(),
    "HUHHHH": "HH AH1 N".split(),
    "WHUTSUP": "W AH1 T S AH2 P".split(),
    "NIGGAS": "N IH1 G AH0 S".split(),
    "NIGGA": "N IH1 G AH0".split(),
    "HOMIE": "HH OW1 M IY0".split(),
    "WORDLESSLY": "W ER1 D L AH0 S l IY0".split(),
    "REPPIN": "R EH1 P IH0 NG".split(),
    "GOTCHU": "G AA1 CH UW1".split(),
    "FAM": "F AH0 M".split(),
    "D'YOU": "D Y UW1".split(),
    "ACCURSED": "AH0 K ER1 S T".split(),
    "UNPERSUADED": "AH2 N P ER0 S W EY1 D IH0 D".split(),
}

misspellings = {
    "AINT": "AIN'T",
    "COMPLETLY": "COMPLETELY",
    "THATS": "THAT'S",
    "SEPERATED": "SEPARATED",
    "STOPPIN": "STOPPING",
    "POPPIN": "POPPING",
    "SLIPPIN": "SLIPPING",
    "WASNT": "WASN'T",
    "BELEIVING": "BELIEVING",
    "DOIN": "DOING",
    "PARLIMENT": "PARLIAMENT"
}

@dataclass
class CMUDict:
    def __init__(self):
        self.map = {}
        self.unknown = []
        ## Load CMUdict
        with open(FILE_DIR/"cmudict-0.7b.txt", "r") as f:
            for line in f:
                if line.startswith(";;;"):
                    continue
                word, phonemes = line.split("  ")
                self.map[word] = phonemes.split()
        for word, phonemes in additional_words.items():
            self.map[word] = phonemes
        for misspelling, spelling in misspellings.items():
            self.map[misspelling] = self.map[spelling]


    def get_phonemes(self, word: str) -> list[str]:
        """Get phonemes for a given word.

        Args:
            word (str): The word to get phonemes for.

        Returns:
            list[str]: The phonemes for the given word.
        """
        word = word.upper()
        if word in self.map:
            return self.map[word.upper()]
        else:
            self.unknown.append(word)
        return ["UNK"]