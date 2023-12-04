from dataclasses import dataclass
from pathlib import Path

FILE_DIR = Path(__file__).parent

UNK = "[UNK]"
PAD = "[PAD]"
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

# TimitBet 61 phoneme mapping to 39 phonemes
# by Lee, K.-F., & Hon, H.-W. (1989). Speaker-independent phone recognition using hidden Markov models. IEEE Transactions on Acoustics, Speech, and Signal Processing, 37(11), 1641–1648. doi:10.1109/29.46546
phon61_map39 = {
    'iy':'iy',  'ih':'ih',   'eh':'eh',  'ae':'ae',    'ix':'ih',  'ax':'ah',   'ah':'ah',  'uw':'uw',
    'ux':'uw',  'uh':'uh',   'ao':'aa',  'aa':'aa',    'ey':'ey',  'ay':'ay',   'oy':'oy',  'aw':'aw',
    'ow':'ow',  'l':'l',     'el':'l',  'r':'r',      'y':'y',    'w':'w',     'er':'er',  'axr':'er',
    'm':'m',    'em':'m',     'n':'n',    'nx':'n',     'en':'n',  'ng':'ng',   'eng':'ng', 'ch':'ch',
    'jh':'jh',  'dh':'dh',   'b':'b',    'd':'d',      'dx':'dx',  'g':'g',     'p':'p',    't':'t',
    'k':'k',    'z':'z',     'zh':'sh',  'v':'v',      'f':'f',    'th':'th',   's':'s',    'sh':'sh',
    'hh':'hh',  'hv':'hh',   'pcl':'sil', 'tcl':'sil', 'kcl':'sil', 'qcl':'sil','bcl':'sil','dcl':'sil',
    'gcl':'sil','h#':'sil',  '#h':'sil',  'pau':'sil', 'epi': 'sil','nx':'n',   'ax-h':'ah','q':'sil',
    'sil': 'sil', UNK: UNK,  PAD: PAD, "sp": "sil"
}
PHON61_MAP39 = {}
for key, value in phon61_map39.items():
    PHON61_MAP39[key.upper()] = value.upper()

VOCAB = { phoneme: i  for i, phoneme in enumerate(set(PHON61_MAP39.values())) }
VOCAB["|"] = len(VOCAB)
@dataclass
class CMUDict:
    def __init__(self):
        self.map = {}
        self.unknown = []
        self.seen = set()
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

    def get_raw_phonemes(self, word: str) -> list[str]:
        """Get raw phonemes for a given word.

        Args:
            word (str): The word to get phonemes for.

        Returns:
            list[str]: The raw phonemes for the given word.
        """
        word = word.upper()
        if word in self.map:
            return self.map[word.upper()]
        else:
            self.unknown.append(word)
        return [UNK]

    def map_phoneme(self, phoneme):
        phoneme = phoneme.upper()
        if phoneme[-1].isdigit():
            phoneme = phoneme[:-1]
        result = PHON61_MAP39.get(phoneme, UNK)
        self.seen.add(result)
        return result

    def get_phonemes(self, word: str) -> list[str]:
        raw_phonemes = self.get_raw_phonemes(word)
        phonemes = []
        for raw_phoneme in raw_phonemes:
            phonemes.append(self.map_phoneme(raw_phoneme))
        return phonemes