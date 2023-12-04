#%%
import datasets
from pathlib import Path
import sys
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))
from CMUdict.utils import CMUDict

DEBUG = False
def get_timit():
    cmuDict = CMUDict()
    timit = datasets.load_dataset("timit_asr", data_dir=Path(__file__).parent.parent)
    train_test_split = timit["train"].train_test_split(test_size=0.2, seed=42)
    timit['train'] = train_test_split['train']
    timit['validation'] = train_test_split['test']
    def get_phonemes(item):
        phonemes = item['phonetic_detail']['utterance']
        item['phonemes'] = [cmuDict.map_phoneme(phoneme) for phoneme in phonemes]
        return item
    timit = timit.map(get_phonemes).remove_columns([
        'phonetic_detail',
        'file',
        'word_detail',
        'dialect_region',
        'id',
        'sentence_type',
        'speaker_id',
    ])
    if DEBUG:
        print(len(cmuDict.seen), cmuDict.seen)
    return timit
# %%
if __name__ == "__main__":
    DEBUG = True
    timit = get_timit()
    timit