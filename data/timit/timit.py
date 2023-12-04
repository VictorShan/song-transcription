#%%
import datasets
from pathlib import Path

#%%
timit = datasets.load_dataset("timit_asr", data_dir=Path(__file__).parent.parent)
# %%
train_test_split = timit["train"].train_test_split(test_size=0.2, seed=42)
timit['train'] = train_test_split['train']
timit['validation'] = train_test_split['test']
timit
#%%
timit.save_to_disk(Path(__file__).parent / "timit_dataset")
# %%
