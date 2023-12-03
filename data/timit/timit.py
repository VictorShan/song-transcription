#%%
import datasets
from pathlib import Path

#%%
timit = datasets.load_dataset("timit_asr", data_dir=Path(__file__).parent.parent)
timit
# %%
timit.save_to_disk(Path(__file__).parent / "timit_dataset")
# %%
