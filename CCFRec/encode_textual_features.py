
# %%

import pandas as pd
import torch
import numpy as np
import os, pickle, re
import html, emoji
from tqdm import tqdm

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# %%

def clean_text(raw_text: str):
    cleaned_text = raw_text.strip()
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
 
    if cleaned_text==".":
        cleaned_text = ""
    new_text = emoji.replace_emoji(cleaned_text, replace='')

    return new_text

root = "../../data"
dataset = "Amazon2014Beauty_550_LOU"
path = os.path.join(root, "Processed", dataset)
item_df = pd.read_csv(os.path.join(path, "item.txt"), sep='\t')

field = 'TITLE'
# field = 'CATEGORIES'
# field = 'BRAND'
item_df[field] = item_df[field].fillna('')

item_df['TEXT'] = item_df.apply(
    lambda row: clean_text(row[field]),
    axis=1
)

print(item_df['TEXT'].head(5))

# %%


def export_pickle(data, file: str):
    r"""
    Export data into pickle format.

    data: Any
    file: str
        The file (path/filename) to be saved
    """
    fh = None
    try:
        fh = open(file, "wb")
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
    except (EnvironmentError, pickle.PicklingError) as err:
        ExportError_ = type("ExportError", (Exception,), dict())
        raise ExportError_(f"Export Error: {err}")
    finally:
        if fh is not None:
            fh.close()

def encode_textual_modality(
    item_df: pd.DataFrame,
    model: str = "all-MiniLM-L12-v2", model_dir: str = "./models",
    batch_size: int = 128
):
    saved_filename = f"{model}_{field}.pkl".lower()
    sentences = item_df['TEXT']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = SentenceTransformer(
        os.path.join(model_dir, model),
        device=device
    ).eval()

    with torch.no_grad():
        tFeats = encoder.encode(
            sentences, 
            convert_to_tensor=True,
            batch_size=batch_size, show_progress_bar=True
        ).cpu()
    assert tFeats.size(0) == len(item_df), f"Unknown errors happen ..."

    export_pickle(
        tFeats,
        os.path.join(path, saved_filename)
    )

    return tFeats


# %%

encode_textual_modality(item_df, model="sentence-t5-xl")