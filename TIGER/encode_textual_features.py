
# %%

import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import numpy as np
import pickle
from tqdm import tqdm

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# %%

root = "../../data"
dataset = "Amazon2014Beauty_550_LOU"
path = os.path.join(root, "Processed", dataset)
item_df = pd.read_csv(os.path.join(path, "item.txt"), sep='\t')

fields = ('TITLE', 'CATEGORIES', 'BRAND')
# fields = ('TITLE',)
for field in fields:
    item_df[field] = item_df[field].fillna('')

item_df['TEXT'] = item_df.apply(
    lambda row: "\n".join([f"{field}: {row[field]}." for field in fields]),
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


def encode_by_llama(
    item_df: pd.DataFrame,
    model_dir: str = "./models",
    model: str = "Llama-2-7b-hf",
    batch_size: int = 32
):
    saved_filename = f"{model}_{'_'.join(fields)}.pkl".lower()
    model_path = os.path.join(model_dir, model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map = 'cuda')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'cuda')

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(item_df), batch_size)):
        # print(i)
        item_names = item_df['TEXT'][i:i+batch_size]
        # 生成输出
        inputs = tokenizer(item_names.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
        seq_embeds = output.hidden_states[-1][:, -1, :].detach().cpu().numpy()
        # break
        if i == 0:
            tFeats = seq_embeds
        else:
            tFeats = np.concatenate((tFeats, seq_embeds), axis=0)
    tFeats = torch.from_numpy(tFeats).float()

    export_pickle(
        tFeats,
        os.path.join(path, saved_filename)
    )

    return tFeats


def encode_textual_modality(
    item_df: pd.DataFrame,
    model: str = "all-MiniLM-L12-v2", model_dir: str = "./models",
    batch_size: int = 128
):
    saved_filename = f"{model}_{'_'.join(fields)}.pkl".lower()
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

# encode_by_llama(item_df)
encode_textual_modality(item_df, model="sentence-t5-xl")