

# CCFRec

[[official-code](https://github.com/BishopLiu/CCFRec)]

> [!NOTE]
> I provide scripts for generating SIDs based on the fields "Title", "Categories", and "Brand". To extend this functionality to other settings, the configurations in `encode_textual_features.py` and `generate_semantic_ids.py` should be modified accordingly.

> [!NOTE]
> The implementation of CCFRec slightly differs from the original in the "forward encoding" component.


## Usage

1. (`encode_textual_features.py`) Encoding textual features for Title|Categories|Brand ...

2. (`generate_semantic_ids.py`) Generating SIDs ...

3. (`main.py`)

Run with full ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool