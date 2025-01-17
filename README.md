# ULMRec


**Step. 1**
- Downlaod Llama-2-7b-hf model:https://huggingface.co/meta-llama/Llama-2-7b-hf

Change the configuration of parameters in config.py based on the position of the model
```
parser.add_argument('--llm_base_model', type=str, default='./llama-2-7b-hf') #need change
parser.add_argument('--llm_base_tokenizer', type=str, default='./llama-2-7b-hf') #need change
```

- Downlaod retrieved retrieved data file in /data/retrieve.<br />
Games dataset: https://www.dropbox.com/scl/fi/l7jc3ql7ykul9yq5tsvoy/games.zip?rlkey=yv0ue23b2w0c6jt2k5halfwc7&st=vt91ccpa&dl=0
Beauty dataset: https://www.dropbox.com/scl/fi/ith9oa1x093eyseqsdnaq/beauty.zip?rlkey=7953mlyclfkoy8y2hq9r41j7a&st=olj1qyjv&dl=0


**Step. 2**
- Install requirements.

**Step. 3**
- Run cp_test.py to load checkpoint model.
```
python cp_test.py --dataset_code games
python cp_test.py --dataset_code beauty
```

Before run the file, it's needed to change some configs in config.py
checkpoint_path for Beauty and Games should be replaced with an abosulate path instead of relative path.
