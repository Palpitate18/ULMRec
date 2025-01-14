# ULMRec

## How to test with ULMRec

**Step. 1 **
- Downlaod Llama-2-7b-hf model:https://huggingface.co/meta-llama/Llama-2-7b-hf

Change the configuration of parameters in config.py based on the position of the model
```
parser.add_argument('--llm_base_model', type=str, default='./llama-2-7b-hf') #need change
parser.add_argument('--llm_base_tokenizer', type=str, default='./llama-2-7b-hf') #need change
```

- Downlaod processed data file, retrieved model and other necessary files.
https://drive.google.com/drive/folders/144myBelhEmVdczcD9g1sQO79oBSpka17?usp=drive_link

- Downlaod checkpoint trained model files.
https://drive.google.com/drive/folders/1JDBZwKB0vFe7Fzhz-u9pzpkOOGpu3dfL?usp=drive_link


**Step. 2 **
- Install requirements.

**Step. 3 **
- Run cp_test.py to load checkpoint model.
```
python cp_test.py --dataset_code games
python cp_test.py --dataset_code beauty
```

Before run the file, it's needed to change some configs in config.py
checkpoint_path for Beauty and Games should be replaced with an abosulate path instead of relative path.