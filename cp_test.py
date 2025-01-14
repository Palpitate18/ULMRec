import os
import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *

from transformers import BitsAndBytesConfig
from pytorch_lightning import seed_everything
from model import LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)


def main(args, export_root=None):
    seed_everything(args.seed)

    if export_root is None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code

    test_loader, tokenizer, test_retrieval = dataloader_factory(args)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint_path,quantization_config=bnb_config)


    trainer = LLMTrainer(args, model, test_loader, tokenizer, export_root, args.use_wandb)

    # 直接进行测试
    trainer.test(test_retrieval)


if __name__ == "__main__":
    set_template(args)
    print(args)
    main(args, export_root=None)