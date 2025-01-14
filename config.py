import numpy as np
import random
import torch
import argparse


RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'


def set_template(args):
    if args.dataset_code == None:
        print('******************** Dataset Selection ********************')
        dataset_code = {'b': 'beauty', 'g': 'games'}
        args.dataset_code = dataset_code['g']

    dataset_configs = {
        'beauty': {
            'llm_retrieved_path': './data/retrieve/beauty',
            'user_ids_path': './data/Beauty.index.json',
            'preference_path': './data/beauty_pref.json',
            'checkpoint_path': './checkpoint/beauty' #need change
        },
        'games': {
            'llm_retrieved_path': './data/retrieve/games',
            'user_ids_path': './data/Games.index.json',
            'preference_path': './data/games_pref.json',
            'checkpoint_path': './checkpoint/games' #need change
        }
    }

    config = dataset_configs[args.dataset_code]
    args.llm_retrieved_path = config['llm_retrieved_path']
    args.user_ids_path = config['user_ids_path']
    args.preference_path = config['preference_path']
    args.checkpoint_path = config['checkpoint_path']


    args.bert_max_len = 50
    batch = 16
    args.lora_micro_batch_size = batch

    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20, 50]
    args.rerank_metric_ks = [5, 10]
    args.best_metric = 'Recall@20'
    args.rerank_best_metric = 'NDCG@10'

    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--sliding_window_size', type=float, default=1.0)
parser.add_argument('--negative_sample_size', type=int, default=10)

################
# Trainer
################
# optimization #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=1600)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)
parser.add_argument('--task_ratio', type=float, default=0.5)


# evaluation #
parser.add_argument('--val_strategy', type=str, default='iteration', choices=['epoch', 'iteration'])
parser.add_argument('--val_iterations', type=int, default=500)
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=20)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50])
parser.add_argument('--rerank_metric_ks', nargs='+', type=int, default=[1, 5, 10, 20])
parser.add_argument('--best_metric', type=str, default='Recall@20')
parser.add_argument('--rerank_best_metric', type=str, default='NDCG@10')
parser.add_argument('--use_wandb', type=bool, default=False)


################
# LLM Model
################
parser.add_argument('--llm_base_model', type=str, default='./llama-2-7b-hf') #need change
parser.add_argument('--llm_base_tokenizer', type=str, default='./llama-2-7b-hf') #need change
parser.add_argument('--llm_max_title_len', type=int, default=32)
parser.add_argument('--llm_max_text_len', type=int, default=1536)
parser.add_argument('--llm_max_history', type=int, default=20)
parser.add_argument('--llm_train_on_inputs', type=bool, default=False)
parser.add_argument('--llm_negative_sample_size', type=int, default=19)
parser.add_argument('--llm_system_template_traintest', type=str,default="Given user index, user history in chronological order and candidate items, recommend an item from the candidate pool with its index letter for this user.")
parser.add_argument('--llm_input_template_trainseq', type=str, default='User ID: {}; \n User history: {}; \n Candidate pool: {}')
parser.add_argument('--llm_load_in_4bit', type=bool, default=True)
parser.add_argument('--llm_retrieved_path', type=str, default='./data/retrieve/games')
parser.add_argument('--llm_cache_dir', type=str, default=None)
parser.add_argument('--user_ids_path', type=str, 
    default='./data/Games.index.json', help='Path to the user indices json file')
parser.add_argument('--preference_path', type=str,
    default='./data/games_pref.json', help='Path to the preference json file')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/games',
                   help='Path to the model checkpoint for testing')

################
# Lora
################
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_target_modules', type=list, default=['q_proj', 'v_proj'])
parser.add_argument('--lora_num_epochs', type=int, default=1)
parser.add_argument('--lora_val_iterations', type=int, default=100)
parser.add_argument('--lora_early_stopping_patience', type=int, default=20)
parser.add_argument('--lora_lr', type=float, default=1e-4)
parser.add_argument('--lora_micro_batch_size', type=int, default=16)

################


args = parser.parse_args()
