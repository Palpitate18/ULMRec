from .base import AbstractDataloader
from .utils import Prompter

import torch
import random
import numpy as np
import pandas as pd
import torch.utils.data as data_utils
import json
import os
import pickle
import transformers
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
from trainer import absolute_recall_mrr_ndcg_for_ks



def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)
    np.random.seed(np.random.get_state()[1][0] + worker_id)



def generate_and_tokenize_eval(args, data_point, tokenizer, prompter):
    in_prompt = prompter.generate_prompt(data_point["system"],
                                         data_point["input"])

    #print("Eval Prompt:", in_prompt)
    tokenized_full_prompt = tokenizer(in_prompt,
                                      truncation=True,
                                      max_length=args.llm_max_text_len,
                                      padding=False,
                                      return_tensors=None)

    tokenized_full_prompt["labels"] = ord(data_point["output"]) - ord('A')
    #print(data_point['output'])
    return tokenized_full_prompt


def seq_to_token_ids(args, seq, candidates, label, text_dict, tokenizer, prompter, user_id=None, preference=None,
                     mode='train', task_type='seq'):
    def truncate_title(title):
        title_ = tokenizer.tokenize(title)[:args.llm_max_title_len]
        title = tokenizer.convert_tokens_to_string(title_)
        return title

    seq_t = ' \n '.join(['(' + str(idx + 1) + ') ' + truncate_title(text_dict[item])
                            for idx, item in enumerate(seq)])

    can_t = ' \n '.join(['(' + chr(ord('A') + idx) + ') ' + truncate_title(text_dict[item])
                         for idx, item in enumerate(candidates)])
    output = chr(ord('A') + candidates.index(label))

    data_point = {}

   
    data_point[
        'system'] = args.llm_system_template_traintest if args.llm_system_template_traintest is not None else DEFAULT_SYSTEM_PROMPT
    input_template = args.llm_input_template_trainseq
    data_point['input'] = input_template.format(user_id, seq_t, can_t)
    data_point['output'] = output
    return generate_and_tokenize_eval(args, data_point, tokenizer, prompter)



class LLMDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        seq_dataset = dataset.load_dataset()
        self.train = seq_dataset['train']
        self.val = seq_dataset['val']
        self.test = seq_dataset['test']
        self.umap = seq_dataset['umap']
        self.reverse_umap = {v: k for k, v in self.umap.items()}
        self.smap = seq_dataset['smap']
        self.text_dict = seq_dataset['meta']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_items = self.item_count
        self.max_len = args.llm_max_history

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_base_tokenizer, cache_dir=args.llm_cache_dir)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.clean_up_tokenization_spaces = True

        self.prompter = Prompter()

        self.llm_retrieved_path = args.llm_retrieved_path
        print('Loading retrieved file from {}'.format(self.llm_retrieved_path))
        retrieved_file = pickle.load(open(os.path.join(args.llm_retrieved_path,
                                                       'retrieved.pkl'), 'rb'))
        with open(args.user_ids_path, 'r') as f:
            self.user_ids = json.load(f)

        with open(args.preference_path, 'r') as f:
            self.preference = json.load(f)

        print('******************** Constructing Test Subset ********************')
        self.test_probs = retrieved_file['test_probs']
        self.test_labels = retrieved_file['test_labels']
        self.test_metrics = retrieved_file['test_metrics']
        self.test_users = [u for u, (p, l) in enumerate(zip(self.test_probs, self.test_labels), start=1) \
                           if l in torch.topk(torch.tensor(p), self.args.llm_negative_sample_size + 1).indices]
        self.test_candidates = [torch.topk(torch.tensor(self.test_probs[u - 1]),
                                           self.args.llm_negative_sample_size + 1).indices.tolist() for u in
                                self.test_users]
        self.non_test_users = [u for u, (p, l) in enumerate(zip(self.test_probs, self.test_labels), start=1) \
                               if l not in torch.topk(torch.tensor(p), self.args.llm_negative_sample_size + 1).indices]
        self.test_retrieval = {
            'original_size': len(self.test_probs),
            'retrieval_size': len(self.test_candidates),
            'original_metrics': self.test_metrics,
            'retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.test_probs)[torch.tensor(self.test_users) - 1],
                torch.tensor(self.test_labels)[torch.tensor(self.test_users) - 1],
                self.args.metric_ks,
            ),
            'non_retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.test_probs)[torch.tensor(self.non_test_users) - 1],
                torch.tensor(self.test_labels)[torch.tensor(self.non_test_users) - 1],
                self.args.metric_ks,
            ),
        }
        self.args.dataloader = self

    @classmethod
    def code(cls):
        return 'llm'

    def get_pytorch_dataloaders(self):

        test_loader = self._get_test_loader()

        return test_loader

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'test':
            dataset = LLMTestDataset(self.args, self.train, self.val, self.test, self.max_len, \
                                     self.rng, self.text_dict, self.tokenizer, self.prompter, self.test_users, \
                                     self.test_candidates, self.user_ids, self.reverse_umap)
        return dataset




class LLMTestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2val, u2answer, max_len, rng, text_dict, tokenizer, prompter, test_users,
                 test_candidates, user_ids ,reverse_umap):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        self.users = sorted(u2seq.keys())
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.test_users = test_users
        self.test_candidates = test_candidates
        self.user_ids = user_ids
        self.reverse_umap = reverse_umap

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, index):
        user = self.test_users[index]
        seq = self.u2seq[user] + self.u2val[user]
        answer = self.u2answer[user][0]

        seq = seq[-self.max_len:]
        candidates = self.test_candidates[index]

        assert answer in candidates
        user_identifier = self.reverse_umap[user]
        user_id = self.user_ids[user_identifier]

        return seq_to_token_ids(self.args, seq, candidates, answer, self.text_dict, self.tokenizer, self.prompter,
                                user_id, mode='test')