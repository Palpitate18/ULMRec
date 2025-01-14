from datasets import dataset_factory

from .llm import *
from .utils import *


def dataloader_factory(args):
    dataset = dataset_factory(args)
    
    dataloader = LLMDataloader(args, dataset)
    
    test = dataloader.get_pytorch_dataloaders()
    tokenizer = dataloader.tokenizer
    test_retrieval = dataloader.test_retrieval
    return test, tokenizer, test_retrieval


def test_subset_dataloader_loader(args):
    dataset = dataset_factory(args)
    dataloader = LLMDataloader(args, dataset)

    return dataloader.get_pytorch_test_subset_dataloader()
