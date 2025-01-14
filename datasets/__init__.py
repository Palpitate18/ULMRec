from .beauty import BeautyDataset
from .games import GamesDataset

DATASETS = {
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
