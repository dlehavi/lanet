import torch

from train import Trainer
from config import get_config
from utils import prepare_dirs
from data_loader import get_data_loader
from datasets import pairs_loader

def main(config):
    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.seed)
    if config.use_gpu:
        torch.cuda.manual_seed(config.seed)
    if config.image_pairs_meta_data_csv_file != '':
        # instantiate train data loaders
        train_loader = get_data_loader(config=config)
    else:
        train_loader = pairs_loader.get_data_loader(config=config)

    trainer = Trainer(config, train_loader=train_loader)
    trainer.train()

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
