import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from module.data_loader import DataLoader
from module.model import DisasterClassifier
from module.trainer import Trainer
from module.preprocess import save_cleaned_text, train_segmentizer

def define_argparse():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', type=str, required=True)
    p.add_argument('--file_path', type=str, required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--max_length', type=int, default=9999)
    # rnn
    p.add_argument('--embedding_dim', type=int, default=32)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--hidden_size', type=int, default=32)
    p.add_argument('--dropout', type=float, default=.2)

    config = p.parse_args()

    return config


def main(config):
    dataloader = DataLoader()
    train_loader, valid_loader = dataloader.get_loaders(config, config.file_path)

    vocab = dataloader.text.vocab
    label = dataloader.label.vocab
    vocab_size = len(vocab)
    n_classes = len(label)

    model = DisasterClassifier(
        input_size=vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        n_classes=n_classes
    )

    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(config)
    trainer.train(model, optimizer, loss, train_loader, valid_loader)

    torch.save({
        'model': model.state_dict(),
        'config': config,
        'vocab': vocab,
        'label': label,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparse()
    main(config)