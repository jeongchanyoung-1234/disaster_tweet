import sys
import re
import argparse

import torch
from torchtext.legacy import data

from module.model import DisasterClassifier

def define_argparse():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', type=str, required=True)
    p.add_argument('--file_path', type=str, required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def open_file(train_config):
    lines = []
    with open(config.file_path, 'r', -1, 'utf-8') as f :
        linelist = f.readlines()
        for line in linelist :
            if line.strip() != '':
                lines.append(line.strip().split(' ')[:train_config.max_length])

    return lines


def main(config):
    saved_data = torch.load(config.model_fn, map_location='cpu' if config.gpu_id < 0 else 'cuda:{}'.format(config.gpu_id))

    model_dict = saved_data['model']
    train_config = saved_data['config']
    vocab = saved_data['vocab']
    label = saved_data['label']

    text_field = data.Field(batch_first=True,
                            unk_token='<unk>')
    label_field = data.Field(sequential=False)

    text_field.vocab = vocab
    label_field.vocab = label

    lines = open_file(train_config)

    with torch.no_grad():
        model = DisasterClassifier(input_size=len(vocab),
                                   embedding_dim=train_config.embedding_dim,
                                   num_layers=train_config.num_layers,
                                   hidden_size=train_config.hidden_size,
                                   dropout=train_config.dropout,
                                   n_classes=len(label))
        model.load_state_dict(model_dict)

        model.eval()

        y_hat = []
        for i in range(0, len(lines), config.batch_size):
            x = text_field.numericalize(
                text_field.pad(lines[i:i + config.batch_size]),
                device='cpu' if config.gpu_id < 0 else 'cuda:{}'.format(config.gpu_id)
            )
            y_hat.append(model(x).cpu())

        y_hat = torch.cat(y_hat, dim=0)

        probs, indices = torch.topk(y_hat, config.top_k, dim=-1)

        with open('{}_prediction.tsv'.format(config.model_fn[:-4]), 'w', -1, encoding='utf-8') as f :
            for i in range(len(lines)) :
                f.write('{}\t{}\n'.format(
                    ' '.join(label.itos[indices[i][j]] for j in range(config.top_k)),
                    ' '.join(lines[i])
                ))

if __name__ == '__main__':
    config = define_argparse()
    main(config)