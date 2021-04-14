import re
import os
import argparse

import sentencepiece as spm

import pandas as pd


def define_argparse():
    p = argparse.ArgumentParser()

    p.add_argument('--file_path', type=str, required=True)
    p.add_argument('--name', type=str, required=True)
    p.add_argument('--is_train', action='store_true')

    config = p.parse_args()

    return config


def train_segmentizer(name, num_vocab=3500, max_length=999999) :
    pat = re.compile('\s+')

    df = pd.read_csv('train_cleaned.tsv', sep='\t', encoding='utf-8', header=None)
    df.iloc[:, 1] = [re.sub(pat, ' ', text) for text in train_df.iloc[:, 1]]
    df.iloc[:, 1] = df.iloc[:, 1].str.replace(' ', '▁')

    with open('{}_feed.txt'.format(name), 'w', -1, encoding='utf-8') as f :
        f.write('\n'.join(df.iloc[:, 1]))

    # segmentize
    spm.SentencePieceTrainer.Train(
        input='{}_feed.txt'.format(name),
        model_prefix='{}'.format(name),
        vocab_size=num_vocab,
        model_type='bpe',
        max_sentence_length=max_length,
        character_coverage=0.9995
    )

    sp = spm.SentencePieceProcessor(model_file='{}.model'.format(name))

    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x : ' '.join(sp.encode_as_pieces(x)))
    df.to_csv('train_segmented.tsv', sep='\t', index=False, header=None)

    os.remove('{}_feed.txt'.format(name))

    return df


def test_segmentizer(name):
    pat = re.compile('\s+')

    df = pd.read_csv('test.tsv', sep='\t', encoding='utf-8', header=None)
    df = [re.sub(pat, ' ', text) for text in df]
    df = df.str.replace(' ', '▁')

    sp = spm.SentencePieceProcessor(model_file='{}.model'.format(name))

    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x : ' '.join(sp.encode_as_pieces(x)))
    df.to_csv('test_segmented.tsv', sep='\t', index=False, header=None)

    return df


def save_cleaned_text(path, is_train=True, format='csv'):
    if format == 'csv':
        df = pd.read_csv(path, sep=',')

    if not is_train:
        df = df.iloc[:, -1]

    df= df.str.replace('\t', ' ')
    df= df.str.replace('\n', ' ')
    df= df.str.replace('http(s|)://[a-zA-Z0-9/.]*', '')
    df= df.str.replace("[^a-zA-Z0-9\s~!@#$%^&*(),.;\'\"\-:|]+", '')
    df= df.str.lower()
    df= pd.DataFrame([text.strip() for text in df])

    df.to_csv('{}.tsv'.format('train' if is_train else 'test'), sep='\t', index=False, header=False)



def reverse_sentence(df) :
    sents = df.iloc[:, 1]
    sents = sents.str.replace(' ', '')
    sents = sents.str.replace('▁', ' ')
    df.iloc[:, 1] = sents

    return df

if __name__ == '__main__':
    config = define_argparse()
    if config.is_train:
        save_cleaned_text(config.file_path, config.is_train)
        train_segmentizer(config.name)

    else:
        save_cleaned_text(config.file_path, config.is_train)
        test_segmentizer(config.name)
