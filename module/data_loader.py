from torchtext.legacy import data

class DataLoader(object):
    def __init__(self, max_size=999999, min_freq=1):
        super().__init__()
        self.label = data.Field(sequential=False)
        self.text = data.Field(batch_first=True,
                               unk_token='<unk>')
        self.max_size = max_size
        self.min_freq = min_freq

    def get_loaders(self, config, file_path):
        train, valid = data.TabularDataset(
            path=file_path,
            format='tsv',
            fields=[('label', self.label), ('text', self.text)]
        ).split(split_ratio=config.train_ratio)

        train_loader, valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=config.batch_size,
            device='cuda:{}'.format(config.gpu_id) if config.gpu_id >= 0 else 'cpu',
            shuffle=True,
            sort_key = lambda x : len(x.text),
            sort_within_batch = True,
        )

        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=self.max_size, min_freq=self.min_freq)

        return train_loader, valid_loader