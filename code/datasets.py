import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torchtext.utils import extract_archive
from tqdm import tqdm
import io


class German2EnglishDataFactory:
    def __init__(self, batch_size: int = 128):
        data_base = '.data/'
        train_file_names = ('train.de.gz', 'train.en.gz')
        val_file_names = ('val.de.gz', 'val.en.gz')
        test_file_names = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

        train_filepaths = [extract_archive(data_base + file_name)[0] for file_name in train_file_names]
        val_filepaths = [extract_archive(data_base + file_name)[0] for file_name in val_file_names]
        test_filepaths = [extract_archive(data_base + file_name)[0] for file_name in test_file_names]

        self.batch_size = batch_size

        self.encoder_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.decoder_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        self.encoder_vocab = self.build_vocab(train_filepaths[0], self.encoder_tokenizer)
        self.decoder_vocab = self.build_vocab(train_filepaths[1], self.decoder_tokenizer)
        self.encoder_vocab.set_default_index(self.encoder_vocab['<unk>'])
        self.decoder_vocab.set_default_index(self.decoder_vocab['<unk>'])

        train_data = self.data_process(train_filepaths)
        val_data = self.data_process(val_filepaths)
        test_data = self.data_process(test_filepaths)

        self.train_data = sorted(train_data, key=lambda pair: max(len(pair[0]), len(pair[1])))
        self.val_data = sorted(val_data, key=lambda pair: max(len(pair[0]), len(pair[1])))
        self.test_data = sorted(test_data, key=lambda pair: max(len(pair[0]), len(pair[1])))

    def get_decoder_pad_token(self):
        return self.decoder_vocab['<pad>']

    def get_encoder_pad_token(self):
        return self.encoder_vocab['<pad>']

    def get_encoder_vocab_size(self):
        return len(self.encoder_vocab)

    def get_decoder_vocab_size(self):
        return len(self.decoder_vocab)

    def get_encoder_vocab(self):
        return self.encoder_vocab

    def get_decoder_vocab(self):
        return self.decoder_vocab

    def get_decoder_tokenizer(self):
        return self.decoder_tokenizer

    def get_encoder_tokenizer(self):
        return self.encoder_tokenizer

    @staticmethod
    def build_vocab(filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_.strip("\n")))
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        return vocab(ordered_dict, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    def data_process(self, filepaths):
        raw_encoder_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_decoder_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_encoder, raw_decoder) in zip(raw_encoder_iter, raw_decoder_iter):
            encoder_tokens_ = [self.encoder_vocab[token] for token in self.encoder_tokenizer(raw_encoder.strip("\n"))]
            decoder_tokens_ = [self.decoder_vocab[token] for token in self.decoder_tokenizer(raw_decoder.strip("\n"))]
            data.append((encoder_tokens_, decoder_tokens_))
        return data

    def create_decoder_cross_attention_mask(self, decoder_tokens: torch.Tensor, encoder_tokens: torch.Tensor):
        n_tokens_decoder = len(decoder_tokens)
        n_tokens_encoder = len(encoder_tokens)
        mask = torch.ones(n_tokens_decoder, n_tokens_encoder).bool() & (encoder_tokens != self.encoder_vocab['<pad>'])
        return mask

    def create_decoder_self_attention_mask(self, decoder_tokens: torch.Tensor):
        n_tokens = len(decoder_tokens)
        mask = torch.tril(torch.ones(n_tokens, n_tokens)).bool() & (decoder_tokens != self.decoder_vocab['<pad>'])
        return mask

    def create_encoder_self_attention_mask(self, encoder_tokens: torch.Tensor):
        n_tokens = len(encoder_tokens)
        mask = torch.ones(n_tokens, n_tokens).bool() & (encoder_tokens != self.encoder_vocab['<pad>'])
        return mask

    def generate_batch(self, raw_pairs_batch):
        ENCODER_PAD_IDX = self.encoder_vocab['<pad>']
        ENCODER_SOS_IDX = self.encoder_vocab['<bos>']
        ENCODER_EOS_IDX = self.encoder_vocab['<eos>']

        DECODER_PAD_IDX = self.decoder_vocab['<pad>']
        DECODER_SOS_IDX = self.decoder_vocab['<bos>']
        DECODER_EOS_IDX = self.decoder_vocab['<eos>']

        max_tokens_encoder = max(
            [len(pair[0]) for pair in raw_pairs_batch]) + 2  # We add 2 because we will add the BOS and EOS tokens
        max_tokens_decoder = max(
            [len(pair[1]) for pair in raw_pairs_batch]) + 2  # We add 2 because we will add the BOS and EOS tokens

        encoder_input_batch = []
        decoder_input_batch = []
        target_batch = []
        encoder_self_attention_mask_batch = []
        decoder_self_attention_mask_batch = []
        decoder_cross_attention_mask_batch = []

        # encoder de, decoder er

        for (encoder_tokens, decoder_tokens) in raw_pairs_batch:
            # Build encoder input
            left_padding_amount = 0
            right_padding_amount = max_tokens_encoder - (len(encoder_tokens) + 2)
            encoder_input_tokens = pad(
                torch.tensor([ENCODER_SOS_IDX] + encoder_tokens + [ENCODER_EOS_IDX]).long(),
                (left_padding_amount, right_padding_amount),
                value=ENCODER_PAD_IDX
            )
            encoder_self_attention_mask = self.create_encoder_self_attention_mask(encoder_input_tokens)

            # Build decoder input
            left_padding_amount = 0
            right_padding_amount = max_tokens_decoder - (len(decoder_tokens) + 2)
            decoder_input_tokens = pad(
                torch.tensor([DECODER_SOS_IDX] + decoder_tokens + [DECODER_EOS_IDX]).long(),
                (left_padding_amount, right_padding_amount),
                value=DECODER_PAD_IDX
            )
            decoder_self_attention_mask = self.create_decoder_self_attention_mask(decoder_input_tokens)
            decoder_cross_attention_mask = self.create_decoder_cross_attention_mask(decoder_input_tokens,
                                                                                    encoder_input_tokens)

            # Build decoder target output
            left_padding_amount = 0
            right_padding_amount = max_tokens_decoder - (len(decoder_tokens) + 1)
            target_tokens = pad(
                torch.tensor(decoder_tokens + [DECODER_EOS_IDX]).long(),
                (left_padding_amount, right_padding_amount),
                value=DECODER_PAD_IDX
            )

            encoder_input_batch.append(encoder_input_tokens)
            decoder_input_batch.append(decoder_input_tokens)
            target_batch.append(target_tokens)
            encoder_self_attention_mask_batch.append(encoder_self_attention_mask)
            decoder_self_attention_mask_batch.append(decoder_self_attention_mask)
            decoder_cross_attention_mask_batch.append(decoder_cross_attention_mask)

        return {
            "encoder_input": torch.stack(encoder_input_batch),
            "decoder_input": torch.stack(decoder_input_batch),
            "encoder_self_attention_mask": torch.stack(encoder_self_attention_mask_batch),
            "decoder_self_attention_mask": torch.stack(decoder_self_attention_mask_batch),
            "decoder_cross_attention_mask": torch.stack(decoder_cross_attention_mask_batch),
            "target": torch.stack(target_batch)
        }

    def get_data(self):
        train_iter = DataLoader(self.train_data, batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.generate_batch)
        valid_iter = DataLoader(self.val_data, batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.generate_batch)
        test_iter = DataLoader(self.test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        return train_iter, valid_iter, test_iter

    def _get_info(self, data_loader: DataLoader, description: str = ""):
        counter = Counter()
        for batch_dict in tqdm(data_loader, desc=description):
            target_batch = batch_dict["target"]
            assert len(target_batch.shape) == 2  # batch_dim, n_tokens
            counter.update(target_batch.numpy().flatten())
        assert self.get_decoder_pad_token() in counter
        assert self.get_decoder_vocab()["<bos>"] not in counter  # There shouldn't be any <bos> token in the decoder's target!  # noqa
        counter[self.get_decoder_pad_token()] = 0  # The <pad> will be ignored during training, we don't need to count them to compute the weights.  # noqa
        total_number_of_samples = counter.total()  # noqa
        weights = torch.zeros(self.get_decoder_vocab_size())
        for c in range(self.get_decoder_vocab_size()):
            if c in counter.keys():
                weights[c] = total_number_of_samples / (counter[c] * len(counter.keys()))\
                    if c != self.get_decoder_pad_token() else 0.0
        return {"weights": weights, "n_samples": total_number_of_samples}

    def get_training_info(self):
        train_iter = DataLoader(self.train_data, batch_size=1,
                                shuffle=True, collate_fn=self.generate_batch)
        return self._get_info(train_iter, "Retrieving training weights and total number of samples...")

    def get_validation_info(self):
        train_iter = DataLoader(self.val_data, batch_size=1,
                                shuffle=True, collate_fn=self.generate_batch)
        return self._get_info(train_iter, "Retrieving training weights and total number of samples...")
