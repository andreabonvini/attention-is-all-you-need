import os
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torchtext.utils import download_from_url, extract_archive
import io


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    return vocab(ordered_dict, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                                  dtype=torch.long)
        data.append((de_tensor_, en_tensor_))
    return data


if __name__ == "__main__":
    # From https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html

    os.system("python -m spacy download en_core_web_sm")
    os.system("python -m spacy download de_core_news_sm")

    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.de.gz', 'train.en.gz')
    val_urls = ('val.de.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

    for train_url, val_url, test_url in train_urls, val_urls, test_urls:
        download_from_url(url_base + train_url)
        download_from_url(url_base + val_url)
        download_from_url(url_base + test_url)



