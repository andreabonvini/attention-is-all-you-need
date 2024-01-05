import argparse
from code.datasets import German2EnglishDataFactory
import torch


def get_checkpoint_path() -> str:
    parser = argparse.ArgumentParser(description='Script to test a trained German2English Transformer model.')
    parser.add_argument('-ckp', '--checkpoint_path', required=True)
    args = parser.parse_args()
    return args.checkpoint_path


if __name__ == "__main__":
    checkpoint_path = get_checkpoint_path()
    # test_sentence = "Ein Mann in gelbem Shirt und khakifarbenen Shorts balanciert auf einer Metallkette"
    # "A man in a yellow shirt and khaki shorts balances on a metal chain"
    data_factory = German2EnglishDataFactory(batch_size=1)
    encoder_vocab = data_factory.get_encoder_vocab()
    decoder_vocab = data_factory.get_decoder_vocab()
    encoder_tokenizer = data_factory.get_encoder_tokenizer()

    _, _, test_iter = data_factory.get_data()
    N = 10
    n = 0
    for batch_dict in test_iter:
        if n == N:
            break
        test_sentence_tokens = batch_dict["encoder_input"][0]
        test_sentence = " ".join([
            encoder_vocab.get_itos()[t] for t in test_sentence_tokens
            if t not in (encoder_vocab["<bos>"], encoder_vocab["<eos>"], encoder_vocab["<pad>"])
        ])
        if "<unk>" in test_sentence:
            continue

        target_sentence_tokens = batch_dict["target"][0]
        target_sentence = " ".join([
            decoder_vocab.get_itos()[t] for t in target_sentence_tokens
            if t not in (decoder_vocab["<bos>"], decoder_vocab["<eos>"], decoder_vocab["<pad>"])
        ])
        transformer_model = torch.load(checkpoint_path, map_location="cpu")

        print("Running inference...")
        translated_sentence = transformer_model.run_inference(
            encoder_sentence=test_sentence,  # noqa
            encoder_tokenizer=encoder_tokenizer,
            encoder_vocab=encoder_vocab,
            decoder_vocab=decoder_vocab,
            max_decoder_length=32
        )

        test_sentence = test_sentence.replace('\n', '\\n')
        target_sentence = target_sentence.replace('\n', '\\n')  # noqa
        print(f"ORIGINAL SENTENCE: '{test_sentence}'")
        print(f"ORIGINAL TRANSLATION: '{target_sentence}'")
        print("OUTPUT ==> '", translated_sentence, "'")
        print("======================================")
        n += 1
