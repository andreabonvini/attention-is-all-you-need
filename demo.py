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
    test_sentence = "Ein Mann in gelbem Shirt und khakifarbenen Shorts balanciert auf einer Metallkette"
    data_factory = German2EnglishDataFactory()
    encoder_vocab = data_factory.get_encoder_vocab()
    decoder_vocab = data_factory.get_decoder_vocab()
    encoder_tokenizer = data_factory.get_encoder_tokenizer()

    transformer_model = torch.load(checkpoint_path, map_location="cpu")

    print("Running inference...")
    translated_sentence = transformer_model.run_inference(
        encoder_sentence=test_sentence,
        encoder_tokenizer=encoder_tokenizer,
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        max_decoder_length=32
    )

    print(f"Translation of '{test_sentence}'")
    print("==>", translated_sentence)

