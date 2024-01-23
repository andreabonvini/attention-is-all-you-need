import argparse

from code import Transformer
from code.datasets import German2EnglishDataFactory
import torch

from code.transformer import PyTorchTransformer


def get_checkpoint_path() -> str:
    parser = argparse.ArgumentParser(description='Script to test a trained German2English Transformer model.')
    parser.add_argument('-ckp', '--checkpoint_path', required=True)
    args = parser.parse_args()
    return args.checkpoint_path


def compute_and_print_example(model, batch_dict):
    test_sentence_tokens = batch_dict["encoder_input"][0]
    test_sentence = " ".join([
        encoder_vocab.get_itos()[t] for t in test_sentence_tokens
        if t not in (encoder_vocab["<bos>"], encoder_vocab["<eos>"], encoder_vocab["<pad>"])
    ])
    if "<unk>" in test_sentence:
        print("<unk> detected. Skipping!")
        return

    target_sentence_tokens = batch_dict["target"][0]
    target_sentence = " ".join([
        decoder_vocab.get_itos()[t] for t in target_sentence_tokens
        if t not in (decoder_vocab["<bos>"], decoder_vocab["<eos>"], decoder_vocab["<pad>"])
    ])

    translated_sentence = model.run_inference(
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


if __name__ == "__main__":
    checkpoint_path = get_checkpoint_path()

    ckp = torch.load(checkpoint_path, map_location="cpu")

    train_loss_list = ckp["mean_train_loss_list"]
    val_loss_list = ckp["mean_val_loss_list"]
    import matplotlib.pyplot as plt

    plt.plot(train_loss_list, label="Training loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    use_pytorch_transformer_class = ckp["use_pytorch_transformer_class"]
    if use_pytorch_transformer_class:
        transformer_model = PyTorchTransformer(**ckp["transformer_args"])
    else:
        transformer_model = Transformer(**ckp["transformer_args"])
    transformer_model.load_state_dict(ckp["transformer_state_dict"])

    # test_sentence = "Ein Mann in gelbem Shirt und khakifarbenen Shorts balanciert auf einer Metallkette"
    # "A man in a yellow shirt and khaki shorts balances on a metal chain"
    data_factory = German2EnglishDataFactory(batch_size=1)
    encoder_vocab = data_factory.get_encoder_vocab()
    decoder_vocab = data_factory.get_decoder_vocab()
    encoder_tokenizer = data_factory.get_encoder_tokenizer()

    train_iter, _, test_iter = data_factory.get_data()
    N = 10
    n = 0
    for batch_dict_train, batch_dict_test in zip(train_iter, test_iter):
        if n == N:
            break
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~======== TRAINING EXAMPLE ==========~~~~~~")
        compute_and_print_example(transformer_model, batch_dict_train)
        print("~~~~~~======== TESTING EXAMPLE ==========~~~~~~~")
        compute_and_print_example(transformer_model, batch_dict_test)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        n += 1
