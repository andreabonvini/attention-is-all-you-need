import os

import numpy as np

from code import Transformer
from tqdm import tqdm
from code.datasets import German2EnglishDataFactory
import argparse
import json
from typing import Dict
import torch
from datetime import datetime

from torch.optim.lr_scheduler import LambdaLR


def get_current_lr(
        d_model: int,  # noqa
        step_num: int,
        warmup_steps: int
) -> float:
    # From section 5.3: "We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10−9."
    # "We varied the learning rate over the course of training, according to the formula:
    #         lrate=d_model^(−0.5) x min(step_num^(−0.5), step_num x warmup_steps^(−1.5))
    # This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
    # and decreasing it thereafter proportionally to the inverse square root of the step number.
    # We used warmup_steps = 4000"
    step_num = max(step_num, 1)  # just to avoid 0^(-0.5)
    return d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))


# TODO: UNDERSTAND IF YOU NEED <eos> token in decoder input during training

def get_config() -> Dict:
    parser = argparse.ArgumentParser(description='Script to train a Transformer model.')
    parser.add_argument('-c', '--config_path', required=True)
    args = parser.parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    return config


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_dict = get_config()
    n_epochs = config_dict["training"]["n_epochs"]

    data_source = German2EnglishDataFactory(batch_size=config_dict["dataset"]["batch_size"])
    args = config_dict["transformer_params"]
    args.update(
        {
            "encoder_vocabulary_dimension": data_source.get_encoder_vocab_size(),
            "decoder_vocabulary_dimension": data_source.get_decoder_vocab_size(),
            "max_tokens": 2048,
        }
    )

    checkpoint_folder = config_dict["training"]["checkpoint_folder"]
    checkpoint_file = config_dict["training"]["checkpoint_file"]
    os.makedirs(checkpoint_folder, exist_ok=True)
    train_loss_list_path = os.path.join(checkpoint_folder, "train_loss_list.txt")
    val_loss_list_path = os.path.join(checkpoint_folder, "val_loss_list.txt")

    # TODO: add consistency check for model parameters...
    if checkpoint_file is not None:
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        assert os.path.isfile(checkpoint_path)
        assert os.path.isfile(train_loss_list_path)
        assert os.path.isfile(val_loss_list_path)
        transformer = torch.load(checkpoint_path)
        start_epoch = int(checkpoint_file.split(".")[0].split("_")[-1])
        assert start_epoch < n_epochs
        print(f"===> Restarting from epoch {start_epoch} ...")
    else:
        transformer = Transformer(**args).to(device)
        print(f"===> Creating new Transformer model ...")
        start_epoch = 0

    transformer = torch.compile(transformer) # Buggy OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.  # noqa

    train_iter, val_iter, test_iter = data_source.get_data()

    # From section 5.4: "During training, we employed label smoothing of value ε=0.1 .
    # This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=data_source.get_training_weights().to(device),
        ignore_index=data_source.get_decoder_pad_token(),
        label_smoothing=0.1
    )
    # Note that in PyTorch CrossEntropyLoss already computed softmax, this is why we didn't include it in our
    # Transformer definition!

    # From section 5.3: "We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10−9."
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    d_model = config_dict["transformer_params"]["embedding_dimension"]
    lr_scheduler = LambdaLR(
        optimizer=optimizer, lr_lambda=lambda step: get_current_lr(d_model=d_model, step_num=step, warmup_steps=4000)
    )
    # IMPORTANT: Note that the way we schedule the learning rate here is fundamental to assure
    # that the model learns properly. We increase the learning rate linearly for 4000 epochs and then we decrease it
    # exponentially.

    train_loss_list = []
    val_loss_list = []

    best_val_loss = + np.inf

    for epoch_index in range(start_epoch, n_epochs):

        # ==== Start Training Loop ====
        transformer.train()
        for batch_dict in tqdm(train_iter):
            # Unpack dictionary
            encoder_input_batch = batch_dict["encoder_input"].to(device)
            decoder_input_batch = batch_dict["decoder_input"].to(device)
            encoder_self_attention_mask = batch_dict["encoder_self_attention_mask"].to(device)
            decoder_self_attention_mask = batch_dict["decoder_self_attention_mask"].to(device)
            decoder_cross_attention_mask = batch_dict["decoder_cross_attention_mask"].to(device)
            target_batch = batch_dict["target"].to(device)

            optimizer.zero_grad()
            out = transformer(
                encoder_input_tokens=encoder_input_batch,
                decoder_input_tokens=decoder_input_batch,
                encoder_self_attention_mask=encoder_self_attention_mask,
                decoder_self_attention_mask=decoder_self_attention_mask,
                decoder_cross_attention_mask=decoder_cross_attention_mask
            )
            assert out.size(-1) == data_source.get_decoder_vocab_size()
            loss = loss_fn(out.view(-1, data_source.get_decoder_vocab_size()), target=target_batch.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss_list.append(loss.item())

        mean_train_loss = np.mean(train_loss_list)
        print(f"Train Loss @ epoch {str(epoch_index).zfill(3)}: {mean_train_loss}")
        with open(train_loss_list_path, "a") as f:
            f.write(f"{str(epoch_index).zfill(4)}: {mean_train_loss}\n")
        train_loss_list = []

        # ==== Start Validation Loop ====

        transformer.eval()
        for batch_dict in tqdm(val_iter):
            # Unpack dictionary
            encoder_input_batch = batch_dict["encoder_input"].to(device)
            decoder_input_batch = batch_dict["decoder_input"].to(device)
            encoder_self_attention_mask = batch_dict["encoder_self_attention_mask"].to(device)
            decoder_self_attention_mask = batch_dict["decoder_self_attention_mask"].to(device)
            decoder_cross_attention_mask = batch_dict["decoder_cross_attention_mask"].to(device)
            target_batch = batch_dict["target"].to(device)

            with torch.no_grad():
                out = transformer(
                    encoder_input_tokens=encoder_input_batch,
                    decoder_input_tokens=decoder_input_batch,
                    encoder_self_attention_mask=encoder_self_attention_mask,
                    decoder_self_attention_mask=decoder_self_attention_mask,
                    decoder_cross_attention_mask=decoder_cross_attention_mask
                )
                val_loss = loss_fn(out.view(-1, data_source.get_decoder_vocab_size()), target=target_batch.view(-1))
                val_loss_list.append(val_loss.item())

        mean_val_loss = np.mean(val_loss_list)
        with open(val_loss_list_path, "a") as f:
            f.write(f"{str(epoch_index).zfill(4)}: {mean_val_loss}\n")
        if mean_val_loss < best_val_loss:
            print(f"[NEW BEST MODEL!] Validation Loss @ epoch {str(epoch_index).zfill(4)}: {mean_val_loss}")
            best_val_loss = mean_val_loss
            # Save model checkpoint
            checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_{str(epoch_index).zfill(4)}.pt")
            torch.save(transformer, checkpoint_path)
            print(f"Model saved @ {checkpoint_path}")
        else:
            print(f"Validation Loss @ epoch {str(epoch_index).zfill(4)}: {mean_val_loss}")
        val_loss_list = []
