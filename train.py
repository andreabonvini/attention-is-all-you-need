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


"""
# based on what Umar says we have
# input_decoder = <start> a   b    c   <pad> <pad>
# target        =    a    b   c  <end> <pad> <pad>

# based on what Annotated transformer implementation we have
# input_decoder = <start> a   b    c   <end> <pad> <pad>
# target        =    a    b   c  <end> <pad> <pad> <pad>
pt_optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-2)
pt_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_source.get_decoder_pad_token())
pt_optimizer.zero_grad()
pt_transformer = PyTorchTransformer(
    encoder_vocabulary_dimension=data_source.get_encoder_vocab_size(),
    decoder_vocabulary_dimension=data_source.get_decoder_vocab_size(),
    max_tokens=3000,
    d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
    dropout=0.1, batch_first=True
)
pt_loss = pt_loss_fn(pt_out.view(-1, data_source.get_decoder_vocab_size()), target=target_batch.view(-1))
pt_loss.backward()
"""


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

    data_source = German2EnglishDataFactory(batch_size=32)
    args = config_dict["transformer_params"]
    args.update(
        {
            "encoder_vocabulary_dimension": data_source.get_encoder_vocab_size(),
            "decoder_vocabulary_dimension": data_source.get_decoder_vocab_size(),
            "max_tokens": 3000,
        }
    )
    transformer = Transformer(**args).to(device)
    # compiled_transformer = torch.compile(transformer) # Buggy OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.  # noqa

    train_iter, val_iter, test_iter = data_source.get_data()

    n_epochs = 2000

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_source.get_decoder_pad_token())
    # Note that in PyTorch CrossEntropyLoss already computed softmax, this is why we didn't include it in our
    # Transformer definition!
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-2)

    train_loss_list = []
    val_loss_list = []
    checkpoint_folder = f"checkpoints/{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pt")
    train_loss_list_path = os.path.join(checkpoint_folder, "train_loss_list.txt")
    val_loss_list_path = os.path.join(checkpoint_folder, "val_loss_list.txt")
    best_val_loss = + np.inf

    for epoch_index in range(n_epochs):

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
            encoder_input_batch = batch_dict["encoder_input"]
            decoder_input_batch = batch_dict["decoder_input"]
            encoder_self_attention_mask = batch_dict["encoder_self_attention_mask"]
            decoder_self_attention_mask = batch_dict["decoder_self_attention_mask"]
            decoder_cross_attention_mask = batch_dict["decoder_cross_attention_mask"]
            target_batch = batch_dict["target"]

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
        with open(val_loss_list_path,"a") as f:
            f.write(f"{str(epoch_index).zfill(4)}: {mean_val_loss}\n")
        if mean_val_loss < best_val_loss:
            print(f"[NEW BEST MODEL!] Validation Loss @ epoch {str(epoch_index).zfill(4)}: {mean_val_loss}")
            best_val_loss = mean_val_loss
            # Save model checkpoint
            torch.save(transformer, checkpoint_path)
        else:
            print(f"Validation Loss @ epoch {str(epoch_index).zfill(4)}: {mean_val_loss}")
        val_loss_list = []


