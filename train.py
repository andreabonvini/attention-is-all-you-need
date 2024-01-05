import os

import numpy as np
import warnings

from code import Transformer
from tqdm.auto import tqdm
from code.datasets import German2EnglishDataFactory
import argparse
import json
from typing import Dict
import torch

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
    print("USING DEVICE: ", device)
    config_dict = get_config()
    n_epochs = config_dict["training"]["n_epochs"]

    data_source = German2EnglishDataFactory(batch_size=config_dict["dataset"]["batch_size"])
    args = config_dict["transformer_params"]
    args.update(
        {
            "encoder_vocabulary_dimension": data_source.get_encoder_vocab_size(),
            "decoder_vocabulary_dimension": data_source.get_decoder_vocab_size()
        }
    )

    checkpoint_folder = config_dict["training"]["checkpoint_folder"]
    checkpoint_file = config_dict["training"]["checkpoint_file"]
    warmup_steps = config_dict["training"]["warmup_steps"]
    os.makedirs(checkpoint_folder, exist_ok=True)
    train_loss_list_path = os.path.join(checkpoint_folder, "train_loss_list.txt")
    val_loss_list_path = os.path.join(checkpoint_folder, "val_loss_list.txt")

    # TODO: add consistency check for model parameters...
    if checkpoint_file is not None:
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        assert os.path.isfile(checkpoint_path)
        assert os.path.isfile(train_loss_list_path)
        assert os.path.isfile(val_loss_list_path)
        state_dict = torch.load(checkpoint_path)
        start_epoch = state_dict['epoch']
        assert start_epoch < n_epochs
        print(f"===> Restarting from epoch {start_epoch} ...")
        checkpoint_transformer_args = state_dict['transformer_args']
        transformer_state_dict = state_dict['transformer_state_dict']
        optimizer_state_dict = state_dict['optimizer_state_dict']
        best_train_loss = state_dict['best_training_loss']
        best_val_loss = state_dict['best_validation_loss']
        if checkpoint_transformer_args != args:
            print(
                f"Warning: the model specified in {checkpoint_path} "
                f"has different parameters than the ones specified in the input configuration."
            )
        transformer = Transformer(**checkpoint_transformer_args)
        transformer.load_state_dict(transformer_state_dict)
        transformer = transformer.to(device)
    else:
        print(f"===> Creating new Transformer model ...")
        transformer = Transformer(**args).to(device)
        start_epoch = 0
        best_val_loss = + np.inf
        best_train_loss = + np.inf

    train_iter, val_iter, test_iter = data_source.get_data()
    train_info = data_source.get_training_info()
    val_info = data_source.get_validation_info()
    # From section 5.4: "During training, we employed label smoothing of value ε=0.1 .
    # This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=train_info["weights"].to(device),
        ignore_index=data_source.get_decoder_pad_token(),
        label_smoothing=0.1
    )

    loss_fn_val = torch.nn.CrossEntropyLoss(
        weight=val_info["weights"].to(device),
        ignore_index=data_source.get_decoder_pad_token()
    )
    # Note that in PyTorch CrossEntropyLoss already computed softmax, this is why we didn't include it in our
    # Transformer definition!
    # From section 5.3: "We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10−9."
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1, betas=(0.9, 0.98),
                                 eps=1e-9)  # fixme: is lr=1 as a starting point correct?  # noqa

    # fixme: why training loss and validation loss are that different during thre training prvess? e. g 21 vs 7
    if checkpoint_file is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: get_current_lr(
            d_model=transformer.embedding_dimension, step_num=step, warmup_steps=warmup_steps
        )
    )
    # IMPORTANT: Note that the way we schedule the learning rate here is fundamental to assure
    # that the model learns properly. We increase the learning rate linearly for 4000 epochs, and then we decrease it
    # exponentially.

    mean_train_loss = 0.0
    mean_val_loss = 0.0

    base_desc = "|| Train Loss: {:.4f} || Val Loss: {:.4f} || BEST Train Loss: {:.4f} || BEST Val Loss: {:.4f}".format(
        mean_train_loss, mean_val_loss, best_train_loss, best_val_loss
    )

    with tqdm(
            total=n_epochs,
            initial=start_epoch,
            position=0, leave=True,
            desc=base_desc
    ) as pbar:

        for epoch_index in range(start_epoch, n_epochs):

            # ==== Start Training Loop ====
            n = len(train_iter)
            i = 0
            transformer.train()
            for batch_dict in train_iter:
                i += 1
                pbar.set_description(base_desc + f" ==> (Training batch {i}/{n})")
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
                # todo: update step counter on pbar
                mean_train_loss = loss.item() * target_batch.size(0)

            # ==== Start Validation Loop ====

            transformer.eval()
            n = len(val_iter)
            i = 0
            for batch_dict in val_iter:
                i += 1
                pbar.set_description(base_desc + f" ==> (Validation batch {i}/{n})")
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
                    val_loss = loss_fn_val(out.view(-1, data_source.get_decoder_vocab_size()),
                                           target=target_batch.view(-1))
                    mean_val_loss = val_loss.item() * target_batch.size(0)

            # Compute mean losses and serialize
            mean_train_loss = mean_train_loss/train_info["n_samples"]
            mean_val_loss = mean_val_loss/val_info["n_samples"]

            if mean_train_loss < best_train_loss:
                best_train_loss = mean_train_loss
                checkpoint_path = os.path.join(checkpoint_folder, f"train_checkpoint.pt")
                torch.save(
                    {
                        'epoch': epoch_index,
                        'transformer_args': args,
                        'transformer_state_dict': transformer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_training_loss': best_train_loss,
                        'best_validation_loss': best_val_loss
                    },
                    checkpoint_path
                )

            with open(train_loss_list_path, "a") as f:
                f.write(f"{str(epoch_index).zfill(4)}: {mean_train_loss}\n")
            with open(val_loss_list_path, "a") as f:
                f.write(f"{str(epoch_index).zfill(4)}: {mean_val_loss}\n")

            if mean_val_loss < best_val_loss:
                print(f" <<[NEW BEST MODEL]>>")
                best_val_loss = mean_val_loss
                checkpoint_path = os.path.join(checkpoint_folder, f"val_checkpoint.pt")
                torch.save(
                    {
                        'epoch': epoch_index,
                        'transformer_args': args,
                        'transformer_state_dict': transformer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_training_loss': best_train_loss,
                        'best_validation_loss': best_val_loss
                    },
                    checkpoint_path
                )
            base_desc = "|| Train Loss: {:.4f} || Val Loss: {:.4f} || BEST Train Loss: {:.4f} || BEST Val Loss: {:.4f}".\
                format(
                    mean_train_loss, mean_val_loss, best_train_loss, best_val_loss
                )
            pbar.set_description(base_desc)
            pbar.update(1)
            mean_train_loss = 0.0
            mean_val_loss = 0.0
