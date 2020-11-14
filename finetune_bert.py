"""Finetune BERT variants

Supports KcBERT, KoBERT
"""
import argparse

import koco
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from kobert_transformers import get_tokenizer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from detox.bert_trainer import BertTrainer
from detox.data_loader import KoreanHateSpeechCollator, KoreanHateSpeechDataset
from utils import get_device_and_ngpus, makedirs, map_label2idx, set_seeds

device, n_gpus = get_device_and_ngpus()


def main(conf):
    # Prepare data
    train_dev = koco.load_dataset("korean-hate-speech", mode="train_dev")
    train, valid = train_dev["train"], train_dev["dev"]

    # Prepare tokenizer
    tokenizer = (
        get_tokenizer()
        if "kobert" in conf.pretrained_model
        else AutoTokenizer.from_pretrained(conf.pretrained_model)
    )
    if conf.tokenizer.register_names:
        names = pd.read_csv("entertainement_biographical_db.tsv", sep="\t")[
            "name_wo_parenthesis"
        ].tolist()
        tokenizer.add_tokens(names)

    # Mapping string y_label to integer label
    if conf.label.hate:
        train, label2idx = map_label2idx(train, "hate")
        valid, _ = map_label2idx(valid, "hate")
    elif conf.label.bias:
        train, label2idx = map_label2idx(train, "contain_gender_bias")
        valid, _ = map_label2idx(valid, "contain_gender_bias")

    # Use bias as an additional context for predicting hate
    if conf.label.hate and conf.label.bias:
        biases = ["gender", "others", "none"]
        tokenizer.add_tokens([f"<{label}>" for label in biases])

    # Prepare DataLoader
    train_dataset = KoreanHateSpeechDataset(train)
    valid_dataset = KoreanHateSpeechDataset(valid)
    collator = KoreanHateSpeechCollator(
        tokenizer, predict_hate_with_bias=(conf.label.hate and conf.label.bias)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.train_hparams.batch_size,
        shuffle=True,
        collate_fn=collator.collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=conf.train_hparams.batch_size,
        shuffle=False,
        collate_fn=collator.collate,
    )

    # Prepare model
    set_seeds(conf.train_hparams.seed)
    model = BertForSequenceClassification.from_pretrained(
        conf.pretrained_model, num_labels=len(label2idx)
    )
    if conf.tokenizer.register_names:
        model.resize_token_embeddings(len(tokenizer))
    elif conf.label.hate and conf.label.bias:
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=conf.train_hparams.lr,
        eps=conf.train_hparams.adam_epsilon,
    )

    n_total_iterations = len(train_loader) * conf.train_hparams.n_epochs
    n_warmup_steps = int(n_total_iterations * conf.train_hparams.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, n_warmup_steps, n_total_iterations
    )

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Train!
    trainer = BertTrainer(conf.train_hparams)
    model = trainer.train(
        model, criterion, optimizer, scheduler, train_loader, valid_loader
    )

    makedirs(conf.checkpoint_dir)
    makedirs(conf.log_dir)
    checkpoint_path = f"{conf.checkpoint_dir}/{conf.model_name}.pt"
    log_path = f"{conf.log_dir}/{conf.model_name}.log"
    torch.save({"model": model.state_dict()}, checkpoint_path)
    torch.save({"config": conf, "classes": label2idx, "tokenizer": tokenizer}, log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Config path", default="configs/detox.yaml", required=True
    )
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    main(conf)
