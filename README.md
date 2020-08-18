# DeToxify Korean Hate Speech
Provides training and inferencing **hatespeech detection model** based on [`kocohub/korean-hate-speech` dataset](https://github.com/kocohub/korean-hate-speech). <br>
Currently, only BERT-based finetuning model is supported.

## Prerequisite
```bash
$ pip install -r requirements.txt
```

## Train
```bash
$ python finetune_bert.py --config <config_path>
```

## Inference
```bash
$ python predict.py --config <config_path> [--save]
```

## Convert to Kaggle Submission format
```bash
$ python convert_to_kaggle_submission.py --config <config_path>
```

