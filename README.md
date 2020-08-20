# DeToxify Korean Hate Speech
Provides **hate speech detection model** trained on [`kocohub/korean-hate-speech` dataset](https://github.com/kocohub/korean-hate-speech). <br>
Additionally, `detox` supports users to easily train their own model. Note that only finetuning BERT is provided, yet.

## Prerequisite
- Python >= 3.6.9
- [`git lfs`](https://git-lfs.github.com/)

### Install packages and model checkpoints
```bash
$ pip install -r requirements.txt
$ git lfs pull
```

## Usage
### 1. Detecting hate speech
```
$ python predict.py --config <config_path> (--koco-test | --filepath <test_file_path>) [--save]
```
- `config` specifies model configuration filepath. e.g., [`configs/detox.yaml`](configs/detox.yaml)
- Either `koco-test` or `filepath` is required. 
  - `koco-test` uses [`kocohub/korean-hate-speech` testset](https://github.com/kocohub/korean-hate-speech) as an input.
  - `filepath` uses designated text file as an input. e.g., [`example/example.txt`](example/example.txt)
- `save` is an optional argument. If supplied, predicted results are saved in `./results` directory.

### 2. Training hate speech detection model
`NOTE`: Currently, `detox` only supports BERT finetuning model. However, any contributions are welcome! :tada:
```
$ python finetune_bert.py --config <config_path>
```
- `config` specifies model configuration filepath. e.g., [`configs/detox.yaml`](configs/detox.yaml)

### 3. Convert to [Kaggle](https://www.kaggle.com/c/korean-hate-speech-detection) submission format
```
$ python convert_to_kaggle_submission.py --result-path <result_path>
```
- `result_path` specifies model prediciton result filepath. e.g., [`results/example.txt.kcbert-base.predict`](results/example.txt.kcbert-base.predict)
