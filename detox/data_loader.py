from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

conf = OmegaConf.load("configs/detox.yaml")


class KoreanHateSpeechDataset(Dataset):
    """Load input and output for Korean Hate Speech Detection.

    A format is customized to koco data loader. koco returns list of ditionary type dataset.
    For example:

    {'comments': '2,30대 골빈여자들은 이 기사에 다 모이는건가ㅋㅋㅋㅋ 이래서 여자는 투표권 주면 안된다. 엠넷사전투표나 하고 살아야지 계집들은',
     'contain_gender_bias': True,
     'bias': 'gender',
     'hate': 'hate',
     'news_title': '"“8년째 연애 중”…‘인생술집’ 블락비 유권♥전선혜, 4살차 연상연하 커플"',
     'label_index': 0}

    Attributes:
        texts: Comments
        biases: Bias labels
        labels: Integer type annotations from the specified labels (e.g., For hate, labels would be one of 0, 1, 2)
        contexts: News title
    """

    def __init__(self, koco_dataset):
        """Inits KoreanHateSpeechDataset with koco_dataset.

        Args:
            koco_dataset (list of dict): Dictionary type dataset containing comments, labels regarding bias and hate, and news titles.

        Returns:
            texts (list): A list of comments
            biases (list): Bias labels
            label_indices (list): A list of integer annotations from the bias labels
            contexts (list): A list of news titles
        """
        self.texts = [koco["comments"] for koco in koco_dataset]
        self.biases = [koco['bias'] for koco in koco_dataset]
        self.label_indices = [koco['label_index'] for koco in koco_dataset]
        self.contexts = [koco["news_title"] for koco in koco_dataset]
        assert len(self.texts) == len(self.biases) == len(self.label_indices) == len(self.contexts)

    def __len__(self):
        return len(self.label_indices)

    def __getitem__(self, item):
        text = self.texts[item]
        bias = self.biases[item]
        label_index = self.label_indices[item]
        context = self.contexts[item]
        return {"text": text, 'bias': bias, 'label_index': label_index, "context": context}


class KoreanHateSpeechCollator():
    """
    """

    def __init__(self, tokenizer, predict_hate_with_bias):
        """
        Args:
            tokenizer: tokenizer to tokenize dataset
            predict_hate_with_bias (bool):
        """
        self.tokenizer = tokenizer
        self.predict_hate_with_bias = predict_hate_with_bias

    def collate(self, data):
        """Collate tokenized input texts and integer converted labels.

        Args:
            data (list of dict): Data that contains input and output in dictionary type.
                                 For example,
                                 [
                                  {'text': '얼래?나경원 자는 왜 온겨?집에서서 애나보지!',
                                   'bias': 'gender',
                                   'label_index': 1,
                                   'context': '송해·이회창·김창숙·유승민···배우 신성일 조문행렬(종합)'},
                                  ...
                                 ]
        """
        texts, label_indices = [], []
        for d in data:
            text = d['text']
            if self.predict_hate_with_bias:
                bias_context = f'<{d["bias"]}>'
                text = f'{bias_context} {text}'
            label_index = d['label_index']
            texts.append(text)
            label_indices.append(label_index)

        encoded_texts = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=conf.train_hparams.max_length,
        )

        return {
            "text": texts,
            "input_ids": encoded_texts["input_ids"],
            "attention_mask": encoded_texts["attention_mask"],
            "label_indices": torch.tensor(label_indices, dtype=torch.long),
        }
