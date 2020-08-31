# DeToxify Korean Hate Speech
Provides **hate speech detection model** trained on [`kocohub/korean-hate-speech` dataset](https://github.com/kocohub/korean-hate-speech). <br>

model | F1 score
-- | --
[`kcbert-base`](https://github.com/Beomi/KcBERT) finetuned | **0.6042**
[`kobert-base`](https://github.com/SKTBrain/KoBERT) finetuned | 0.5390

- Check out [Kaggle Leaderboard](https://www.kaggle.com/c/korean-hate-speech-detection/leaderboard) to see more scores!

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
- `config` specifies model configuration filepath. e.g., [`configs/kcbert-base.yaml`](configs/kcbert-base.yaml)
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

## Example

1. Suppose we have `example.txt` like below:
```
파이콘 너무 좋아요!
인간적으로 손예진 너무 예쁘다 ㅠㅠ

나는 전라도에서 왔어 ㅎㅎ
이니수엘라보다 못한 이니전라수앨라가 되겠죠
이니스프리

내가 어떻게 하면 너의 기운을 북돋아 줄 수 있을까?
지금도 북극에선 얼음이 녹고 있다
남한과 북한의 관계는 참 어렵다
사랑의 불시착은 북한에 떨어진 남한 여자의 이야기다
우리집은북향이야
그저 북한북한북한 아주 정은이 똥꼬 빨아주느라 정신이 없구나
ㅋㅋ어후 대깨들 아주 신나서 난리났네 저기 북한 의료인 지원 법안 발의 기사엔 코빼기도 안비치고 ㅋㅋ

시험 망했어 제길!!
한남
한녀
저런 거 찍어주는 개돼지들이 있어서 이 나라는 노답이다
영길이 너 이리오너라. 게이 소개시켜줄께.둘이 더듬으면서 행복하게 살아볼래?
ㅋ 쿠팡 디졌다 수익좀 날라니까 네이버가 왔네 대형마트 강제휴무좀 풀면서 인터넷슈퍼하든가 해라 대형슈퍼만 죽일일있냐 강제휴무 없애고 하든가 ㅉㅉ 동내상권살린다면서 인터넷은 규제도없고 뭐임이게
네이버 주문만 받고- (홈플, 이마트, 롯데) 애들이 배송하는데 또 뭐? 4만원 이상 구입시 무배아니냐? 최저가 해도 쿠팡보다 비싸고 그리고 로켓배송 안되잖아? 그럼 짜져있어 얼마나 버티려고ㅋㅋ
네이버 욕심이 과하네 써글 시장도 나눠먹어야 질서가 유지되는데 다 해쳐드실라고?
```

2. Run fine-tuned `kcbert-base`model
```
$ python predict.py --config configs/kcbert-base.yaml --filepath example.txt --save
```

3. View `results/example.txt.kcbert-base.predict`
```
comments    prediction
파이콘 너무 좋아요!!    none
인간적으로 손예진 너무 예쁘다 ㅠㅠ  none

나는 전라도에서 왔어 ㅎㅎ   none
이니수엘라보다 못한 이니전라수앨라가 되겠죠 hate
이니스프리  none

내가 어떻게 하면 너의 기운을 북돋아 줄 수 있을까?   offensive
지금도 북극에선 얼음이 녹고 있다    none
남한과 북한의 관계는 참 어렵다  none
사랑의 불시착은 북한에 떨어진 남한 여자의 이야기다  offensive
우리집은북향이야    none
ㅋㅋ어후 대깨들 아주 신나서 난리났네 저기 북한 의료인 지원 법안 발의 기사엔 코빼기도 안비치고 ㅋㅋ  hate
그저 북한북한북한 아주 정은이 똥꼬 빨아주느라 정신이 없구나 hate

시험 망했어 제길!!  offensive
한남    hate
한녀    hate
저런 거 찍어주는 개돼지들이 있어서 이 나라는 노답이다   hate
영길이 너 이리오너라. 게이 소개시켜줄께.둘이 더듬으면서 행복하게 살아볼래?  hate
ㅋ 쿠팡 디졌다 수익좀 날라니까 네이버가 왔네 대형마트 강제휴무좀 풀면서 인터넷슈퍼하든가 해라 대형슈퍼만 죽일일있냐 강제휴무 없애고 하든가 ㅉㅉ 동내상권살린다면서 인터넷은 규제도없고 뭐임이게 offensive
네이버 주문만 받고- (홈플, 이마트, 롯데) 애들이 배송하는데 또 뭐? 4만원 이상 구입시 무배아니냐? 최저가 해도 쿠팡보다 비싸고 그리고 로켓배송 안되잖아? 그럼 짜져있어 얼마나 버티려고ㅋㅋ hate
네이버 욕심이 과하네 써글 시장도 나눠먹어야 질서가 유지되는데 다 해쳐드실라고?  offensive
```
