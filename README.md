# 인공지능 스터디 (with 문겸)

## 스터디 진행상황
| 회차 | 날짜        | 스터디 내용                                  |
|------|-------------|--------------------------------------------|
| 0    | 2023. 12. 27 | 타이타닉 생존자 예측                        |
| 1    | 2024. 01. 07 | VS Code 개발 환경 / CIFAR-10 데이터셋 분류 CNN Model 설계 |
| 1.5    | 2024. 01. 07 | 깃헙 익히기 [@eastshine2741] (https://www.github.com/eastshine2741) 다녀감 |
| 2    | 2024. 01. 17 | CIFAR-10 데이터셋 분류 모델 개선 / 모델 시각화 |


### CIFAR-10 데이터셋

#### 첫 번째 모델

전처리 - 픽셀값 정규화

| Layer | Output Shape  | Description |
|-------|---------------|-------------|
| Conv1  | 32 * 32 * 16  | Filters: (2 * 2 * 3) * 16 / Activation: ReLU|
| MaxPool | 16 * 16 * 16 |             |
| Conv2 | 16 * 16 * 32 | Filters: (2 * 2 * 16) * 32 / Activation: ReLU|
| MaxPool | 8 * 8 * 32 |            |
| Conv3 | 8 * 8 * 64 | Filters: (2 * 2 * 32) * 64 / Activation: ReLU|
| MaxPool | 4 * 4 * 64 |            |
| Dropout | 4 * 4 * 64   | 훈련 시 Layer의 30%의 뉴런을 무작위로 비활성화 | 
| Flatten | 1024  |           |
| FC1   |  500    | Activation: ReLU    |
| Dropout |  500  | 훈련 시 Layer의 40%의 뉴런을 무작위로 비활성화 |
| FC2   | 10     | Activation: SoftMax   | 

Test Accuracy: 0.6965000033378601 (2024-01-07 21:58)

특이사항 - 개와 고양이를 잘 구분 못 함

#### 두 번째 모델

전처리 - 픽셀값 정규화

| Layer | Output Shape  | Description |
|-------|---------------|-------------|
| Conv1  | 32 * 32 * 16  | Filters: (3 * 3 * 3) * 16 / Activation: ReLU|
| Conv2 | 32 * 32 * 32 | Filters: (3 * 3 * 16) * 32 / Activation: ReLU|
| MaxPool | 16 * 16 * 32 |            |
| Conv3 | 16 * 16 * 64 | Filters: (3 * 3 * 32) * 64 / Activation: ReLU|
| MaxPool | 8 * 8 * 64 |            |
| Conv4 | 8 * 8 * 128 | Filters: (3 * 3 * 64) * 128 / Activation: ReLU|
| Conv5 | 8 * 8 * 64 | Filters: (3 * 3 * 128) * 64 / Activation: ReLU|
| Dropout | 4 * 4 * 64   | 훈련 시 Layer의 30%의 뉴런을 무작위로 비활성화 | 
| Flatten | 1024  |           |
| FC  |  10    | Activation: SoftMax   |

Test Accuracy: 0.7813000082969666 (2024-01-17 19:05)

특이사항

1. Parameter 개수 약 18만 개, 층을 더 많이 쌓아도 될 듯
2. 개와 고양이 여전히 잘 구분 못 함
3. 틀린 이미지를 보면 이미지 내에 불필요한 부분들이 존재. 전처리에도 신경을 써야할 듯함.