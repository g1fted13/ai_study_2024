# 인공지능 스터디 (with 문겸)

## 스터디 진행상황
| 회차 | 날짜        | 스터디 내용                                  |
|------|-------------|--------------------------------------------|
| 0    | 2023. 12. 27 | 타이타닉 생존자 예측                        |
| 1    | 2024. 01. 07 | VS Code 개발 환경 / CIFAR-10 데이터셋 분류 CNN Model 설계 |
| 1.5    | 2024. 01. 07 | 깃헙 익히기 [@eastshine2741] (https://www.github.com/eastshine2741) 다녀감 |
| 2    | 2024. 01. 17 | CIFAR-10 데이터셋 분류 모델 개선 / 모델 시각화 |
| 3    | 2024. 01. 29 | CIFAR-10 데이터셋 분류 모델 개선(optimizer/learning rate 조정) |

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
| MaxPool | 4 * 4 * 64 |            |
| Dropout | 4 * 4 * 64   | 훈련 시 Layer의 30%의 뉴런을 무작위로 비활성화 | 
| Flatten | 1024  |           |
| FC  |  10    | Activation: SoftMax   |

Test Accuracy: 0.7813000082969666 (2024-01-17 19:05)


특이사항

1. Parameter 개수 약 18만 개, 층을 더 많이 쌓아도 될 듯
2. 개와 고양이 여전히 잘 구분 못 함
3. 틀린 이미지를 보면 이미지 내에 불필요한 부분들이 존재. 전처리에도 신경을 써야할 듯함.

#### 세 번째 모델

전처리 - 픽셀값 정규화 + 데이터 증강(좌우 반전)

| Layer | Output Shape  | Description |
|-------|---------------|-------------|
| Conv1  | 32 * 32 * 32  | Filters: (3 * 3 * 3) * 32 / Activation: ReLU|
| Conv2 | 32 * 32 * 32 | Filters: (3 * 3 * 32) * 32 / Activation: ReLU|
| MaxPool | 16 * 16 * 32 |            |
| Conv3 | 16 * 16 * 64 | Filters: (3 * 3 * 32) * 64 / Activation: ReLU|
| Conv4 | 16 * 16 * 64 | Filters: (3 * 3 * 64) * 64 / Activation: ReLU|
| MaxPool | 8 * 8 * 64 |            |
| Conv5 | 8 * 8 * 128 | Filters: (3 * 3 * 64) * 128 / Activation: ReLU|
| Conv6 | 8 * 8 * 128 | Filters: (3 * 3 * 128) * 128 / Activation: ReLU|
| Conv7 | 8 * 8 * 128 | Filters: (3 * 3 * 128) * 128 / Activation: ReLU|
| MaxPool | 4 * 4 * 128 |            |
| Dropout | 4 * 4 * 128   | 훈련 시 Layer의 20%의 뉴런을 무작위로 비활성화 | 
| Flatten | 2028  |           |
| FC1   |  500    | Activation: ReLU    |
| Dropout |  500  | 훈련 시 Layer의 40%의 뉴런을 무작위로 비활성화 |
| FC2   | 10     | Activation: SoftMax   | 

Test Accuracy: 0.739799976348877 (2024-01-18 05:13)
Test Accuracy: 0.8113999962806702 (2024-01-29 18:53 (데이터 증강 + 3번째 모델 + 에포크 80 + optimizer SGD learning_rate 0.01 -> 0.001))

특이사항

1. Layer를 확 늘리고 전처리 과정에 데이터 증강도 포함했으나 오히려 정확도가 떨어짐
2. Parameter 개수 약 146만 개, 오히려 너무 Parameter 수가 많은 것이 문제일 수도
3. 이전 모델에서 Dropout 비율과 FC Layer 수도 바꿨는데 이 부분이 문제일 수도
4. 데이터 증강 과정에 줌인/줌아웃을 넣었다가 없앴는데 다시 넣어보는 게 나을지도

5. 이 모델에서 데이터 증강 과정을 수정하고(원 데이터의 150% 랜덤하게 조작) Dropout 비율만 바꿨는데 정확도가 오히려 훨씬 떨어짐 

6. 에포크 수를 늘리고 optimizer를 SGD로 바꾼 후 learning rate를 낮춰 test accuracy 향상시킴