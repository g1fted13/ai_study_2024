import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D, GlobalAveragePooling2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import AdamW
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import confusion_matrix

# os.system('pip install tensorflow') # 필요한 경우 주석 해제

# 1. 케라스에서 데이터 가져오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리(픽셀값 정규화)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 3. 레이블 원-핫 인코딩
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 4. 훈련/검증 데이터 분리 (모델 학습 전에 미리 나누어야 함)
# 데이터 증강을 위해 원본 x_train을 x_train_main과 x_valid로 나눕니다.
(x_train_main, x_valid) = x_train[5000:], x_train[:5000]
(y_train_main, y_valid) = y_train[5000:], y_train[:5000]

# 5. ImageDataGenerator 정의 (실시간 증강 설정)
datagen = ImageDataGenerator(
    rotation_range=60,      # 이미지를 45도 내외로 회전
    width_shift_range=0.1,  # 좌우로 10% 이동
    height_shift_range=0.1, # 상하로 10% 이동
    horizontal_flip=True,   # 좌우 반전
    zoom_range=0.05
)

# 6. CNN 모델링 (모델을 먼저 정의해야 학습을 시킬 수 있습니다)

# 학습률 조정 함수 정의
def scheduler(epoch, lr):
    if epoch < 10:
        return 0.01
    elif epoch < 40:
        return 0.001
    else:
        return 0.0001
    
model = Sequential()

# 첫 번째 Conv층
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3))) # 첫 층에는 input_shape 명시 필요
model.add(BatchNormalization())
model.add(Activation('relu'))

# 두 번째 Conv층
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))

# 세 번째 Conv층
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 네 번째 Conv층
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))

# 다섯 번째 Conv층
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 여섯 번째 Conv층
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 일곱 번째 Conv층
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))

# Output
model.add(GlobalAveragePooling2D()) # Flatten 대신 사용
model.add(Dense(10, activation='softmax'))

# 모델 요약
model.summary()

# 7. 모델 컴파일
optimizer = AdamW(learning_rate=0.001, weight_decay=0.004)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 8. 모델 학습하기 (여기로 fit 함수 이동)
lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

# datagen.flow를 사용하여 학습
history = model.fit(
    datagen.flow(x_train_main, y_train_main, batch_size=32),
    epochs=150,
    validation_data=(x_valid, y_valid),
    callbacks=[lr_scheduler, checkpointer],
    verbose=2,
    steps_per_epoch=len(x_train_main) // 32 # 전체 데이터를 배치 사이즈로 나눈 만큼 반복
)

# 9. 모델 평가 및 시각화 (기존 코드 유지)
model.load_weights('model.weights.best.hdf5')
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test Accuracy: ', score[1])

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

misclassified_idx = np.where(y_pred_classes != y_true)[0]

plt.figure(figsize=(15, 4))
for i, idx in enumerate(misclassified_idx[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow((x_test[idx] * 255).astype('uint8'))
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred_classes[idx]]
    plt.title(f"True: {true_label}, Pred: {pred_label}")
    plt.axis('off')
plt.show()







'''
2024-01-07 21:58
Test Accuracy:  0.6965000033378601

2024-01-17 19:05
Test Accuracy:  0.7813000082969666

2024-01-18 05:13
Test Accuracy:  0.739799976348877

2024-01-18 20:10
Test Accuracy: 0.6894999742507935

2024-01-29 16:26 (데이터 증강 + 2번째 모델)
Test Accuracy: 0.785099983215332

2024-01-29 16:56 (데이터 증강 + 2번째 모델 + 에포크 30)
Test Accuracy:  0.784500002861023

2024-01-29 17:21 (데이터 증강 + 3번째 모델 + 에포크 30)
Test Accuracy: 0.6814000010490417

2024-01-29 18:18 (데이터 증강 + 3번째 모델 + 에포크 80 + optimizer SGD learning_rate 0.001)
Test Accuracy: 0.8033000230789185

2024-01-29 18:53 (데이터 증강 + 3번째 모델 + 에포크 80 + optimizer SGD learning_rate 0.01 -> 0.001)
Test Accuracy: 0.8113999962806702

2025-12-29 (전역 후 처음!) (SGD에 momentum 추가한 게 제일 효과적이었음)
Test Accuracy:  0.8805000185966492

2025-12-29 (AdamW, Augmentation 증강, Epoch 150)
Test Accuracy:  0.8379999995231628
'''