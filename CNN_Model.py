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
# LearningRateScheduler 대신 ReduceLROnPlateau, EarlyStopping 추가
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
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

# 4. 훈련/검증 데이터 분리
(x_train_main, x_valid) = x_train[5000:], x_train[:5000]
(y_train_main, y_valid) = y_train[5000:], y_train[:5000]

# 5. ImageDataGenerator 정의 (수정됨: 회전 각도 60 -> 25)
datagen = ImageDataGenerator(
    rotation_range=25,      # [수정] 과도한 회전은 오히려 독이 되므로 25도로 완화
    width_shift_range=0.1,  # 좌우로 10% 이동
    height_shift_range=0.1, # 상하로 10% 이동
    horizontal_flip=True,   # 좌우 반전
    zoom_range=0.05
)

# 6. CNN 모델링

model = Sequential()

# 첫 번째 Conv층
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
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

# Output (수정됨: Dense 층 추가)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu')) # [수정] 모델의 표현력을 높이기 위해 은닉층 추가
model.add(Dense(10, activation='softmax'))

# 모델 요약
model.summary()

# 7. 모델 컴파일
optimizer = AdamW(learning_rate=0.001, weight_decay=0.004)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 8. 모델 학습하기 (수정됨: Callbacks 변경)

# [수정] ReduceLROnPlateau: val_loss가 10번 동안 안 줄어들면 학습률 절반으로 깎음
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=10, 
    min_lr=1e-6, 
    verbose=1
)

# [수정] EarlyStopping: 15번 동안 안 줄어들면 학습 조기 종료 + 최고 성능 가중치 복구
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    verbose=1, 
    restore_best_weights=True 
)

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

# datagen.flow를 사용하여 학습
history = model.fit(
    datagen.flow(x_train_main, y_train_main, batch_size=32),
    epochs=180, # 넉넉하게 설정 (EarlyStopping이 있으므로 괜찮음)
    validation_data=(x_valid, y_valid),
    callbacks=[reduce_lr, checkpointer, early_stopping], # 콜백 리스트 수정됨
    verbose=2,
    steps_per_epoch=len(x_train_main) // 32
)

# 9. 모델 평가 및 시각화
# EarlyStopping의 restore_best_weights=True 덕분에 학습 종료 후 model은 이미 최적의 가중치를 가지고 있음
# 하지만 안전을 위해 저장된 파일에서 다시 로드
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

2025-12-29 (AdamW, 회전각 다시 감소, Dense층 추가, Epoch 86에서 early stopping)
Test Accuracy:  0.8931000232696533
'''