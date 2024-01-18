'''
import os
os.system('pip install tensorflow')
'''

#케라스에서 데이터 가져오기
from keras.datasets import cifar10  
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#데이터 전처리(픽셀값 정규화)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


#레이블 원-핫 인코딩
import keras
import numpy as np

num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#데이터 보강
from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(zoom_range= (0.7, 1.3), 
width_shift_range = 0.2, 
height_shift_range=0.2, 
horizontal_flip=True)

augment_ratio = 1.5 #전체 데이터의 150%
augment_size = int(augment_ratio * x_train.shape[0])

randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented, y_augmented = gen.flow(x_augmented, y_augmented,
                                    batch_size=augment_size,
                                    shuffle=False).next()

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))


#보강된 학습데이터/정답데이터를 랜덤하게 섞음
s = np.arange(x_train.shape[0])
np.random.shuffle(s)

x_train = x_train[s]
y_train = y_train[s]


#훈련 데이터, 검증 데이터 분류
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]


#데이터 모양 확인
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

print("x_valid shape:", x_valid.shape)
print("y_valid shape:", y_valid.shape)



#CNN 모델링

#케라스에서 모델과 층 import + 모델 생성
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

model = Sequential()

#첫 번째 Conv층 쌓기
#Input : 32 * 32 * 3 image
#Filters: (3 * 3 * 3) * 16 -> 32 filters
#Output: 32 * 32 * 32

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))


#두 번째 Conv층 쌓기
#Input: 32 * 32 * 32
#Filters: (3 * 3 * 32) * 32 -> 32 filters
#Output: 32 * 32 * 32
#MaxPool: 16 * 16 * 32

model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))


#세 번째 Conv층 쌓기
#Input: 16 * 16 * 32
#Filters: (3 * 3 * 32) * 64 -> 64 filters
#Output: 16 * 16 * 64

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

#네 번째 Conv층 쌓기
#Input: 16 * 16 * 64
#Filters: (3 * 3 * 64) * 64 -> 64 filters
#Output: 16 * 16 * 64
#MaxPool: 8 * 8 * 64

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))

#다섯 번째 Conv층 쌓기
#Input: 8 * 8 * 64
#Filters: (3 * 3 * 64) * 128 -> 128 filters
#Output: 8 * 8 * 128

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

#여섯 번째 Conv층 쌓기
#Input: 8 * 8 * 128
#Filters: (3 * 3 * 128) * 128 -> 128 filters
#Output: 8 * 8 * 128

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

#일곱 번째 Conv층 쌓기
#Input: 8 * 8 * 128
#Filters: (3 * 3 * 128) * 128 -> 128 filters
#Output: 8 * 8 * 128
#MaxPool: 4 * 4 * 128

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))

#Dropout 층 쌓기
#훈련 시 Layer의 30%의 뉴런을 무작위로 비활성화
model.add(Dropout(0.3))


#Flatten
#Input: 4 * 4 * 128
#Output: 2028
model.add(Flatten())


#Two Fully-Connected Layer
#Input: 2028
#Intermediate Output: 500
#Dropout
#30%의 뉴런을 무작위로 비활성화
#Final Output: 10
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

#모델 요약
model.summary()


#모델 컴파일하기
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#모델 학습하기
#callback의 진행상황에 대한 상세 출력 활성화; 각 에포크마다 결과가 출력됨
#전체 훈련 데이터에 대해서 10번 에포크 반복

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

hist=  model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_valid, y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)


#val_acc가 가장 좋았던 가중치 사용하기
model.load_weights('model.weights.best.hdf5')

#모델 평가하기
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test Accuracy: ', score[1])


# 모델 예측
y_pred = model.predict(x_test)

# 예측 레이블을 원-핫 인코딩에서 정수 레이블로 변환
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 레이블에 해당하는 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 혼동 행렬 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred_classes)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 잘못 분류된 이미지 식별
misclassified_idx = np.where(y_pred_classes != y_true)[0]




# 잘못 분류된 이미지 중 일부 시각화
plt.figure(figsize=(15, 4))
for i, idx in enumerate(misclassified_idx[:10]): # 처음 10개만 표시
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
'''
