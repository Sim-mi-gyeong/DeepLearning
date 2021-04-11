import tensorflow as tf
from tensorflow import keras

# 하이퍼파라미터 튜닝할 때 사용할 kerastuner 설치 및 import 
!pip install -q -U keras-tuner
import kerastuner as kt

import IPython

# 데이터셋으로 keras에서 제공하는 fashion_mnist data load
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

img_train=img_train.astype('float32')/255.0
img_test=img_test.astype('float32')/255.0


# 파라미터 튜닝할 간단한 모델 빌드
def model_builder(hp):
  model=keras.Sequential()

  # 입력 이미지 1차원으로
  model.add(keras.layers.Flatten(input_shape=(28,28)))

  # Tune the number of units in the first Denselayer
  # Choose an optimal value 32-512

  hp_units=hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, 0.0001

  hp_learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),# from_logits=True: 정규화
                metrics=['accuracy']) 
  
  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy', # 최적화할 하이퍼모델
                     max_epochs=10,
                     factor=3, # 로그의 밑부분
                     directory='my_dir',
                     project_name='intro_to_kt')

# Hyperband 튜닝 알고리즘: 적응형 리소스 할당 및 조기 중단 사용-> 고성능 모델에서 신속하게 수렴
# -> epoch 몇 개에 대해 많은 수의 모델 훈련 후 최고 성능을 보이는 절반만 다음 단계로
# 1+log(factor)(max_epochs) -> 가장 가까운 정수로 반올림해서 한 브래킷에서 훈련할 모델 수 결정

# 하이퍼파라미터 검색 실행 전 - 훈련 단계가 끝날 때마다 '훈련 결과 지우도록' 콜백
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait=True)

# 하이퍼파라미터 검색 실행
# 검색 메서드의 인수는 위의 콜백 외에 tf.keras.model.fit에 사용되는 인수와 같음
tuner.search(img_train, label_train, epochs = 10, validation_data = (img_test, label_test), callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# 최적 파라미터 찾는 데에 시간이 너무 오래 소요됨.

# 최적의 하이퍼파라미터로 모델 재훈련

model=tuner.hypermodel.build(best_hps)
model.fit(img_train, label_train, epochs=10, validation_data=(img_test, label_test))

# Epoch 9/10 -> 10/10 val accuracy 감소

# my_dir/intro_to_kt 디렉토리: 하이퍼파라미터 검색 중에 실행되는 모든 시험(모델 구성)에 대한 상세 로그와 체크포인트
# 하이퍼파라미터 검색 재실행시 Keras Tuner가 로그의 기존 상태를 사용하여 검색을 재개
# 비활성화하려면 튜너를 인스턴스화하는 동안 추가 overwrite = True 인수를 전달
