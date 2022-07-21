import matplotlib.pyplot as plt
import tensorflow as tf
from elice_utils import EliceUtils
elice_utils = EliceUtils()

with open("train.csv") as csv_f:
    head = "\n".join([next(csv_f) for x in range(5)])
print(head)


# Train 함수


def train(train_dataset, valid_dataset, epochs=5):
    # vocab.csv에 적혀진 단어를 기반으로 단어를 벡터로 바꾸는 encoder를 만듭니다.
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(output_sequence_length=200,
                                                                           vocabulary="./vocab.csv")

    # RNN classifier 모델을 만듭니다.
    # 단어 => encoder => Embedding => 양방향 RNN => Dense => Dense의 구조입니다.
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=300, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300)),
        tf.keras.layers.Dense(300, activation='relu'),
        # <ToDo>: model의 마지막에 classification을 위해 dense layer를 추가해주세요.
        tf.keras.layers.Dense(3)
    ])

    # 모델의 loss 함수와 optimizier를 정합니다.
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    # train 데이터로 학습시키며 valid 데이터로 성능을 확인합니다.
    history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset, validation_steps=1,
                        use_multiprocessing=True, workers=32)

    return model, history


# Test 함수


def test(model, test_dataset):
    # test 데이터를 이용하여 모델을 검증합니다.
    test_loss, test_acc = model.evaluate(test_dataset)
    
    # 결과를 출력합니다.
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))


# 그래프 그리는 함수


def draw_graph(history, metric='loss'):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.savefig('train_valid_loss.png', bbox_inches='tight')
    elice_utils.send_image('train_valid_loss.png')


# 데이터 불러오기 


# 데이터의 기본 형태에 대한 정보입니다.
column_names = ["text", "label"]
column_defaults = ["string", "int32"]
root_path = "./"
train_file_path = root_path + "train.csv"
valid_file_path = root_path + "valid.csv"

# train 데이터 csv 파일을 읽어옵니다.
train_dataset = tf.data.experimental.make_csv_dataset(train_file_path, column_names=column_names, batch_size=320,
                                                      label_name="label", column_defaults=column_defaults,
                                                      header=False, num_epochs=1, shuffle_seed=0)

# <ToDo>: valid_dataset을 불러오세요.
valid_dataset = tf.data.experimental.make_csv_dataset(valid_file_path, column_names=column_names, batch_size=320,
                                                      label_name="label", column_defaults=column_defaults,
                                                      header=False, num_epochs=1, shuffle_seed=0) # Problem 2

# <ToDo>: valid_dataset과 test_dataset을 불러오세요.
train_dataset = train_dataset.map(lambda text, label: (text["text"], label))
valid_dataset = valid_dataset.map(lambda text, label: (text['text'], label)) # Problem 2


# 모델학습


model, history = train(train_dataset, valid_dataset, epochs=1)


# 그래프 출력


draw_graph(history)