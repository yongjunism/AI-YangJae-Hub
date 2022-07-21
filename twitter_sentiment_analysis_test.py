import matplotlib.pyplot as plt
import tensorflow as tf

with open("test.csv") as csv_f:
    head = "\n".join([next(csv_f) for x in range(5)])
print(head)


model_path = 'saved_model/'
# load_model 함수를 통해 model_path에 있는 모델을 불러옵니다.
new_model = tf.keras.models.load_model(model_path)

def test(model, test_dataset):
    if model == None or test_dataset == None:
        return
    # test 데이터를 이용하여 모델을 검증합니다.
    test_loss, test_acc = model.evaluate(test_dataset)
    
    # 결과를 출력합니다.
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    

# 데이터의 기본 형태에 대한 정보입니다.
column_names = ["text", "label"]
column_defaults = ["string", "int32"]
root_path = "./"
test_file_path = root_path + "test.csv"

# 데이터 불러오기
# <ToDo>: test_dataset을 불러오세요.
test_dataset = tf.data.experimental.make_csv_dataset(test_file_path, column_names=column_names, label_name='label', column_defaults=column_defaults, batch_size=320, header=False, num_epochs=1, shuffle_seed=0) # Problem 1

test_dataset = test_dataset.map(lambda text, label: (text['text'], label))




# 테스트
# <ToDo>: 학습된 모델의 검증을 위해 test 함수의 적절한 parameter를 전달해주세요.
test(new_model, test_dataset)  # Problem 2