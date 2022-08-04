from sklearn.model_selection import train_test_split

# 파일을 읽어오세요.
data = []
with open('emotions_train.txt', 'r') as f:
    for line in f:
        sent, emotion = line.rstrip().split(';')
        data.append((sent, emotion))

# 읽어온 파일을 학습 데이터와 평가 데이터로 분할하세요.
train, test = train_test_split(data, test_size=0.2, random_state=7)


# 학습 데이터셋의 문장과 감정을 분리하세요.
Xtrain = []
Ytrain = []

for train_data in train:
    Xtrain.append(train_data[0])
    Ytrain.append(train_data[1])

print(len(Xtrain))
print(len(set(Ytrain)))

# 평가 데이터셋의 문장과 감정을 분리하세요.
Xtest = []
Ytest = []

for test_data in test:
    Xtest.append(test_data[0])
    Ytest.append(test_data[1])

print(len(Xtest))
print(len(set(Ytest)))