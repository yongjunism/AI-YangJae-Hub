import pandas as pd
import numpy as np

def cal_partial_freq(texts, emotion):
    filtered_texts = texts[texts['emotion'] == emotion]
    filtered_texts = filtered_texts['sentence']
    partial_freq = dict()

    # 실습 2에서 구현한 부분을 완성하세요.
    for sent in filtered_texts:
        words = sent.rstrip().split()
        for word in words:
            if word not in partial_freq:
                partial_freq[word] = 1
            else:
                partial_freq[word] += 1
    return partial_freq

def cal_total_freq(partial_freq):
    total = 0
    # 실습 2에서 구현한 부분을 완성하세요.
    for word, freq in partial_freq.items():
        total += freq
    return total

def cal_prior_prob(data, emotion):
    filtered_texts = data[data['emotion'] == emotion]
    # data 내 특정 감정의 로그발생 확률을 반환하는 부분을 구현하세요.
    
    return np.log(len(filtered_texts) / len(data))

def predict_emotion(sent, data):
    emotions = ['anger', 'love', 'sadness', 'fear', 'joy', 'surprise']
    predictions = []
    train_txt = pd.read_csv(data, delimiter=';', header=None, names=['sentence', 'emotion'])

    # sent의 각 감정별 로그 확률을 predictions 리스트에 저장하세요.
    for emotion in emotions:
        prob = 0
        for word in sent.split():
            emotion_counter = cal_partial_freq(train_txt, emotion)
            prob += np.log((emotion_counter[word] + 10) / (cal_total_freq(emotion_counter) + 10)) 
        prob += cal_prior_prob(train_txt, emotion)
        predictions.append((emotion, prob))
    predictions.sort(key = lambda a: a[1])
    return predictions[-1]

# 아래 문장의 예측된 감정을 확인하세요.
test_sent = "i really want to go and enjoy this party"
predicted = predict_emotion(test_sent, 'emotions_train.txt')
print(predicted)