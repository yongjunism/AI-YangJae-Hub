from numpy import sqrt, dot
from scipy.spatial import distance
from sklearn.metrics import pairwise

sent_1 = [0.3, 0.2, 0.2133, 0.3891, 0.8852, 0.586, 1.244, 0.777, 0.882]
sent_2 = [0.03, 0.223, 0.1, 0.4, 2.931, 0.122, 0.5934, 0.8472, 0.54]
sent_3 = [0.13, 0.83, 0.827, 0.92, 0.1, 0.32, 0.28, 0.34, 0]

def cal_cosine_sim(v1, v2):
    # 벡터 v1, v2 간 코사인 유사도를 계산 후 반환하세요.
    top = dot(v1, v2)
    size1 = sqrt(dot(v1, v1))
    size2 = sqrt(dot(v2, v2))
    return top / (size1 * size2)

# 정의한 코사인 유도 계산 함수를 확인합니다.
print(cal_cosine_sim(sent_1, sent_2))

# scipy의 distance.cosine() 함수를 이용한 코사인 유사도를 계산하세요.
scipy_cosine_sim = 1 - distance.cosine(sent_1, sent_2)

# scipy를 이용해 계산한 코사인 유사도를 확인합니다.
print(scipy_cosine_sim)

# scikit-learn의 pairwise.cosine_similarity() 함수를 이용한 코사인 유사도를 계산하세요.
all_sent = [sent_1] + [sent_2] + [sent_3]
scikit_learn_cosine_sim  = pairwise.cosine_similarity(all_sent)

# scikit-learn을 이용해 계산한 코사인 유사도를 확인합니다.
print(scikit_learn_cosine_sim)