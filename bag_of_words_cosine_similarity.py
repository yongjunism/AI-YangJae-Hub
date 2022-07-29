# 경고문을 제거합니다.
import warnings
warnings.filterwarnings(action='ignore')

import pickle
from sklearn.metrics.pairwise import cosine_similarity

sent1 = ["I first saw this movie when I was a little kid and fell in love with it at once."]
sent2 = ["Despite having 6 different directors, this fantasy hangs together remarkably well."]

with open('bow_models.pkl', 'rb') as f:
    # 저장된 모델을 불러와 객체와 벡터를 각각vectorizer와 X에 저장하세요.
    vectorizer, X = pickle.load(f)

# sent1, sent2 문장을 vectorizer 객체의 transform() 함수를 이용해 변수 vec1, vec2에 저장합니다.
vec1 = vectorizer.transform(sent1)
vec2 = vectorizer.transform(sent2)

#  vec1과 vec2의 코사인 유사도를 변수 sim1에 저장합니다.
sim1 = cosine_similarity(vec1, vec2)
# 두 벡터의 코사인 유사도를 확인해봅니다.
print(sim1)

# vec1과 행렬 X의 첫 번째 문서 벡터 간 코사인 유사도를 변수 sim2에 저장합니다.
sim2 = cosine_similarity(vec1, X[0])
# X의 첫 번째 문서와 vec1의 코사인 유사도를 확인해봅니다.
print(sim2)