import re
from sklearn.feature_extraction.text import TfidfVectorizer

regex = re.compile('[^a-z ]')

# 리뷰 데이터를 가져옵니다. 이전 실습과 동일하게 리스트 `documents`에는 전처리되어 있는 리뷰 데이터가 들어있습니다.
with open("text.txt", 'r') as f:
    documents = []
    for line in f:
        lowered_sent = line.rstrip().lower()
        filtered_sent = regex.sub('', lowered_sent)
        documents.append(filtered_sent)

# TfidfVectorizer() 객체를 이용해 TF-IDF Bag of words 문서 벡터를 생성하여 변수 X에 저장하세요.
tv = TfidfVectorizer()
X = tv.fit_transform(documents)

# 변수 X의 차원을 변수 dim1에 저장하세요.
dim1 = X.shape
# X 변수의 차원을 확인해봅니다.
print(dim1)

# 첫 번째 문서의 TF-IDF Bag of words를 vec1 변수에 저장하세요.
vec1 = X[0]
# 첫 번째 문서의 TF-IDF Bag of words를 확인합니다.
print(vec1)

# 위에서 생성한 TfidfVectorizer() 객체를 이용해 TF-IDF 기반 Bag of N-grams 문서 벡터를 생성하세요.
unibi_v = TfidfVectorizer(ngram_range=(1, 2))
unibigram_X = unibi_v.fit_transform(documents)


# 생성한 TF-IDF 기반 Bag of N-grams 문서 벡터의 차원을 변수 dim2에 저장하세요.
dim2 = unibigram_X.shape
# 문서 벡터의 차원을 확인합니다.
print(dim2)