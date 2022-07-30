# -*- coding: utf-8 -*-
import random
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import sqrt, dot

random.seed(10)

doc1 = ["homelessness has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school work or vote for the matter"]

doc2 = ["it may have ends that do not tie together particularly well but it is still a compelling enough story to stick with"]

# 데이터를 불러오는 함수입니다.
def load_data(filepath):
    regex = re.compile('[^a-z ]')

    gensim_input = []
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            lowered_sent = line.rstrip().lower()
            filtered_sent = regex.sub('', lowered_sent)
            tagged_doc = TaggedDocument(filtered_sent, [idx])
            gensim_input.append(tagged_doc)
            
    return gensim_input
    
def cal_cosine_sim(sent1, sent2):
    # 벡터 간 코사인 유사도를 계산해 주는 함수를 완성합니다.
    top = dot(sent1, sent2)
    size1 = sqrt(dot(sent1, sent1))
    size2 = sqrt(dot(sent2, sent2))
    return top / (size1 * size2)
    
# doc2vec 모델을 documents 리스트를 이용해 학습하세요.
documents = load_data("text.txt")
d2v_model = Doc2Vec(window=2, vector_size=50)
d2v_model.build_vocab(documents)
d2v_model.train(documents, total_examples=d2v_model.corpus_count, epochs=5)

# 학습된 모델을 이용해 doc1과 doc2에 들어있는 문서의 임베딩 벡터를 생성하여 각각 변수 vector1과 vector2에 저장하세요.
vector1 = d2v_model.infer_vector(doc1)
vector2 = d2v_model.infer_vector(doc2)

# vector1과 vector2의 코사인 유사도를 변수 sim에 저장하세요.
sim = cal_cosine_sim(vector1, vector2)
# 계산한 코사인 유사도를 확인합니다.
print(sim)