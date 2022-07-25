from elice_utils import EliceUtils
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import codecs

elice_utils = EliceUtils()


def create_similarity_matrix(corpus):
    """
    문서를 담은 corpus list에서 각 문서 간 유사도 측정
    :param corpus: 문서를 text로 가지는 list
    :return: 문서 간 유사도 값을 가지는 행렬
    """
    corpus_size = len(corpus)
    # <ToDo>: 문서들마다 유사도 값을 가지는 행렬 sim_mat을 만드세요.
    # sim_mat[i, j] == i번째 문서와 j번째 문서의 유사도
    sim_mat = np.zeros((corpus_size, corpus_size))
    
    corpus_vsm = make_vector_space_model_tf_idf(corpus)
    
    for idx in range(corpus_size):
        sim_mat[idx, idx] = 1.
        
        for jdx in range(idx+1, corpus_size):
            doc_1 = corpus_vsm[idx, :].reshape(1, -1)
            doc_2 = corpus_vsm[jdx, :].reshape(1, -1)
            pair_sim = compute_doc_pair_similarity(doc_1, doc_2)
            sim_mat[idx, jdx] = pair_sim
            sim_mat[jdx, idx] = pair_sim
    return sim_mat


def make_vector_space_model_tf_idf(corpus):
    """
    문서를 담은 corpus list에서 tf-idf 가중치를 가지는 vector space model 생성
    :param corpus: 문서를 text로 가지는 list
    :return: tf-idf 가중치를 가지는 vector space model (numpy array)
    """
    # <ToDo>: tf-idf 가중치를 가지는 vector space model corpus_vsm을 만드세요.
    vectorizer = TfidfVectorizer()
    corpus_vsm = vectorizer.fit_transform(corpus)

    return corpus_vsm


def compute_doc_pair_similarity(doc_1, doc_2):
    """
    두 문서의 유사도를 cosine similarity로 계산
    :param doc_1: 첫 번째 문서 tf-idf 가중치를 가지는 벡터
    :param doc_2: 두 번째 문서 tf-idf 가중치를 가지는 벡터
    :return: 두 문서의 유사도 값
    """
    # <ToDo>: 두 문서 doc_1, doc_2의 cosine similarity값을 계산하여 이를 cos_sim에 넣어주세요.
    cos_sim = cosine_similarity(doc_1, doc_2).flatten()

    return cos_sim


def main():
    # 데이터 'news.txt' 파일을 불러옵니다.
    corpus = list()
    with codecs.open("./data/news.txt", "r", "utf-8") as txt_f:
        for line in txt_f:
            corpus.append(line.strip())
    
    # corpus 내 문서들의 유사도값을 계산합니다.
    sim_mat = create_similarity_matrix(corpus)
    
    # 결과 출력
    print("First news title: {}".format(corpus[0]))
    print("Second news title: {}".format(corpus[1]))
    print("Similarity between them: {}".format(sim_mat[0, 1]))
    print()

    print("Second news title: {}".format(corpus[1]))
    print("Third news title: {}".format(corpus[2]))
    print("Similarity between them: {}".format(sim_mat[1, 2]))
    print()

    print("Third news title: {}".format(corpus[2]))
    print("Forth news title: {}".format(corpus[3]))
    print("Similarity between them: {}".format(sim_mat[3, 2]))


if __name__ == "__main__":
    main()