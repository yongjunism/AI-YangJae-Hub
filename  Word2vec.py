from elice_utils import EliceUtils
from gensim.models.word2vec import Word2Vec
elice_utils = EliceUtils()


def compute_similarity(model, word1, word2):
    """
    word1과 word2의 similarity를 구하는 함수
    :param model: word2vec model
    :param word1: 첫 번째 단어
    :param word2: 두 번째 단어
    :return: model에 따른 word1과 word2의 cosine similarity
    """
    # <ToDo>: model에 따른 word1과 word2의 cosine similarity를 계산하세요.
    similarity = model.wv.similarity(word1, word2)

    return similarity


def get_word_by_calculation(model, word1, word2, word3):
    """
    단어 벡터들의 연산 결과 추론하는 함수
    연산: word1 - word2 + word3
    :param model: word2vec model
    :param word1: 첫 번째 단어로 연산의 시작
    :param word2: 두 번째 단어로 빼고픈 단어
    :param word3: 세 번째 단어로 더하고픈 단어
    :return: 벡터 계산 결과에 가장 알맞는 단어
    """
    # <ToDo>: model을 이용하여 word1 - word2 + word3 결과에 가장 근접한 단어를 찾으세요.
    output_word = model.wv.most_similar(positive=[word1, word3], negative=[word2])[0][0]

    return output_word


def main():
    # 학습된 word2vec model을 불러옵니다.
    model = Word2Vec.load('./data/w2v_model')
    
    # 두 단어의 유사도를 찾습니다.
    word1 = "이순신"
    word2 = "원균"
    word1_word2_sim = compute_similarity(model, word1, word2)
    print("{}와/과 {} 유사도: {}".format(word1, word2, word1_word2_sim))
    
    # '대한민국'에서 '서울'을 뺀 후 '런던'을 더하면 어떤 단어가 나올까요?
    word1 = "대한민국"
    word2 = "서울"
    word3 = "런던"
    cal_result = get_word_by_calculation(model, word1, word2, word3)
    print("{} - {} + {}: {}".format(word1, word2, word3, cal_result))

    return word1_word2_sim, cal_result


if __name__ == "__main__":
    main()