import nltk

sent_1 = "오늘 중부지방을 중심으로 소나기가 예상됩니다"
sent_2 = "오늘 전국이 맑은 날씨가 예상됩니다"

def cal_jaccard_sim(sent1, sent2):
    # 각 문장을 토큰화 후 set 타입으로 변환하세요.
    words_sent1 = set(sent1.split())
    words_sent2 = set(sent2.split())

    # 공통된 단어의 개수를 intersection 변수에 저장하세요.
    intersection = words_sent1.intersection(words_sent2)
    
    # 두 문장 내 발생하는 모든 단어의 개수를 union 변수에 저장하세요.
    union = words_sent1.union(words_sent2)

    # intersection과 union을 사용하여 자카드 지수를 계산하고 float 타입으로 반환하세요.
    return float(len(intersection) / len(union))

# cal_jaccard_sim() 함수 실행 결과를 확인합니다.
print(cal_jaccard_sim(sent_1, sent_2))

# nltk의 jaccard_distance() 함수를 이용해 자카드 유사도를 계산하세요.
sent1_set = set(sent_1.split())
sent2_set = set(sent_2.split())
nltk_jaccard_sim = 1- nltk.jaccard_distance(sent1_set, sent2_set)

# 직접 정의한 함수와 결과가 같은지 비교합니다.
print(nltk_jaccard_sim)