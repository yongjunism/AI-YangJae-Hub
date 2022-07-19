from elice_utils import EliceUtils
import codecs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

elice_utils = EliceUtils()
# 엘리스 환경에서 nltk data를 사용하기 위해서 필요합니다.
nltk.data.path.append("./")

def count_words(input_text):
    """
    input_text 내 단어들의 개수를 세는 함수
    :param input_text: 텍스트
    :return: dictionary, key: 단어, value: input_text 내 단어 개수
    """
    # <ToDo>: key: 단어, value: input_text 내 단어 개수인 output_dict을 만듭니다.
    output_dict = dict()
    tokens = word_tokenize(input_text)

    print('tokens', tokens[:5])

    for one_token in tokens:
        try:
            output_dict[one_token] += 1
        except KeyError:
            output_dict[one_token] = 1
    return output_dict

def remove_stopwords(input_dict):
    """
    input_dict 내 단어 중 stopwords 제거
    :param input_dict: count_words 함수 반환값인 dictionary
    :return: input_dict에서 stopwords가 제거된 것
    """
    # <ToDo>: count_words에서 만든 input_dict에서 stopwords를 제거하세요.
    output_dict = dict()
    
    stop_words = set(stopwords.words('english'))

    for one_word, one_value in input_dict.items():
        if one_word not in stop_words:
            output_dict[one_word] = one_value
    return output_dict


def remove_less_freq(input_dict, lower_bound=10):
    """
    input_dict 내 단어 중 lower_bound 이상 나타난 단어만 추출
    :param input_dict: count_words 함수 반환값인 dictionary
    :param lower_bound: 단어를 제거하는 기준값
    :return: input_dict에서 lower_bound보다 많이 나타난 단어들
    """
    # <ToDo>: count_words에서 만든 input_dict에서 lower_bound보다 많이 나타난 단어들을 제거하세요.
    output_dict = dict()

    for one_word, one_value in input_dict.items():
        if one_value >= lower_bound:
            output_dict[one_word] = one_value

    return output_dict


def main():
    # 데이터 파일인 'text8_1m_part_aa.txt' 파일을 불러옵니다.
    with codecs.open("data/text8_1m_part_aa.txt", "r", "utf-8") as html_f:
        text8_text = "".join(html_f.readlines())
    
    # 데이터 내 단어 및 등장 횟수를 세어봅시다.
    word_dict1 = count_words(text8_text)
    print('word_dict1', word_dict1)
    # 해당 단어들 중 stopwords를 제거합시다.
    word_dict2 = remove_stopwords(word_dict1)
    print('word_dict2', word_dict2)
    # 그리고 단어들 중 나타난 횟수가 10 이하인 것은 제거합시다.
    word_dict3 = remove_less_freq(word_dict2, 10)
    
    # 각 사전들의 단어 개수를 출력합시다.
    print("# word_dict1: {}".format(len(word_dict1)))
    print("# word_dict2: {}".format(len(word_dict2)))
    print("# word_dict3: {}".format(len(word_dict3)))
    
    # 각 사전들을 단어 등장 횟수를 기준으로 정렬하여 상위 15개를 출력합시다.
    top_words1 = sorted(word_dict1.items(), key=lambda x: x[1], reverse=True)[:15]
    print("word_dict3 topwords: {}".format(top_words1))
    top_words2 = sorted(word_dict2.items(), key=lambda x: x[1], reverse=True)[:15]
    print("word_dict3 topwords: {}".format(top_words2))
    top_words3 = sorted(word_dict3.items(), key=lambda x: x[1], reverse=True)[:15]
    print("word_dict3 topwords: {}".format(top_words3))

    return word_dict3


if __name__ == "__main__":
    main()