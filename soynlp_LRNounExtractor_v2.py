from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2

sent = '트와이스 아이오아이 좋아여 tt가 저번에 1위 했었죠?'

# 학습에 사용할 데이터가 train_data에 저장되어 있습니다.
corpus_path = 'articles.txt'
train_data = DoublespaceLineCorpus(corpus_path)
print("학습 문서의 개수: %d" %(len(train_data)))

# LRNounExtractor_v2 객체를 이용해 train_data에서 명사로 추정되는 단어를 nouns 변수에 저장하세요.
noun_extractor = LRNounExtractor_v2()
nouns = noun_extractor.train_extract(train_data)

# 생성된 명사의 개수를 확인해봅니다.
print(len(nouns))

# 생성된 명사 목록을 사용해서 sent에 주어진 문장에서 명사를 sent_nouns 리스트에 저장하세요.
sent_nouns = []
for word in sent.split():
    if word in nouns:
        sent_nouns.append(word)

print(sent_nouns)