from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

test_sentences = [
    "i have looked forward to seeing this since i first saw it amoungst her work",
    "this is a superb movie suitable for all but the very youngest",
    "i first saw this movie when I was a little kid and fell in love with it at once",
    "i am sooo tired but the show must go on",
]

# 영어 stopword를 저장하세요.
stopwords = stopwords.words('english')

print(stopwords)

# stopword를 추가하고 업데이트된 stopword를 저장하세요.
new_keywords = ['noone', 'sooo', 'thereafter', 'beyond', 'amoungst', 'among']
updated_stopwords = stopwords + new_keywords

print(updated_stopwords)

# 업데이트된 stopword로 test_sentences를 전처리하고 tokenized_word에 저장하세요.
tokenized_word = []

for sentence in test_sentences:
    tokens = word_tokenize(sentence)
    new_sent = []
    for token in tokens:
        if token not in updated_stopwords:
            new_sent.append(token)
    tokenized_word.append(new_sent)

print(tokenized_word)

# stemming을 해보세요.
stemmed_sent = []
stemmer = PorterStemmer()

for word in tokenized_word[0]:
    stemmed_sent.append(stemmer.stem(word))

print(stemmed_sent)