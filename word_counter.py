import re

word_counter = dict()

# 단어가 key, 빈도수가 value로 구성된 딕셔너리 변수를 만드세요.
with open('text.txt', 'r') as f:
    for line in f:
        for word in line.rstrip().split():
            if word not in word_counter:
                word_counter[word] = 1
            else:
                word_counter[word] += 1
        pass

print(word_counter)


# 텍스트 파일에 내 모든 단어의 총 빈도수를 구해보세요.
total = 0

# 텍스트 파일 내 100회 이상 발생하는 단어를 리스트 형태로 저장하세요.
up_five = list()

for word, freq in word_counter.items():
    total += freq
    if freq >= 100:
        up_five.append(word)

print(total)
print(up_five)