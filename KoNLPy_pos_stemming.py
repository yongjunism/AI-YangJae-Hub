# 경고문을 무시합니다.
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
from konlpy.tag import Kkma, Okt

# sts-train.tsv 파일에 저장되어 있는 KorSTS 데이터셋을 불러옵니다.
sent = pd.read_table("sts-train.tsv", delimiter='\t', header=0)['sentence1']

# sent 변수에 저장된 첫 5개 문장을 확인해봅니다.
print(sent[:5])

# 꼬꼬마 형태소 사전을 이용해서 sent 내 문장의 명사를 nouns 리스트에 저장하세요.
nouns = []
kkma = Kkma()
for s in sent:
    nouns += kkma.nouns(s)

# 명사의 종류를 확인해봅니다.
print(set(nouns))

# Open Korean Text 형태소 사전을 이용해서 sent 내 형태소 분석 결과를 pos_results 리스트에 저장하세요.
pos_results = []
okt = Okt()
for s in sent:
    pos_results.append(okt.pos(s))

# 분석 결과를 확인해봅니다.
print(pos_results)

# stemming 기반 형태소 분석이 적용된 sent의 두 번째 문장을 stem_pos_results 리스트에 저장하세요.
stem_pos_results = okt.pos(sent[1], stem=True)
print(stem_pos_results)