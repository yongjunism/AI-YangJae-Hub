import requests

# 아래 문장의 감정을 예측합니다.
test_data = ['i am happy', 'i want to go', 'i wake too early so i feel grumpy', 'i feel alarmed']

myobj = {"infer_texts": test_data}

x = requests.post("http://0.0.0.0:8080/predict", json = myobj)
print("감정 분석 결과: " + x.text)ß