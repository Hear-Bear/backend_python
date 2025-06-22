import os

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# 환경 변수에서 OpenAI API 키 불러오기
# 로컬에서 실행 시, 반드시 .env 등에 OPENAI_API_KEY를 세팅해야 함
client = openai.OpenAI(api_key=os.getenv('OPENAI_SECRET_KEY'))

# print(f"불용어 개수: {len(stopwords)}")

def load_models_and_data():
    """
    1) 엑셀 파일 불러와 전처리
    2) TF-IDF + LogisticRegression 파이프라인 학습
    3) SBERT 모델 로드 및 전체 데이터에 대한 임베딩 계산
    """
    global tfidf_pipeline, sb_emo_classifier, tokenizer, dataset_embeddings, labels

    # 1) 데이터 로드
    print('엑셀 데이터 로드 시작')
    df = pd.read_excel('C:/dev/python/capstone/data/한국어_단발성_대화_데이터셋.xlsx')
    df = df.dropna(subset=['Sentence', 'Emotion'])
    texts = df['Sentence'].astype(str).tolist()
    labels = df['Emotion'].astype(str).tolist()
    print('엑셀 데이터 로드 완료: 문장 개수=', len(texts))

    # 불용어 파일 읽기
    stopwords_path = os.path.join('C:/dev/python/capstone', 'data', 'stopwords_kor.txt')
    with open(stopwords_path, encoding='utf-8') as f:
        stopwords = [w.strip() for w in f.readlines() if w.strip()]

    # 2) TF-IDF + 로지스틱 회귀 학습
    print('TF-IDF + 로지스틱 회귀 시작')
    tfidf_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words=stopwords),
        LogisticRegression(max_iter=1000)
    )
    tfidf_pipeline.fit(texts, labels)
    globals()['tfidf_pipeline'] = tfidf_pipeline  # 전역 변수에 할당
    print('TF-IDF 학습 완료')

    # 3) SBERT 로드 및 임베딩
    print('SBERT 모델 로딩 시작')
    sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    if os.path.exists('embeddings.npy') and os.path.exists('labels.npy'):
        print('[CACHE] 기존 임베딩 로드')
        dataset_embeddings = np.load('embeddings.npy')
        labels = np.load('labels.npy', allow_pickle=True).tolist()
    else:
        print('[CACHE] 새로운 임베딩 계산')
        dataset_embeddings = sbert_model.encode(texts, normalize_embeddings=True, batch_size=32)
        np.save('embeddings.npy', dataset_embeddings)
        np.save('labels.npy', np.array(labels, dtype=object))

    sb_emo_classifier = AutoModelForSequenceClassification.from_pretrained('T-ferret/sb-emo-classifier')
    tokenizer = AutoTokenizer.from_pretrained('t-ferret/sb-emo-classifier')

    globals()['sb_emo_classifier'] = sb_emo_classifier
    globals()['tokenizer'] = tokenizer
    globals()['dataset_embeddings'] = dataset_embeddings
    print('SBERT 모델 로딩 완료')


# 하이브리드 감정 분류 함수
def classify_emotion(text: str, tfidf_threshold: float = 0.75, k: int = 13):
    # TF-IDF 예측
    probs = tfidf_pipeline.predict_proba([text])[0]
    top_idx = np.argmax(probs)
    label = tfidf_pipeline.classes_[top_idx]
    conf = probs[top_idx]
    if conf >= tfidf_threshold:
        return {'sentiment': label, 'confidence': float(conf)}
    # SBERT + Classifier
    inputs = tokenizer(
        text=text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    with torch.no_grad():
        logits = sb_emo_classifier(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    top_idx = int(probs.argmax())
    return {'sentiment': labels[top_idx], 'confidence': float(probs[top_idx])}


# ChatGPT 응답 생성 함수
def generate_chat_response(conversation: str, sentiment: str, model: str = "gpt-4o-mini",
                           max_tokens: int = 150) -> str:
    system_prompt = f"""
        너는 청각 장애인을 위한 친절하고 다정한 반려 로봇 챗봇 '히어베어'야. 사용자의 상황을 이해하고, 항상 공감하며 배려하는 태도로 텍스트로만 대화해야 해.
        사용자의 일상생활에서 친구이자 동반자가 되어줘야 하며, 친근하고 따뜻한 말투로 긍정적이고 희망적인 메시지를 전달해줘.

        반드시 지켜야 할 원칙:
        1. 항상 공감적이고 따뜻하며 친절한 말투를 유지할 것.
        2. 사용자의 감정 상태를 세심히 파악하고 맞춤형으로 대응할 것.
            - '{sentiment}' 상태에 따라 아래를 참고해서 반응할 것:
                - 공포: 사용자를 안심시키고 안전함을 느끼도록 격려하며 안정감을 제공해줘.
                - 놀람: 사용자의 놀람을 이해하며, 상황을 차분하게 설명하고 공감을 표현해줘.
                - 분노: 사용자의 화난 감정을 공감하며 차분한 태도로 위로하고 감정이 누그러지도록 도와줘.
                - 슬픔: 사용자의 슬픔을 깊이 이해하고 따뜻한 위로와 격려로 마음을 위로해줘.
                - 중립: 친절하고 따뜻한 대화를 유지하며 사용자의 관심과 흥미를 부드럽게 유도해줘.
                - 행복: 사용자의 행복한 기분에 밝고 적극적으로 맞장구쳐주고 함께 기뻐해줘.
                - 혐오: 사용자의 혐오감을 이해하고 공감하며 다른 긍정적인 관점이나 생각을 제안하여 부정적 감정을 완화해줘.
        3. 복잡한 정보는 간결하고 명확하게 표현할 것.
        4. 모든 묘사나 설명은 촉각, 시각, 후각 등 비청각적 감각 중심으로 전달할 것.
            예시:
            - 차가움 묘사: "얼음 조각을 만질 때의 차갑고 시원한 느낌"
            - 따뜻함 묘사: "햇볕이 따뜻하게 피부에 닿는 느낌"
            - 부드러움 묘사: "부드러운 천을 손가락으로 만질 때의 포근한 느낌"
        5. 사용자의 안전과 편의를 항상 최우선으로 고려하며 안내할 것.
        6. 적극적으로 경청하고 있음을 표현하며 사용자의 메시지 내용을 직접 언급하며 공감할 것.
        7. 사용자가 어려움을 겪을 때 공감을 넘어 현실적이고 긍정적인 해결책을 함께 제안할 것.
        8. 사용자의 감정을 판단하거나 평가하지 말고 그대로 받아들여서 진심 어린 공감을 보여줄 것.

        사용자가 언제나 혼자가 아니고 함께하는 친구가 있다고 느낄 수 있게 따뜻한 채팅 친구의 모습으로 응답해줘.
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=max_tokens,
        # top_p=0.9
    )
    response = resp.choices[0].message.content
    # print(response)
    return response

    # return f'넘겨받은 문장은 {conversation}이며 감정은 {sentiment}입니다.'


# 메인 대화 루프
if __name__ == "__main__":
    history = []
    load_models_and_data()
    print("채팅 시작 (exit 입력 시 종료)")
    while True:
        usr = input("User: ")
        if usr.lower() == "exit":
            print("대화 종료")
            break
        cls = classify_emotion(usr)
        print(f"[감정 분류] {cls['sentiment']} ({cls['confidence']:.2f})")
        reply = generate_chat_response(history, cls['sentiment'])
        print("Assistant:", reply)
