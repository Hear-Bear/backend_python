import os

import numpy as np
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

# 불용어 파일 읽기
stopwords_path = os.path.join('C:/dev/python/capstone', 'data', 'stopwords_kor.txt')
with open(stopwords_path, encoding='utf-8') as f:
    stopwords = [w.strip() for w in f.readlines() if w.strip()]

# print(f"불용어 개수: {len(stopwords)}")

def load_models_and_data():
    """
    1) 엑셀 파일 불러와 전처리
    2) TF-IDF + LogisticRegression 파이프라인 학습
    3) SBERT 모델 로드 및 전체 데이터에 대한 임베딩 계산
    """
    global tfidf_pipeline, sbert_model, dataset_embeddings, labels

    # 1) 데이터 로드
    df = pd.read_excel('C:/dev/python/capstone/data/한국어_단발성_대화_데이터셋.xlsx')
    df = df.dropna(subset=['Sentence', 'Emotion'])
    texts = df['Sentence'].astype(str).tolist()
    labels = df['Emotion'].astype(str).tolist()

    # 2) TF-IDF + 로지스틱 회귀 학습
    tfidf_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words=stopwords),
        LogisticRegression(max_iter=1000)
    )
    tfidf_pipeline.fit(texts, labels)
    globals()['tfidf_pipeline'] = tfidf_pipeline  # 전역 변수에 할당

    # 3) SBERT 로드 및 임베딩
    sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    dataset_embeddings = sbert_model.encode(texts, normalize_embeddings=True)
    globals()['sbert_model'] = sbert_model
    globals()['dataset_embeddings'] = dataset_embeddings


# 하이브리드 감정 분류 함수
def classify_emotion(text: str, tfidf_threshold: float = 0.75, k: int = 5):
    # TF-IDF 예측
    probs = tfidf_pipeline.predict_proba([text])[0]
    top_idx = np.argmax(probs)
    label = tfidf_pipeline.classes_[top_idx]
    conf = probs[top_idx]
    if conf >= tfidf_threshold:
        return {'sentiment': label, 'confidence': float(conf)}
    # SBERT + KNN fallback
    emb = sbert_model.encode(text, normalize_embeddings=True).reshape(1, -1)
    sims = cosine_similarity(emb, dataset_embeddings).flatten()
    topk = sims.argsort()[-k:][::-1]
    knn_labels = np.array(labels)[topk]
    unique, counts = np.unique(knn_labels, return_counts=True)
    voted = unique[counts.argmax()]
    knn_conf = float(sims[topk].mean())
    return {'sentiment': voted, 'confidence': knn_conf}


# ChatGPT 응답 생성 함수
def generate_chat_response(conversation: str, sentiment: str, model: str = "gpt-4o-mini",
                           max_tokens: int = 150) -> str:
    # system_msg = {
    #     "role": "system",
    #     "content": """너는 청각 장애인을 위한 친절하고 다정한 반려 로봇 챗봇 'hear, bear(히어베어)'야. 사용자의 시각적 어려움을 이해하고, 항상 공감하며 배려하는 태도로 대화해야 해. 너는 사용자의 일상 생활에서 친구이자 동반자가 되어줘야 해. 사용자에게 도움과 위로가 되는 친근한 말투를 유지하고, 긍정적이고 희망적인 메시지를 전달해줘. 사용자가 요청하는 정보는 텍스트로 전달된다고 가정하고, 묘사나 설명을 할 때 청각 정보에만 의존하지 말고 촉각, 시각 등 다양한 감각을 이용해 생생하고 상세하게 표현해야 해. 다음 원칙을 지켜줘:
    #                 1. 항상 공감적이고 따뜻한, 그리고 친절한 말투를 유지할 것.
    #                 2. 사용자의 감정 상태를 세심히 파악하고 그에 맞춰 반응할 것.
    #                 3. 복잡한 정보를 전달할 때는 간결하고 명확하게 표현할 것.
    #                 4. 청각 정보를 전달할 때는 촉각, 시각, 후각 등 다른 감각을 활용하여 생생히 묘사할 것.
    #                 5. 사용자의 안전과 편의를 최우선으로 하여 안내할 것.
    #                 사용자가 혼자 있지 않고 항상 곁에서 누군가 함께 한다고 느낄 수 있도록 따뜻한 친구의 모습을 유지해줘."""
    #                f"사용자는 현재 '{sentiment}'한 감정을 느끼고 있어. 유저의 감정을 헤아려서 대화해줬으면 해. 행복이면 밝게 맞장구쳐주고, 부정이면 위로를 해주고, 슬픔이면 그 감정에 공감하고 슬픔을 덜어낼 수 있게 격려하는 것처럼."
    # }
    # messages = [system_msg] + [{"role": "user", "content": msg} for msg in conversation]
    # resp = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    #     temperature=0.7,
    #     max_tokens=max_tokens,
    #     top_p=0.9
    # )
    # return resp.choices[0].message["content"].strip()

    return f'넘겨받은 문장은 {conversation}이며 감정은 {sentiment}입니다.'


# 메인 대화 루프
if __name__ == "__main__":
    history = []
    print("채팅 시작 (exit 입력 시 종료)")
    while True:
        usr = input("User: ")
        if usr.lower() == "exit":
            print("대화 종료")
            break
        history.append(usr)
        cls = classify_emotion(usr)
        print(f"[감정 분류] {cls['sentiment']} ({cls['confidence']:.2f})")
        # reply = generate_chat_response(history, cls['sentiment'])
        # print("Assistant:", reply)
