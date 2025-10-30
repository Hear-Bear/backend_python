# test.py
# 학습된 모델로 단일 파일 추론

import torch
import librosa
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def test_single_audio(model_path, audio_file, target_sr, id2label):
    """
    학습된 모델을 로드하여 단일 오디오 파일에 대한 추론을 실행.
    """
    print(f"\n--- 4. 테스트(추론) 시작: {audio_file} ---")

    # 사용 가능한 디바이스 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"추론 디바이스: {device}")

    try:
        model = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    except OSError:
        print(f"오류: {model_path}에서 모델을 찾을 수 없습니다.")
        return

    try:
        wav, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    except FileNotFoundError:
        print(f"오류: 테스트 파일 '{audio_file}'을 찾을 수 없습니다.")
        return

    inputs = feature_extractor(
        wav,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1024
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    scores = F.softmax(logits, dim=1).squeeze().cpu().tolist()
    prediction_idx = torch.argmax(logits, dim=1).item()
    predicted_label = id2label[prediction_idx]

    print(f"추론 결과: {predicted_label} (신뢰도: {scores[prediction_idx]:.4f})")

    print("--- 클래스별 신뢰도 ---")
    for i, score in enumerate(scores):
        print(f"{id2label[i]:<10}: {score:.4f}")

    print(f"--- 4. 테스트(추론) 완료 ---")
    return predicted_label, scores