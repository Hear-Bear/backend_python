# test.py
# 학습된 모델로 단일 파일 추론

import config
import torch
import librosa
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def test_single_audio(model_path, audio_file, target_sr, id2label):
    """
    학습된 모델을 로드하여 단일 오디오 파일에 대한 추론을 실행.
    """
    print(f"\n--- 테스트(추론) 시작: {audio_file} ---")

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
        max_length=config.MAX_AUDIO_LENGTH,
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    # sigmoid 적용
    probs = torch.sigmoid(logits).squeeze().cpu().tolist()

    print("\n--- 추론 결과 (신뢰도 50% 이상) ---")
    found = False
    # 50%가 넘는 모든 클래스 찾기
    predictions = []
    for i, prob in enumerate(probs):
        if prob > 0.7:
            predictions.append((id2label[i], prob))
            found = True

    if found:
        # 신뢰도 순으로 정렬하여 출력
        predictions.sort(key=lambda x: x[1], reverse=True)
        for label, prob in predictions:
            print(f"- {label}: {prob:.4f}")
    else:
        print("신뢰도 50%를 넘는 클래스를 찾지 못했습니다.")

    print("\n--- 클래스별 신뢰도 (전체) ---")
    all_scores = [(id2label[i], prob) for i, prob in enumerate(probs)]
    # 신뢰도 순으로 정렬
    all_scores.sort(key=lambda x: x[1], reverse=True)

    for label, prob in all_scores:
        # 너무 낮은 값은 제외하고 상위 10개 정도만 보거나...
        # 여기서는 일단 다 출력 (필요시 조절)
        print(f"{label:<25}: {prob:.4f}")

    print(f"\n--- 테스트(추론) 완료 ---")
    return predictions, probs