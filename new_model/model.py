# model.py
# 모델 및 피처 추출기 로드 함수

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def load_model_and_extractor(model_id, num_labels, label2id, id2label):
    """
    사전 학습된 모델과 피처 추출기를 로드하고,
    분류 헤드를 교체하여 반환.
    """
    print("--- 2. 모델 및 피처 추출기 로드 ---")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
    return model, feature_extractor