from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd
import evaluate
import torch

# 1) 데이터 로드 및 전처리
df = pd.read_excel(
    'C:/dev/python/capstone/data/한국어_단발성_대화_데이터셋.xlsx'
).dropna(subset=['Sentence','Emotion'])[['Sentence','Emotion']]

label_list = sorted(df['Emotion'].unique())
class_label = ClassLabel(names=label_list)
df['label'] = df['Emotion'].map(lambda x: label_list.index(x))

dataset = Dataset.from_pandas(df[['Sentence','label']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset['train']
eval_ds = dataset['test']

# 2) 토크나이저 & 모델 로드
model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list)
)

# 3) 토크나이즈 함수
def preprocess(batch):
    return tokenizer(
        batch['Sentence'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

train_ds = train_ds.map(preprocess, batched=True)
eval_ds = eval_ds.map(preprocess, batched=True)
train_ds = train_ds.remove_columns(['Sentence'])
eval_ds = eval_ds.remove_columns(['Sentence'])
train_ds.set_format('torch')
eval_ds.set_format('torch')

# 4) 평가 지표 함수
metric_f1 = evaluate.load('f1')
def compute_metrics(pred):
    logits = pred.predictions
    preds  = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=preds, references=pred.label_ids, average='macro')

# 5) 트레이닝 설정
training_args = TrainingArguments(
    output_dir='./sbert_emotion',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

# 6) Trainer 생성 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 7) 모델 저장
trainer.save_model('./sbert_emotion/best_model')
torch.save(trainer.model.state_dict(), './sbert_emotion/fine-tuned_model.pt')
