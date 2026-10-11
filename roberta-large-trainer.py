import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

# 读取数据
train = pd.read_csv('/kaggle/working/labeledTrainData.tsv',
                    delimiter='\t', quoting=3)
test = pd.read_csv('/kaggle/working/testData.tsv',
                   delimiter='\t', quoting=3)

# 划分训练集和验证集
train_data, val_data = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train['sentiment']
)

train_data = Dataset.from_dict({
    'text': train_data['review'].tolist(),
    'label': train_data['sentiment'].tolist()
})

val_data = Dataset.from_dict({
    'text': val_data['review'].tolist(),
    'label': val_data['sentiment'].tolist()
})

test_data = Dataset.from_dict({
    'text': test['review'].tolist()
})

# 加载 RoBERTa-Large 分词器
tokenizer = AutoTokenizer.from_pretrained(
    'FacebookAI/roberta-large'
)

# 对文本进行编码
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=256
    )

train_data = train_data.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

val_data = val_data.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

test_data = test_data.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

# 动态填充
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# 加载 RoBERTa-Large 分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    'FacebookAI/roberta-large',
    num_labels=2
)

# 显存优化
model.gradient_checkpointing_enable()

# 计算评价指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'f1': f1
    }

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type='linear',
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    eval_strategy='epoch',
    logging_strategy='steps',
    logging_steps=100,
    save_strategy='no',
    report_to='none',
    seed=42,
    data_seed=42,
    dataloader_num_workers=2
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 训练模型
trainer.train()

# 在验证集上评估模型
eval_results = trainer.evaluate()
print('验证集评估结果：', eval_results)

# 对测试集进行预测
prediction_outputs = trainer.predict(test_data)
predictions = np.argmax(
    prediction_outputs.predictions,
    axis=-1
).flatten()

# 保存预测结果
import os

os.makedirs('/kaggle/working/result', exist_ok=True)

# 去除ID两端多余的双引号和空格
test['id'] = test['id'].astype(str).str.strip().str.strip('"')

submission = pd.DataFrame({
    'id': test['id'],
    'sentiment': predictions.astype(int)
})

submission.to_csv(
    '/kaggle/working/result/roberta_large_trainer.csv',
    index=False
)

print('预测结果已保存！')
print(submission.head())
