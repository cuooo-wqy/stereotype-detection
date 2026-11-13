import os

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "model_c")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    print("正在读取数据...")
    en_path = os.path.join(DATA_DIR, "StereoQueerEval_EN_training.tsv")
    nl_path = os.path.join(DATA_DIR, "StereoQueerEval_NL_training.tsv")
    it_path = os.path.join(DATA_DIR, "StereoQueerEval_IT_training.tsv")

    df_en = pd.read_csv(en_path, sep="\t")
    df_nl = pd.read_csv(nl_path, sep="\t")
    df_it = pd.read_csv(it_path, sep="\t")
    df = pd.concat([df_en, df_nl, df_it], ignore_index=True)

    # 任务 C 只针对仇恨言论（非仇恨的 target 都是 none）
    # 过滤掉 'none' 的样本进行训练
    df = df[df["target"] != "none"].copy()

    df["text"] = (
            "Title: " + df["yt_title"].fillna("") +
            " | Description: " + df["yt_description"].fillna("") +
            " | Comment: " + df["yt_comment"].fillna("")
    )

    texts = df["text"].tolist()

    # 把 "group_t,q" 这种字符串按逗号拆分成列表 ["group_t", "q", "group_l..."]
    # 为了简单处理，我们直接把冒号前后的词作为一个整体类别（比如 group_t）
    labels = df["target"].apply(lambda x: x.split(",")).tolist()
    return texts, labels


texts, raw_labels = load_data()

# 多标签二值化：把 ["group_t", "q"] 变成 0/1 向量
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(raw_labels)
num_labels = len(mlb.classes_)
print(f"多标签总类别数: {num_labels}")
print(f"类别字典: {mlb.classes_}")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class HFDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 多标签的 y 需要是 float 类型
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = HFDataset(train_texts, train_labels)
val_dataset = HFDataset(val_texts, val_labels)

# 这里的 problem_type="multi_label_classification" 非常重要
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 多标签通常用 sigmoid 计算概率，阈值设为 0.5
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.3).astype(int)

    # 计算 micro 和 macro F1
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=16,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",  # 多标签通常看 micro_f1
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

print("开始训练任务 C (多标签 Target 识别)...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"任务 C 训练完成！模型已保存到 {OUTPUT_DIR}")
