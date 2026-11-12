import os

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================= 1. 路径与参数配置 =================
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "model_b")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 2. 数据加载 =================
def load_data():
    print("正在读取数据...")
    en_path = os.path.join(DATA_DIR, "StereoQueerEval_EN_training.tsv")
    nl_path = os.path.join(DATA_DIR, "StereoQueerEval_NL_training.tsv")
    it_path = os.path.join(DATA_DIR, "StereoQueerEval_IT_training.tsv")

    df_en = pd.read_csv(en_path, sep="\t")
    df_nl = pd.read_csv(nl_path, sep="\t")
    df_it = pd.read_csv(it_path, sep="\t")

    df = pd.concat([df_en, df_nl, df_it], ignore_index=True)

    df["text"] = (
            "Title: " + df["yt_title"].fillna("") +
            " | Description: " + df["yt_description"].fillna("") +
            " | Comment: " + df["yt_comment"].fillna("")
    )

    texts = df["text"].tolist()
    labels = df["hate_speech"].tolist()
    return texts, labels


texts, raw_labels = load_data()

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(raw_labels)
num_labels = len(label_encoder.classes_)
print(f"标签映射: {dict(zip(label_encoder.classes_, range(num_labels)))}")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ================= 3. 构建 Dataset =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class HFDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = HFDataset(train_texts, train_labels)
val_dataset = HFDataset(val_texts, val_labels)

# ================= 4. 训练模型 & 加权计算 =================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# 自动计算类别权重
class_counts = np.bincount(train_labels)
total_samples = len(train_labels)
# 样本越少的类别，权重越大
class_weights = total_samples / (num_labels * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")
print(f"自动计算的类别权重: {class_weights}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # 【改动 2】：直接打印完整的分类报告，包含 no 的 F1
    print("\n" + "=" * 50)
    print(classification_report(labels, preds, target_names=label_encoder.classes_, zero_division=0))
    print("=" * 50 + "\n")

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "f1_no": report.get("0", {}).get("f1-score", 0.0),  # 非仇恨/支持
        "f1_explicit": report.get("1", {}).get("f1-score", 0.0),  # 显式仇恨
        "f1_implicit": report.get("2", {}).get("f1-score", 0.0)  # 隐式仇恨
    }


# 自定义带权重的 Trainer
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # 使用带有权重的交叉熵损失函数
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
)

# 自定义的 WeightedLossTrainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

print(" 开始训练任务 B (仇恨识别)...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成！模型已保存到 {OUTPUT_DIR}")
