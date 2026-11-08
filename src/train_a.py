import os
os.environ["USE_TF"] = "0"               # 强制禁止加载 TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================= 1. 路径与参数配置 =================
MODEL_NAME = "xlm-roberta-base"  # 多语言模型
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3

# 自动获取项目根目录和 data 目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "model_a")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 2. 数据加载与预处理 =================
def load_data():
    print("正在读取数据...")
    en_path = os.path.join(DATA_DIR, "StereoQueerEval_EN_training.tsv")
    nl_path = os.path.join(DATA_DIR, "StereoQueerEval_NL_training.tsv")
    il_path = os.path.join(DATA_DIR, "StereoQueerEval_IT_training.tsv")

    df_en = pd.read_csv(en_path, sep="\t")
    df_nl = pd.read_csv(nl_path, sep="\t")
    df_il = pd.read_csv(il_path, sep="\t")

    # 把英文和荷兰文合并在一起训练
    df = pd.concat([df_en, df_nl,df_il], ignore_index=True)

    # 拼接文本上下文：标题 + 描述 + 评论
    df["text"] = (
            "Title: " + df["yt_title"].fillna("") +
            " | Description: " + df["yt_description"].fillna("") +
            " | Comment: " + df["yt_comment"].fillna("")
    )

    # 任务 A 的标签是 stereotype 列 (no/yes)
    texts = df["text"].tolist()
    labels = df["stereotype"].tolist()
    return texts, labels


texts, raw_labels = load_data()

# 将文本标签转为数字（如 no->0, yes->1）
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(raw_labels)
num_labels = len(label_encoder.classes_)
print(f"标签映射: {dict(zip(label_encoder.classes_, range(num_labels)))}")

# 划分 80% 训练集，20% 验证集
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

# ================= 4. 训练模型 =================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


# 设置训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=16,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",  # 每个 epoch 评估一次
    save_strategy="epoch",  # 每个 epoch 保存一次
    load_best_model_at_end=True,  # 训练结束加载最好的一次
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # 有 GPU 自动加速
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

print("开始训练任务 A (刻板印象检测)...")
trainer.train()

# 保存最终模型和分词器
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成！模型已保存到 {OUTPUT_DIR}")
