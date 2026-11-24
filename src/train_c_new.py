"""
子任务 C（多标签分类）
一条评论可能归属多个标签：group_t, q, i, nb
"""
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import inspect

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ================= 1. 全局配置 =================
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 6  # 多标签任务稍微多训练几轮

# LLRD 超参数
ENCODER_LR = 2e-5
HEAD_LR = 1e-4
LAYER_DECAY = 0.90
WEIGHT_DECAY = 0.01

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "model_c_llrd")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 2. 数据加载与处理 =================
def load_data():
    print("正在加载数据，任务: C (多标签)")
    en_path = os.path.join(DATA_DIR, "StereoQueerEval_EN_training.tsv")
    nl_path = os.path.join(DATA_DIR, "StereoQueerEval_NL_training.tsv")
    it_path = os.path.join(DATA_DIR, "StereoQueerEval_IT_training.tsv")

    df = pd.concat([pd.read_csv(p, sep="\t") for p in [en_path, nl_path, it_path]], ignore_index=True)

    #注意：任务 C 只需要仇恨言论样本。过滤掉 target 为 "none" 的非仇恨样本
    # 因为对于非仇恨样本，没有什么目标好识别的。
    df = df[df["target"] != "none"].copy()

    # 拼接输入文本（Title + Description + Comment）
    df["text"] = (
            "Title: " + df["yt_title"].fillna("") +
            " | Description: " + df["yt_description"].fillna("") +
            " | Comment: " + df["yt_comment"].fillna("")
    )
    texts = df["text"].tolist()

    # 切分标签：原始标签是逗号分隔的字符串，如 "group_t,q,i"。
    # 我们需要把它按逗号切分成列表：["group_t", "q", "i"]
    raw_labels = df["target"].apply(lambda x: x.split(",")).tolist()

    return texts, raw_labels


texts, raw_labels = load_data()

#MultiLabelBinarizer（多标签二值化器），每一个位置代表一个独立的标签。1表示存在，0表示不存在
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(raw_labels)  # labels 的形状是 (样本数, 总标签类别数)
num_labels = len(mlb.classes_)
print(f"多标签总类别数: {num_labels}")
print(f"识别到的所有类别: {mlb.classes_}")

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ================= 3. 构建 Dataset =================
class HFDataset(Dataset):
    def __init__(self, texts, labels):
        # 文本统一分词
        self.encodings = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 【核心区别 4】：多标签的标签必须是 Float (浮点型) 类型，而不是 Long (长整型)。
        # 因为多标签分类的标签是一排 0.0 和 1.0，而不是一个确定的类别索引。
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = HFDataset(train_texts, train_labels)
val_dataset = HFDataset(val_texts, val_labels)

# ================= 4. 模型加载 =================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    # 【核心区别 5】：告诉模型这是多标签问题。这会让模型底层自动使用 BCEWithLogitsLoss，
    # 并且输出层不加 Softmax，而是使用 Sigmoid。
    problem_type="multi_label_classification"
)


# ================= 5. 导师的 LLRD 逻辑 =================
# 与 A/B 任务完全一致的 LLRD 实现
def create_llrd_parameter_groups(model, encoder_lr, head_lr, layer_decay, weight_decay):
    num_layers = len(model.roberta.encoder.layer)
    no_decay_tokens = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = []
    assigned_ids = set()

    def add_module(module, prefix, lr):
        for decay in (True, False):
            params = []
            for name, param in module.named_parameters():
                if not param.requires_grad: continue
                use_decay = not any(token in name for token in no_decay_tokens)
                if use_decay != decay: continue
                params.append(param)
                assigned_ids.add(id(param))
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": weight_decay if decay else 0.0})

    add_module(model.roberta.embeddings, "embeddings", encoder_lr * layer_decay ** num_layers)
    for layer_id, layer in enumerate(model.roberta.encoder.layer):
        lr = encoder_lr * layer_decay ** (num_layers - 1 - layer_id)
        add_module(layer, f"encoder.layer.{layer_id}", lr)
    if model.roberta.pooler is not None:
        add_module(model.roberta.pooler, "pooler", head_lr)
    add_module(model.classifier, "classifier", head_lr)
    return groups


class C_Trainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            groups = create_llrd_parameter_groups(self.model, ENCODER_LR, HEAD_LR, LAYER_DECAY, WEIGHT_DECAY)
            self.optimizer = torch.optim.AdamW(groups, lr=ENCODER_LR)
        return self.optimizer

    # 【注意】：在这里我们不需要重写 compute_loss，因为当我们在模型加载时指定了
    # problem_type="multi_label_classification"，HuggingFace 在底层会自动使用 BCEWithLogitsLoss


# ================= 6. 评估指标 =================
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # 【核心区别 6】：多标签的推理方式。
    # 单标签是用 np.argmax (挑出最大的那个)，多标签要把 logits 经过 Sigmoid 映射到 (0,1) 区间。
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # 设定阈值：概率大于 0.5，我们就认为该标签存在，置为 1；否则置为 0。
    preds = (probs > 0.5).astype(int)

    # 计算 F1
    # micro: 把所有类的 TP、FP、FN 加起来算总体的 F1（受高频类别影响大，但能反映总体水平）
    # macro: 各个类别的 F1 求平均（受稀有类别影响大，分数容易很低）
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }


# ================= 7. 训练配置 =================
args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=16,
    learning_rate=ENCODER_LR,
    weight_decay=WEIGHT_DECAY,
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    # 【注意】：多标签任务，官方一般更看重 micro_f1。所以我们根据 micro_f1 来保存最好模型
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    seed=42,
)

# 完美兼容新旧版本 transformers
arg_names = inspect.signature(TrainingArguments.__init__).parameters
args_kwargs["eval_strategy" if "eval_strategy" in arg_names else "evaluation_strategy"] = "epoch"
training_args = TrainingArguments(**args_kwargs)

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

# ================= 8. 启动训练 =================
trainer = C_Trainer(**trainer_kwargs)

print("开始训练任务 C (多标签分类 + LLRD)...")
trainer.train()

# 保存模型
trainer.save_model(OUTPUT_DIR)
# 保存 tokenizer（以后预测 FA 波斯语时要用）
tokenizer.save_pretrained(OUTPUT_DIR)
# 【极为重要】：我们需要把 mlb（标签处理字典）也保存下来，预测时才知道 0 对应哪个字母
import pickle

with open(os.path.join(OUTPUT_DIR, "mlb.pkl"), "wb") as f:
    pickle.dump(mlb, f)

print(f"任务 C 训练完成！模型已保存至 {OUTPUT_DIR}")
