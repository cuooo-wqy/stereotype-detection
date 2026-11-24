"""
子任务A刻板印象（二分类）和子任务B仇恨言论（多分类）
基于XLM-R、LLRD(分层学习率衰减)
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
from sklearn.preprocessing import LabelEncoder
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
TASK = "B"  #任务："A" 或 "B"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5

#LLRD 超参数
ENCODER_LR = 2e-5
HEAD_LR = 1e-4
LAYER_DECAY = 0.90
WEIGHT_DECAY = 0.01

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", f"model_{TASK.lower()}_llrd")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= 2. 数据加载 =================
def load_data():
    print(f"正在加载数据，任务: {TASK}")
    en_path = os.path.join(DATA_DIR, "StereoQueerEval_EN_training.tsv")
    nl_path = os.path.join(DATA_DIR, "StereoQueerEval_NL_training.tsv")
    it_path = os.path.join(DATA_DIR, "StereoQueerEval_IT_training.tsv")

    df = pd.concat([pd.read_csv(p, sep="\t") for p in [en_path, nl_path, it_path]], ignore_index=True)
    df["text"] = (
            "Title: " + df["yt_title"].fillna("") +
            " | Description: " + df["yt_description"].fillna("") +
            " | Comment: " + df["yt_comment"].fillna("")
    )
    texts = df["text"].tolist()

    if TASK == "A":
        labels = df["stereotype"].tolist()
    elif TASK == "B":
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class HFDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = HFDataset(train_texts, train_labels)
val_dataset = HFDataset(val_texts, val_labels)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


# ================= 3. LLRD 逻辑 =================
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


class A_B_Trainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            groups = create_llrd_parameter_groups(self.model, ENCODER_LR, HEAD_LR, LAYER_DECAY, WEIGHT_DECAY)
            self.optimizer = torch.optim.AdamW(groups, lr=ENCODER_LR)

            # 如果是 B 任务，计算类别权重
            if TASK == "B":
                class_counts = np.bincount(train_labels)
                class_weights = len(train_labels) / (num_labels * class_counts)
                self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.args.device)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if TASK == "A":
            loss_fct = nn.CrossEntropyLoss()
        else:  # TASK == "B"
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "f1_implicit": report.get("2", {}).get("f1-score", 0.0),  # 仅对 B 任务有实际意义
        "f1_explicit": report.get("1", {}).get("f1-score", 0.0)
    }


# ================= 4. 训练 =================
args_kwargs = dict(
    output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=16, learning_rate=ENCODER_LR, weight_decay=WEIGHT_DECAY,
    logging_steps=50, save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", greater_is_better=True, fp16=torch.cuda.is_available(), seed=42,
)
arg_names = inspect.signature(TrainingArguments.__init__).parameters
args_kwargs["eval_strategy" if "eval_strategy" in arg_names else "evaluation_strategy"] = "epoch"
training_args = TrainingArguments(**args_kwargs)

trainer_kwargs = dict(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=compute_metrics,
)
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = A_B_Trainer(**trainer_kwargs)
print(f"开始训练任务 {TASK}...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"保存至 {OUTPUT_DIR}")
