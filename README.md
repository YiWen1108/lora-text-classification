# LoRA Text Classification (情緒分類)

This project demonstrates how to fine-tune a DistilBERT model using **LoRA (Low-Rank Adaptation)** for binary sentiment classification using the IMDb dataset.

本專案展示如何使用 LoRA 微調 DistilBERT 模型，實作 IMDb 電影評論的情緒二分類任務。

---

## 📦 Requirements (安裝需求)

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run (如何執行)

### Step 1: Train the model
```bash
python lora_finetune.py
```

### Step 2: Run sentiment prediction
```bash
python evaluate.py
```

---

## 🧠 Model

- Base model: `distilbert-base-uncased`
- Fine-tuning method: LoRA (via PEFT library)
- Dataset: IMDb (2000 samples)

---

## 📝 Output example

```json
[
  {"label": "POSITIVE", "score": 0.9998},
  {"label": "NEGATIVE", "score": 0.9983}
]
```

---

## 💡 Tip

This is a lightweight and fast training example. You can extend it to other tasks or datasets easily.

這是一個簡潔快速的訓練範例，適合用來學習 LoRA 微調，可進一步應用到其他任務或資料集。
