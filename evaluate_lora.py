from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel, PeftConfig

# 讀取 LoRA 設定
config = PeftConfig.from_pretrained("./lora-model")

# 載入 base model（例如 DistilBERT）
base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)

# 載入 LoRA adapter
model = PeftModel.from_pretrained(base_model, "./lora-model")

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./lora-model")

# 建立推論 pipeline
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 測試句子
test_sentences = [
    "This movie was absolutely wonderful and inspiring.",
    "I hated every minute of it. Totally boring and predictable."
]

# 輸出結果
for sentence in test_sentences:
    result = pipe(sentence)[0]
    label = result['label']
    score = result['score']

    if label == 'LABEL_0':
        label = 'NEGATIVE'
    elif label == 'LABEL_1':
        label = 'POSITIVE'

    print(f"Input: {sentence}")
    print(f"Prediction: {label} (confidence: {score:.2%})\n")
