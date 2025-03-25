from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="./lora-model")
print(pipe("This movie was amazing and emotional!"))
print(pipe("This was boring, I almost fell asleep."))
