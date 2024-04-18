import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 500 dummy prompts (replace with your actual prompts)
prompt = ["This is a test prompt {}".format(i) for i in range(10)]

# Load tokenizer and models
model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    # "gpt2",
    "albert-base-v2"
]
all_scores=[]
start = time.time()

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(model_name)
    input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    output = model(**input)
    score = output.logits.detach().numpy().tolist()[0]
    all_scores.append(score)

aggregated_scores = np.mean(all_scores, axis=0)
print("Final Aggregated Result:", aggregated_scores[0])
end = time.time()
elapsed_time = end - start
print("Elapsed Time:", elapsed_time)
