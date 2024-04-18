from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

#########################################################################################
#########################################################################################
############################# FLASK API FOR SCORING PROMPT ##############################
#########################################################################################
#########################################################################################

# Setup Flask app
app = Flask(__name__)

# Load tokenizer and models
model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    # "gpt2",
    "albert-base-v2"
]
all_scores=[]
# Define endpoint for scoring prompt
@app.route('/serial', methods=['POST'])
def score_prompt():
    data = request.get_json()
    prompt = data['prompt']
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print(model_name)
        input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        output = model(**input)
        score = output.logits.detach().numpy().tolist()[0]
        all_scores.append(score)

    aggregated_scores = np.mean(all_scores, axis=0)
    return jsonify({"score": aggregated_scores[0]})

@app.route('/parallel', methods=['POST'])
def score_prompt():
    data = request.get_json()
    prompt = data['prompt']
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print(model_name)
        input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        output = model(**input)
        score = output.logits.detach().numpy().tolist()[0]
        all_scores.append(score)

    aggregated_scores = np.mean(all_scores, axis=0)
    return jsonify({"score": aggregated_scores[0]})
if __name__ == '__main__':
    app.run(port=5000)


