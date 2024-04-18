from mpi4py import MPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# MPI setup
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

start_time = MPI.Wtime()
# Dummy prompt
prompt = "This is a test prompt."

# Scatter model names to all processes
model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    # "gpt2",
    "albert-base-v2"
]
tokenizer = None
model = None
if rank == 1:
    tokenizer = AutoTokenizer.from_pretrained(model_names[rank])
    model = AutoModelForSequenceClassification.from_pretrained(model_names[rank])
    print(f"Rank {rank}: {model_names[rank]}")
if rank == 2:
    tokenizer = AutoTokenizer.from_pretrained(model_names[rank])
    model = AutoModelForSequenceClassification.from_pretrained(model_names[rank])
    print(f"Rank {rank}: {model_names[rank]}")
if rank == 3:
    tokenizer = AutoTokenizer.from_pretrained(model_names[rank])
    model = AutoModelForSequenceClassification.from_pretrained(model_names[rank])
    print(f"Rank {rank}: {model_names[rank]}")

if rank!=0:
    # Compute scores
    input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    output = model(**input)
    score = output.logits.detach().numpy().tolist()[0]

# Gather scores to rank 0
all_scores = comm.gather(score, root=0)

# Master process aggregates scores through voting
if rank == 0:
    aggregated_scores = np.mean(all_scores, axis=0)
    print("Final Aggregated Result:", aggregated_scores[0])
    elapsed_time = MPI.Wtime() - start_time
    print("Elapsed Time:", elapsed_time)
else:
    # Send score to rank 0
    comm.send(score, dest=0)
