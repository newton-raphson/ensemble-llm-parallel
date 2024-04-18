from mpi4py import MPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = MPI.Wtime()

# 500 dummy prompts (replace with your actual prompts)
prompts = ["This is a test prompt {}".format(i) for i in range(10)]

# Scatter model names to all processes
model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2"
]

# Ensure enough processes for the models
assert size >= len(model_names) + 1, "Not enough processes for the models!"

# Distribute models (excluding rank 0)
my_model_name = None
if rank != 0:
    my_model_name = model_names[(rank - 1) % len(model_names)]
    tokenizer = AutoTokenizer.from_pretrained(my_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(my_model_name)
    print(f"Rank {rank}: {my_model_name}")

if rank != 0:
    # Process a portion of the prompts
    my_prompts = np.array_split(prompts, size - 1)[rank - 1]

    print(f"Rank {rank}: Processing {len(my_prompts)} prompts")

    # Initialize aggregated scores for my prompts
    my_aggregated_scores = np.zeros(len(my_prompts))

    for i, prompt in enumerate(my_prompts):
        # Compute scores
        input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        output = model(**input)
        score = output.logits.detach().numpy()[0]

        # Store the score
        my_aggregated_scores[i] = score 

    # Gather scores from all processes to rank 0
    all_scores = comm.gather(my_aggregated_scores, root=0)

comm.Barrier()  

if rank == 0:
    # Combine scores from all processes
    all_scores = np.concatenate(all_scores)

    print("Final Aggregated Results:")
    print(all_scores)
    elapsed_time = MPI.Wtime() - start_time
    print("Elapsed Time:", elapsed_time)