# Parallelizing LLM Ensemble for large inference
LLMs has taken world by storm and zero-shot prediction and classification from such foundational model has shown to
perform well for variety of cases. In this project, we parallelize inference from multiple llms  which we take as constant. For prompts of large size we increase the number of processor to show some scaling.
## LOAD MPI
```module load intel,miniconda3```
## ACTIVATE
```conda env create -f environment.yml```
## RUN 
```python serial.py```</br>
```mpirun -np 6 python parallel.py```