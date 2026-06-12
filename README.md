# SRR
#### Requirements:
einops==0.7.0  
matplotlib==3.5.0  
pandas==2.2.1  
scipy==1.12.0  
seaborn==0.13.2  
torch==2.1.0  
torchaudio==2.1.0  
torchdata==0.7.1  
torchmetrics==1.2.1  
torchvision==0.16.0  
wandb==0.16.5

#### Datasets:
We used three public datasets in this study. 
- UCIHAR
- MFD
- SSC

#### Training procedure:
- Here, we provide a demo for running the experiments.  
To train a model using the following script file:
```
python trainers/train.py
