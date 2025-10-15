import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchinfo import summary
import pickle

from data import PtychographyDataset
from model import PtychoViT
from training import *

import wandb
wandb.login()

datafile = '/home/beams/AILEENLUO/ptycho_simulation_factory/outputs/eagle/ptychodus_dp.hdf5'
dataset = PtychographyDataset(datafile)

generator0 = torch.Generator().manual_seed(8)
subsets = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=generator0)

# TODO: make config.yaml
NGPUS = 1
BATCH_SIZE = 32
LR = 1e-4 
print("GPUs:", NGPUS, "| Batch size:", BATCH_SIZE, "| Learning rate:", LR)
EPOCHS = 20
MODEL_SAVE_PATH = '/scratch/aileenluo/ptycho-vit/models'

trainloader = DataLoader(subsets[0], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
validloader = DataLoader(subsets[1], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Currently hard-coded, change to flexible layers set by config
model = PtychoViT()
# Had to do some tricks to get around DataParallel not moving complex tensors correctly, so this torchinfo summary will break
# dummy_data = torch.randn((1, 1, 512, 512))
# dummy_probe = torch.randn((1, 8, 512, 512))
# summary(model, input_data={'x': dummy_data, 'probe': dummy_probe, 
#         'normalization': torch.randn((1, 1)), 'scale': torch.randn((1, 1))}, device='cpu')

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NGPUS > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)

metrics = {'training_loss': [], 'train_amp_loss': [], 'train_ph_loss': [], 'validation_loss': [], 
           'val_amp_loss': [], 'val_ph_loss': [], 'best_val_loss': np.inf}

run = wandb.init(
    entity="SYNAPS-I",
    project="PtychoViT",
    config={
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "dataset": 'eagle',
        "epochs": EPOCHS,
        "notes": "Prototype"
    }
)

trainer = Trainer(model, 42, DEVICE, MODEL_SAVE_PATH)

print('Training')
for epoch in range(EPOCHS):
    # Training loop
    model.train()
    trainer.train(trainloader, criterion, optimizer, metrics)

    # Validation loop
    model.eval()
    if epoch % 5 == 0: 
        trainer.validate(validloader, criterion, optimizer, metrics, plot=True)
    else: 
        trainer.validate(validloader, criterion, optimizer, metrics, plot=False)

    print('Epoch: %d | Train Loss: %.4f | Val. Loss: %.4f' 
          %(epoch, metrics['training_loss'][-1], metrics['validation_loss'][-1]))

trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, scheduler=None)

with open(os.path.join(MODEL_SAVE_PATH, 'metrics.pickle'), 'wb') as file:
    pickle.dump(metrics, file)

print('Finished Training')
wandb.finish()