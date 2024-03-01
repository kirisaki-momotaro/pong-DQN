import torch
import gym
import numpy as np
from PIL import Image
from agent import Agent
import os

from model import AtariNet

from environment import *

print(torch.cuda.is_available())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNEnvironment(device=device , render_mode='human')
#environment = DQNKungFuMaster(device=device)

model = AtariNet(nb_actions= 6)

model.to(device)

model.load_the_model('models/latest.pt')
print(device)
agent = Agent(model=model,device=device,epsilon=0.05,
              nb_warmup=500, nb_actions=6, learning_rate=0.00001,memory_capacity=200000,
                batch_size=256)

agent.test(env=environment)