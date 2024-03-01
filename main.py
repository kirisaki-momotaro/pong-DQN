from agent import Agent
import os

from model import AtariNet

from environment import *

print(torch.cuda.is_available())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNEnvironment(device=device, render_mode='human')
#environment = DQNEnvironment(device=device)

model = AtariNet(nb_actions=6)
model.to(device)
agent = Agent(model=model, device=device, epsilon=1.0,
              nb_warmup=1000, nb_actions=6, learning_rate=0.00025, memory_capacity=10000,
              batch_size=128)

agent.train(env=environment, epochs=20000)
