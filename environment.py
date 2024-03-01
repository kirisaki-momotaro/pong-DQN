import collections
import cv2
import gym
import numpy as np
from numpy import asarray
import torch
from PIL import Image


class DQNEnvironment(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', device='cpu'):
        env = gym.make("ALE/Pong-v5", difficulty=1, render_mode=render_mode)

        super(DQNEnvironment, self).__init__(env)
        self.image_shape = (84, 84)
        self.device = device
        self.lives = env.ale.lives()
        self.total_rewards = 0

    def step(self, action):  # take step and get observation
        total_reward = 0
        done = False
        image = None

        for i in range(4):  # you dont really want to react on every frame, goup frames 4 at a time
            # print(action)
            observation, reward, done, trucated, info = self.env.step(action)
            self.total_rewards += reward
            if reward != 0:
                print(self.total_rewards)

            proccessed_img = Image.fromarray(observation)
            proccessed_img = proccessed_img.resize(self.image_shape)
            proccessed_img = proccessed_img.convert("L")
            if (i == 0):
                image = proccessed_img
            if done == True:
                image = proccessed_img
                self.total_rewards = 0
                break

        frame = image
        frame = np.array(frame)
        frame = torch.from_numpy(frame)
        frame = frame.unsqueeze(0)
        frame = frame.unsqueeze(0)
        frame = frame / 255.0
        frame = frame.to(self.device)

        self.total_rewards = torch.tensor(self.total_rewards).view(1, -1).float()
        self.total_rewards = self.total_rewards.to(self.device)

        done = torch.tensor(done).view(1, -1)
        done = done.to(self.device)

        return frame, self.total_rewards, done, info

    def reset(self):
        observation, _ = self.env.reset()
        proccessed_img = Image.fromarray(observation)
        proccessed_img = proccessed_img.resize(self.image_shape)
        proccessed_img = proccessed_img.convert("L")
        frame = proccessed_img
        frame = np.array(frame)
        frame = torch.from_numpy(frame)
        frame = frame.unsqueeze(0)
        frame = frame.unsqueeze(0)
        frame = frame / 255.0
        frame = frame.to(self.device)

        return frame
