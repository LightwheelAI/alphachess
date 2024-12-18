"""Policy Value Network"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import CONFIG
from torch.cuda.amp import autocast

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_filters)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_filters)
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_act(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        return self.conv2_act(residual + x)

# Backbone Network
class Net(nn.Module):
    def __init__(self, num_channels=256, num_res_blocks=7):
        super().__init__()
        self.conv_block = nn.Conv2d(9, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv_block_bn = nn.BatchNorm2d(num_channels)
        self.conv_block_act = nn.ReLU()
        
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 9 * 10, 2086)

        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 9 * 10, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Common Head
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)

        # Policy Head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 16 * 10 * 9])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy)
        
        # Value Head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * 10 * 9])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value

# Policy Value Network for Training
class PolicyValueNet:
    def __init__(self, model_file=None, use_gpu=True, device='cuda'):
        self.use_gpu = use_gpu
        self.device = device
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=2e-3)

        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))

    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        return act_probs, value.detach().cpu().numpy()

    def policy_value_fn(self, board):
        self.policy_value_net.eval()
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 9, 10, 9)).astype('float16')
        current_state = torch.as_tensor(current_state).to(self.device)

        with autocast():  # Mixed precision
            log_act_probs, value = self.policy_value_net(current_state)

        act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])

        return act_probs, value.detach().cpu().numpy()

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)

        self.optimizer.zero_grad()
        for params in self.optimizer.param_groups:
            params['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        value = value.view(-1)

        # Compute losses
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))

        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()

if __name__ == '__main__':
    net = Net().to('cuda')
    test_data = torch.ones([8, 9, 10, 9]).to('cuda')
    action_probs, value = net(test_data)
    print(action_probs.shape)  # (8, 2086)
    print(value.shape)         # (8, 1)