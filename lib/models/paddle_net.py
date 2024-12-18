"""Policy Value Network"""

import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F


# Residual Block
class ResBlock(nn.Layer):
    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2D(num_features=num_filters)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2D(num_features=num_filters)
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
class Net(nn.Layer):
    def __init__(self, num_channels=256, num_res_blocks=13):
        super().__init__()
        self.conv_block = nn.Conv2D(in_channels=9, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.conv_block_bn = nn.BatchNorm2D(num_features=num_channels)
        self.conv_block_act = nn.ReLU()

        # Global Feature Extraction
        self.global_conv = nn.Conv2D(in_channels=9, out_channels=512, kernel_size=(10, 9))
        self.global_bn = nn.BatchNorm1D(512)

        # Residual Blocks
        self.res_blocks = nn.LayerList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])

        # Policy Head
        self.policy_conv = nn.Conv2D(in_channels=num_channels, out_channels=16, kernel_size=1)
        self.policy_bn = nn.BatchNorm2D(16)
        self.policy_fc = nn.Linear(16 * 9 * 10, 2086)
        self.global_policy_fc = nn.Linear(512, 2086)
        self.policy_act = nn.ReLU()

        # Value Head
        self.value_conv = nn.Conv2D(in_channels=num_channels, out_channels=8, kernel_size=1)
        self.value_bn = nn.BatchNorm2D(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 9 * 10, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.global_value_fc = nn.Linear(512, 256)

    def forward(self, x):
        # Global Features
        global_x = self.global_conv(x)
        global_x = paddle.reshape(global_x, [-1, 512])
        global_x = self.global_bn(global_x)

        # Shared Feature Extraction
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)

        # Policy Head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = paddle.reshape(policy, [-1, 16 * 10 * 9])
        policy = self.policy_fc(policy)
        global_policy = self.policy_act(self.global_policy_fc(global_x))
        policy = F.log_softmax(policy + global_policy, axis=1)

        # Value Head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = paddle.reshape(value, [-1, 8 * 10 * 9])
        global_value = self.value_act1(self.global_value_fc(global_x))
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value + global_value)
        value = F.tanh(value)

        return policy, value


# Policy Value Network for Training
class PolicyValueNet:
    def __init__(self, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3  # L2 regularization
        self.policy_value_net = Net()
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=0.001,
            parameters=self.policy_value_net.parameters(),
            weight_decay=self.l2_const
        )
        if model_file:
            net_params = paddle.load(model_file)
            self.policy_value_net.set_state_dict(net_params)

    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = paddle.to_tensor(state_batch)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.numpy())
        return act_probs, value.numpy()

    def policy_value_fn(self, board):
        self.policy_value_net.eval()
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 9, 10, 9)).astype('float32')
        current_state = paddle.to_tensor(current_state)
        
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        
        return act_probs, value.numpy()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        paddle.save(net_params, model_file)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        state_batch = paddle.to_tensor(state_batch)
        mcts_probs = paddle.to_tensor(mcts_probs)
        winner_batch = paddle.to_tensor(winner_batch)

        self.optimizer.clear_gradients()
        self.optimizer.set_lr(lr)

        log_act_probs, value = self.policy_value_net(state_batch)
        value = paddle.reshape(value, shape=[-1])

        # Compute Losses
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -paddle.mean(paddle.sum(mcts_probs * log_act_probs, axis=1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        entropy = -paddle.mean(paddle.sum(paddle.exp(log_act_probs) * log_act_probs, axis=1))
        return loss.numpy(), entropy.numpy()[0]


if __name__ == '__main__':
    net = Net()
    test_data = paddle.ones([8, 9, 10, 9])
    x_act, x_val = net(test_data)
    print(x_act.shape)  # (8, 2086)
    print(x_val.shape)  # (8, 1)