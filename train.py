import random
from collections import defaultdict, deque
import numpy as np
import pickle
import time

import lib.utils.encoding as encoding
from lib.config import cfg
from lib.utils.game import Game, Board
from lib.models.mcts import MCTSPlayer
from lib.models.mcts_pure import MCTS_Pure

if cfg.data.use_redis:
    import lib.utils.my_redis as my_redis

# Load the model based on the framework
if cfg.framework.name == 'paddle':
    from lib.models.paddle_net import PolicyValueNet
elif cfg.framework.name == 'pytorch':
    from lib.models.pytorch_net import PolicyValueNet
else:
    raise ValueError("Unsupported framework: {}".format(cfg.framework.name))

class TrainPipeline:
    def __init__(self, init_model=None):
        self._initialize_parameters()
        self._load_model(init_model)
        self._initialize_game()

    def _initialize_parameters(self):
        """Initialize training parameters"""
        self.board = Board()
        self.n_playout = cfg.data.play_out
        self.c_puct = cfg.data.c_puct
        self.learn_rate = 1e-3
        self.lr_multiplier = 1
        self.temp = 1.0
        self.batch_size = cfg.train.batch_size
        self.epochs = cfg.train.epochs
        self.kl_targ = cfg.train.kl_targ
        self.check_freq = 100
        self.game_batch_num = cfg.train.game_batch_num
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 500
        self.buffer_size = cfg.data.buffer_size
        self.data_buffer = deque(maxlen=self.buffer_size)

        if cfg.data.use_redis:
            self.redis_cli = my_redis.get_redis_cli()

    def _load_model(self, init_model):
        """Load the model or initialize a new one"""
        try:
            if init_model:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print('Loaded the last trained model')
            else:
                self.policy_value_net = PolicyValueNet()
                print('Starting training from scratch')
        except Exception as e:
            print(f'Model path does not exist, starting training from scratch due to error: {e}')
            self.policy_value_net = PolicyValueNet()

    def _initialize_game(self):
        """Initialize the game"""
        self.game = Game(self.board)

    def policy_evaluate(self, n_games=10):
        """Evaluate the trained policy"""
        current_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                     c_puct=self.c_puct,
                                     n_playout=self.n_playout)
        pure_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_count = defaultdict(int)

        for i in range(n_games):
            winner = self.game.start_play(current_player, pure_player, start_player=i % 2 + 1, is_shown=1)
            win_count[winner] += 1

        win_ratio = (win_count[1] + 0.5 * win_count[-1]) / n_games
        print(f"num_playouts: {self.pure_mcts_playout_num}, win: {win_count[1]}, lose: {win_count[2]}, tie: {win_count[-1]}")
        return win_ratio

    def policy_update(self):
        """Update the policy value network"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        mini_batch = [encoding.recovery_state_mcts_prob(data) for data in mini_batch]

        state_batch = np.array([data[0] for data in mini_batch], dtype='float32')
        mcts_probs_batch = np.array([data[1] for data in mini_batch], dtype='float32')
        winner_batch = np.array([data[2] for data in mini_batch], dtype='float32')

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        self._adjust_learning_rate(kl)
        explained_var_old = self._explained_variance(winner_batch, old_v)
        explained_var_new = self._explained_variance(winner_batch, new_v)

        print(f"kl: {kl:.5f}, lr_multiplier: {self.lr_multiplier:.3f}, loss: {loss}, entropy: {entropy}, "
              f"explained_var_old: {explained_var_old:.9f}, explained_var_new: {explained_var_new:.9f}")
        return loss, entropy

    def _adjust_learning_rate(self, kl):
        """Adaptively adjust the learning rate"""
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

    def _explained_variance(self, winner_batch, value):
        """Calculate explained variance"""
        return 1 - np.var(winner_batch - value.flatten()) / np.var(winner_batch)

    def run(self):
        """Start training"""
        try:
            for i in range(self.game_batch_num):
                self._load_training_data()

                print(f'step {self.iters}: ')
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    self._save_model()

                time.sleep(cfg.train.train_update_interval)

                if (i + 1) % self.check_freq == 0:
                    print(f"current self-play batch: {i + 1}")
                    self.policy_value_net.save_model(f'models/current_policy_batch{i + 1}.model')
        except KeyboardInterrupt:
            print('\n\rquit')

    def _load_training_data(self):
        """Load training data"""
        if not cfg.data.use_redis:
            self._load_data_from_file()
        else:
            self._load_data_from_redis()

    def _load_data_from_file(self):
        """Load training data from file"""
        while True:
            try:
                with open(cfg.train.train_data_buffer_path, 'rb') as data_dict:
                    data_file = pickle.load(data_dict)
                    self.data_buffer = data_file['data_buffer']
                    self.iters = data_file['iters']
                    print('Data loaded successfully')
                break
            except Exception as e:
                print(f'Failed to load data: {e}')
                time.sleep(30)

    def _load_data_from_redis(self):
        """Load training data from Redis"""
        while True:
            try:
                l = len(self.data_buffer)
                data = my_redis.get_list_range(self.redis_cli, 'train_data_buffer', l if l == 0 else l - 1, -1)
                self.data_buffer.extend(data)
                self.iters = self.redis_cli.get('iters')
                if self.redis_cli.llen('train_data_buffer') > self.buffer_size:
                    self.redis_cli.lpop('train_data_buffer', self.buffer_size / 10)
                break
            except Exception as e:
                print(f'Failed to load data from Redis: {e}')
                time.sleep(5)

    def _save_model(self):
        """Save the model"""
        if cfg.framework.name == 'paddle':
            self.policy_value_net.save_model(cfg.model.paddle_model_path)
        elif cfg.framework.name == 'pytorch':
            self.policy_value_net.save_model(cfg.model.pytorch_model_path)
        else:
            print('Unsupported framework')

if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model=cfg.model)
    training_pipeline.run()