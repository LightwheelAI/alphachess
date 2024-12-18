"""Self-play data collection"""
import os
import time
import pickle
import random
import copy
from collections import deque

from lib.utils.game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from lib.utils import encoding
from lib.models.mcts import MCTSPlayer
from lib.config import cfg

if cfg.data.use_redis:
    import lib.utils.my_redis as my_redis

# Import based on the framework
if cfg.framework.name == 'paddle':
    from lib.models.paddle_net import PolicyValueNet
elif cfg.framework.name == 'pytorch':
    from lib.models.pytorch_net import PolicyValueNet
else:
    raise ValueError('The selected framework is not supported')

class GeneratePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.board = Board()
        self.game = Game(self.board)
        self.temp = 1  # Temperature
        self.n_playout = cfg.play_out  # Number of simulations for each move
        self.c_puct = cfg.c_puct  # Weight of U
        self.buffer_size = cfg.buffer_size  # Experience pool size
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.redis_cli = my_redis.get_redis_cli() if cfg.use_redis else None
        self.policy_value_net = self.load_model()

    def load_model(self):
        if cfg.framework.name == 'paddle':
            model_path = cfg.model.get('paddle_model', None)
        elif cfg.framework.name == 'pytorch':
            model_path = cfg.model.get('torch_model', None)
        else:
            raise NotImplementedError('The selected framework is not supported')
        
        if model_path is not None and os.path.exists(model_path):
            model = PolicyValueNet(model_file=model_path)
            print('Loaded the latest model')
        else:
            model = PolicyValueNet()
            print('Loaded the initial model')
        self.mcts_player = MCTSPlayer(model.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)
        return model

    def get_equi_data(self, play_data):
        """Augment the dataset by doubling it, speeding up training"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            extend_data.append(encoding.zip_state_mcts_prob((state, mcts_prob, winner)))
            state_flip, mcts_prob_flip = self.flip_state(state, mcts_prob)
            extend_data.append(encoding.zip_state_mcts_prob(state_flip, mcts_prob_flip, winner))
        return extend_data

    def flip_state(self, state, mcts_prob):
        """Flip the state and adjust probabilities"""
        state_flip = state.transpose([1, 2, 0])
        state_flip = state_flip[:, :, ::-1]  # Horizontal flip
        mcts_prob_flip = [mcts_prob[move_action2move_id[flip_map(move_id2move_action[i])]] for i in range(len(mcts_prob))]
        return state_flip, mcts_prob_flip

    def store_data(self, play_data):
        if self.cfg.use_redis:
            while True:
                try:
                    for d in play_data:
                        self.redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                    self.redis_cli.incr('iters')
                    self.iters = self.redis_cli.get('iters')
                    print("Storage completed")
                    break
                except Exception:
                    print("Storage failed")
                    time.sleep(1)
        else:
            self.load_local_data()
            self.data_buffer.extend(play_data)
            self.iters += 1
            self.save_local_data()

    def load_local_data(self):
        if os.path.exists(self.cfg.train_data_buffer_path):
            while True:
                try:
                    with open(self.cfg.train_data_buffer_path, 'rb') as data_file:
                        data_dict = pickle.load(data_file)
                        self.data_buffer.extend(data_dict['data_buffer'])
                        self.iters = data_dict['iters'] + 1
                    print('Successfully loaded data')
                    break
                except Exception:
                    time.sleep(30)

    def save_local_data(self):
        data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
        with open(self.cfg.train_data_buffer_path, 'wb') as data_file:
            pickle.dump(data_dict, data_file)

    def collect_selfplay_data(self, n_games=1):
        for _ in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=False)
            play_data = self.get_equi_data(list(play_data))
            self.store_data(play_data)
        return self.iters

    def run(self):
        """Start collecting data"""
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('batch i: {}, episode_len: {}'.format(iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == "__main__":
    collecting_pipeline = GeneratePipeline(cfg.data)
    collecting_pipeline.run()