import pickle
from lib.config import cfg
import redis

def get_redis_cli():
    r = redis.StrictRedis(host=cfg.data.redis_host, port=cfg.data.redis_port, db=cfg.data.redis_db)
    return r
def get_list_range(redis_cli,name,l,r=-1):
    assert isinstance(redis_cli,redis.Redis)
    list = redis_cli.lrange(name,l,r)
    return [pickle.loads(d) for d in list]

if __name__ == '__main__':
    r = get_redis_cli()
    with open(cfg.data.train_data_buffer_path, 'rb') as data_dict:
        data_file = pickle.load(data_dict)
        data_buffer = data_file['data_buffer']
    for d in data_buffer:
        r.rpush(cfg.data.train_data_buffer_path,pickle.dumps(d))