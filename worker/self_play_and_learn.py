from agent.mcts_self_play import mcts_self_play
from agent.mcts_learn import mcts_learn
from agent.config import Config

model_config = None
weight = None
model_config = "results/bababax/models/2020-03-29-10-28-38-mainNN.json"
weight = "results/bababax/models/2020-03-29-10-28-38-mainNN.h5"

# paths = [
#     "results/bababax/kifu/2020-02-26-11-42-06.npz",
#     "results/bababax/kifu/2020-02-26-12-16-17.npz",
#     "results/bababax/kifu/2020-02-26-12-17-42.npz",
#     "results/bababax/kifu/2020-03-08-20-14-13.npz",
#     "results/bababax/kifu/2020-03-08-20-24-58.npz",
#     "results/bababax/kifu/2020-03-09-02-13-04.npz",
#     "results/bababax/kifu/2020-03-09-13-10-41.npz",
# ]

if __name__ == "__main__":
    config = Config(temperature=100.)
    mc = config.mcts
    config.learn_func = 'self_play_and_learn'

    for k in range(1000):
        config.n_period = k
        paths = []
        for _ in range(10):
            path = mcts_self_play(10, 50, model_config, weight, model_config, weight,
                                  mc.temperature, mc.n_playout, mc.c_puct)
            paths.append(path)
        model_config, weight, _ = mcts_learn(paths, config, model_config, weight)
