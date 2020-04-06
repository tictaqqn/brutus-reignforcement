from agent.mcts_self_play import mcts_self_play
from agent.mcts_learn import mcts_learn
from agent.config import Config

model_config = None
weight = None
model_config = "results/bababax/models/2020-03-30-01-50-11-mainNN.json"
weight = "results/bababax/models/2020-03-30-01-50-11-mainNN.h5"

# paths = [
#     "results/bababax/kifu/2020-03-29-12-46-22.npz",
#     "results/bababax/kifu/2020-03-29-12-55-19.npz",
#     "results/bababax/kifu/2020-03-29-13-15-11.npz",
#     "results/bababax/kifu/2020-03-29-13-27-18.npz",
#     "results/bababax/kifu/2020-03-29-13-49-37.npz",
#     "results/bababax/kifu/2020-03-29-14-04-54.npz",
#     "results/bababax/kifu/2020-03-29-14-24-14.npz",
#     "results/bababax/kifu/2020-03-29-14-51-35.npz",
#     "results/bababax/kifu/2020-03-29-15-07-34.npz",
#     "results/bababax/kifu/2020-03-29-15-21-06.npz",
#     "results/bababax/kifu/2020-03-29-15-34-36.npz",
# ]

if __name__ == "__main__":
    config = Config(temperature=10000., c_puct=1.4, ignore_draw=True, learning_rate=0.001)
    mc = config.mcts
    config.learn_func = 'self_play_and_learn'

    for k in range(1000):
        config.n_period = k
        paths = []
        for _ in range(20):
            path = mcts_self_play(20, 40, model_config, weight, model_config, weight,
                                  mc.temperature, mc.n_playout, mc.c_puct, mc.ignore_draw)
            paths.append(path)
        model_config, weight, _ = mcts_learn(paths, config, model_config, weight)
