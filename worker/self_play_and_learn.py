from agent.mcts_self_play import mcts_self_play
from agent.mcts_learn import mcts_learn
from agent.config import Config

model_config = None
weight = None
model_config = "results/bababax/models/2020-04-04-10-32-50-mainNN.json"
weight = "results/bababax/models/2020-04-04-10-32-50-mainNN.h5"

# paths = [    
#     "results/bababax/kifu/2020-03-30-19-22-31.npz",
#     "results/bababax/kifu/2020-03-30-19-24-51.npz",
#     "results/bababax/kifu/2020-03-30-19-33-45.npz",
#     "results/bababax/kifu/2020-03-30-19-37-26.npz",
#     "results/bababax/kifu/2020-03-30-19-41-28.npz",
#     "results/bababax/kifu/2020-03-30-19-43-28.npz",
#     "results/bababax/kifu/2020-03-30-19-51-08.npz",
#     "results/bababax/kifu/2020-03-30-19-58-49.npz",
#     "results/bababax/kifu/2020-03-30-20-03-42.npz",
#     "results/bababax/kifu/2020-03-30-20-09-41.npz",
#     "results/bababax/kifu/2020-03-30-20-15-21.npz",
#     "results/bababax/kifu/2020-03-30-20-17-48.npz",
#     "results/bababax/kifu/2020-03-30-20-28-00.npz",
#     "results/bababax/kifu/2020-03-30-20-37-10.npz",
#     "results/bababax/kifu/2020-03-30-20-39-26.npz",
#     "results/bababax/kifu/2020-03-30-21-00-23.npz",
#     "results/bababax/kifu/2020-03-30-21-02-45.npz",
#     "results/bababax/kifu/2020-03-30-21-08-57.npz",
#     "results/bababax/kifu/2020-03-30-21-18-25.npz",
#     "results/bababax/kifu/2020-03-30-21-20-47.npz",
# ]
# paths = [
#     "results/bababax/kifu/2020-04-04-20-14-25.npz",
#     "results/bababax/kifu/2020-04-04-22-56-28.npz",
#     "results/bababax/kifu/2020-04-05-01-41-25.npz",
#     "results/bababax/kifu/2020-04-05-04-25-37.npz",
#     "results/bababax/kifu/2020-04-05-07-02-19.npz",
#     "results/bababax/kifu/2020-04-05-09-39-09.npz",
#     "results/bababax/kifu/2020-04-05-12-10-07.npz",
#     "results/bababax/kifu/2020-04-05-14-50-07.npz",
#     "results/bababax/kifu/2020-04-05-17-32-18.npz",
#     "results/bababax/kifu/2020-04-05-20-02-40.npz",
# ]

if __name__ == "__main__":
    config = Config(temperature=10000., c_puct=1.4, ignore_draw=True, learning_rate=0.001)
    mc = config.mcts
    config.learn_func = 'self_play_and_learn'
    config.tsumero = True
    config.all_random = True

    # model_config, weight, _ = mcts_learn(paths, config, model_config, weight)
    # exit()
    if config.tsumero:
        from tsumero.mcts_self_play_tsumero import mcts_self_play_tsumero
        mcts_self_play = mcts_self_play_tsumero

    for k in range(1000):
        config.n_period = k
        paths = []
        for _ in range(1):
            path = mcts_self_play(100, 40, model_config, weight, model_config, weight,
                                  mc.temperature, mc.n_playout, mc.c_puct, mc.ignore_draw, config.all_random)
            paths.append(path)
        model_config, weight, _ = mcts_learn(paths, config, model_config, weight)
