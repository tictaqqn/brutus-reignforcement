from agent.mcts_self_play import mcts_self_play
from agent.mcts_learn import mcts_learn

model_config = None
weight = None
# model_config = "results/bababax/models/2020-02-26-12-17-48-mainNN.json"
# weight = "results/bababax/models/2020-02-26-12-17-48-mainNN.h5"
# model_config = "results/bababax/models/2020-03-09-02-13-23-mainNN.json"
# weight = "results/bababax/models/2020-03-09-02-13-23-mainNN.h5"
# path = "results/bababax/kifu/2020-03-09-02-13-04.npz"e
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
    for _ in range(1):
        paths = []
        for _ in range(10):
            path = mcts_self_play(10, 50, model_config, weight)
            paths.append(path)
        model_config, weight, _ = mcts_learn(paths, model_config, weight)
        
