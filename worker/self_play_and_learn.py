from agent.mcts_self_play import mcts_self_play
from agent.mcts_learn import mcts_learn

model_config = None#"results/bababax/models/2020-03-08-20-25-05-mainNN.json"
weight = None#"results/bababax/models/2020-03-08-20-25-05-mainNN.h5"

if __name__ == "__main__":
    path = mcts_self_play(100, 50, model_config, weight)
    for _ in range(1000):
        model_config, weight, _ = mcts_learn([path])
        path = mcts_self_play(100, 50, model_config, weight)

    