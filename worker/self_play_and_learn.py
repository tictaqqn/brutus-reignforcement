from agent.mcts_self_play import mcts_self_play
from agent.mcts_learn import mcts_learn

if __name__ == "__main__":
    path = mcts_self_play(1, 5)
    for _ in range(1):
        model_config, weight, _ = mcts_learn([path])
        path = mcts_self_play(1, 5, model_config, weight)

    