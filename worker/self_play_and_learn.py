import datetime
import os
from agent.mcts_self_play import mcts_self_play
from agent.mcts_learn import mcts_learn
from agent.config import Config

model_config = None
weight = None
# model_config = "results/bekasa/2020-08-20-10-41/models/2020-08-20-14-23-57-mainNN.json"
# weight = "results/bekasa/2020-08-20-10-41/models/2020-08-20-14-23-57-mainNN.h5"

kifu_folders = []
kifu_folders = ['results/bekasa/2020-08-20-01-22/kifu',
                'results/bekasa/2020-08-20-10-41/kifu']

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
    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    config = Config(temperature=10000., n_playout=150, c_puct=1.4, ignore_draw=False)
    mc = config.mcts
    config.learn_func = 'self_play_and_learn'
    # model_config, weight, _ = mcts_learn(paths, config, model_config, weight)
    # exit()

    basedir = f'results/bekasa/{d}'
    os.makedirs(basedir + '/kifu')
    os.makedirs(basedir + '/models')

    paths = []
    for kifu_folder in kifu_folders:
        paths.extend([kifu_folder + '/' + kifu_file for kifu_file in os.listdir(kifu_folder)])

    for k in range(1):
        print(f'n_period:{k}')
        config.n_period = k
        for _ in range(1):
            path = mcts_self_play(1, 100, model_config, weight, model_config, weight,
                                  mc.temperature, mc.n_playout, mc.c_puct, mc.ignore_draw, folder=(basedir + '/kifu'), verbose=True)
            paths.append(path)
        model_config, weight, _ = mcts_learn(paths, config, model_config, weight, weight_reduction = 0.1, folder=(basedir + '/models'))
