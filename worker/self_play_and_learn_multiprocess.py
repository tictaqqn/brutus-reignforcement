import datetime
import os
import multiprocessing
import ctypes
import tensorflow as tf
from agent.mcts_self_play import mcts_self_play
from agent.mcts_learn import mcts_learn
from agent.config import Config

#メインプロセスでは対局と学習を交互に繰り返し、
#サブプロセスでは対局のみ行い棋譜データを溜めます

#グローバル変数は共有されないのでありません

# model_config = "results/bekasa/2020-08-17-21-15/models/2020-08-17-22-02-06-mainNN.json"
# weight = "results/bekasa/2020-08-17-21-15/models/2020-08-17-22-02-06-mainNN.h5"

# kifu_folders = ['results/bekasa/2020-08-17-18-22/kifu',
#                 'results/bekasa/2020-08-17-21-15/kifu']

#プロセス用メソッド
def process_func(global_model_config, global_weight, shared_queue, shared_lock, config, n_process):
    k = 0
    while (True):
        config.n_period = k
        with global_model_config.get_lock():
            model_config = global_model_config.value.decode()
        with global_weight.get_lock():
            weight = global_weight.value.decode()
        if (model_config == ''):
            model_config = None
        if (weight == ''):
            weight = None
        path = mcts_self_play(1, 100, model_config, weight, model_config, weight,
                              mc.temperature, mc.n_playout, mc.c_puct, mc.ignore_draw, folder=(basedir + '/kifu'), n_process=n_process, verbose=False)
        with shared_lock:
            shared_queue.put(path)
        k += 1


if __name__ == "__main__":
    # #tensorflow accerelation
    # tf.config.optimizer.set_jit(True)

    # configs
    kifu_folders = []
    kifu_folders = ['results/bekasa/2020-08-22-00-36/kifu',]
    n_processes = 6
    MAX_strlen = 256
    model_config = None
    weight = None
    model_config = "results/bekasa/2020-08-22-00-36/models/2020-08-22-15-32-59-mainNN.json"
    weight = "results/bekasa/2020-08-22-00-36/models/2020-08-22-15-32-59-mainNN.h5"

    global_model_config = multiprocessing.Array(ctypes.c_char, MAX_strlen)
    global_weight = multiprocessing.Array(ctypes.c_char, MAX_strlen)
    global_model_config.value = (model_config if model_config is not None else '').encode()
    global_weight.value = (weight if weight is not None else '').encode()
    shared_queue = multiprocessing.Queue()
    shared_lock = multiprocessing.Lock()

    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    config = Config(temperature=10000., n_playout=150, c_puct=1.4, ignore_draw=False)
    mc = config.mcts
    config.learn_func = 'self_play_and_learn'
    # model_config, weight, _ = mcts_learn(paths, config, model_config, weight)
    # exit()
    global basedir
    basedir = f'results/bekasa/{d}'
    os.makedirs(basedir + '/kifu', exist_ok=True)
    os.makedirs(basedir + '/models', exist_ok=True)

    processes = []
    for i in range (n_processes - 1):
        processes.append(multiprocessing.Process(target=process_func, args=(global_model_config, global_weight, shared_queue, shared_lock, config, i + 1), daemon=True))
        processes[i].start()

    paths = []
    for kifu_folder in kifu_folders:
        paths.extend([kifu_folder + '/' + kifu_file for kifu_file in os.listdir(kifu_folder)])

    for k in range(60):
        print(f'n_period:{k}')
        config.n_period = k
        for _ in range(1):
            path = mcts_self_play(1, 100, model_config, weight, model_config, weight,
                                  mc.temperature, mc.n_playout, mc.c_puct, mc.ignore_draw, folder=(basedir + '/kifu'), n_process=0, verbose=True)
            paths.append(path)
        with shared_lock:
            while not shared_queue.empty():
                try:
                    paths.append(shared_queue.get_nowait())
                except:
                    print('Error')
        while (len(paths) > 256):
            del paths[0]
        model_config, weight, _ = mcts_learn(paths, config, model_config, weight, weight_reduction = 0.1 / n_processes, folder=(basedir + '/models'))
        with global_model_config.get_lock():
            global_model_config.value = (model_config if model_config is not None else '').encode()
        with global_weight.get_lock():
            global_weight.value = (weight if weight is not None else '').encode()

    for i in range (n_processes - 1):
        processes[i].terminate()
