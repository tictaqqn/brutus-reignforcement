from typing import List
import math
import time
import copy
from threading import Thread, Lock
import numpy as np

from .uct_node import NodeHash, UctNode, UCT_HASH_SIZE, NOT_EXPANDED
from game.game_state import GameState, Winner
from agent.model_zero import ModelZero
from agent.config import Config

# UCBのボーナス項の定数
C_PUCT = 1.0
# 投了する勝率の閾値
RESIGN_THRESHOLD = 0.01
# Virtual Loss
VIRTUAL_LOSS = 1
# 探索スレッド数
THREAD_NUM = 4


def softmax_temperature_with_normalize(logits, temperature: float):
    # 温度パラメータを適用
    logits /= temperature

    # 確率を計算(オーバーフローを防止するため最大値で引く)
    max_logit = np.max(logits)
    probabilities = np.exp(logits - max_logit)

    # 合計が1になるように正規化
    sum_probabilities = np.sum(probabilities)
    probabilities /= sum_probabilities

    return probabilities


class PlayoutInfo:
    def __init__(self):
        self.halt = 0  # 探索を打ち切る回数
        self.count = 0  # 現在の探索回数


class MCTSPlayer:
    def __init__(self, my_side: int, temperature=100.0, n_playout=300):
        self.model = None  # モデル

        # ノードの情報
        self.node_hash = NodeHash()
        self.node_hash.initialize()
        self.uct_nodes = [UctNode() for _ in range(
            UCT_HASH_SIZE)]  # type: List[UctNode]

        # プレイアウト回数管理
        self.po_info = PlayoutInfo()
        self.playout = n_playout

        # 温度パラメータ
        self.temperature = temperature
        self.gs = GameState()

        # ロック
        self.lock_node = [Lock() for _ in range(UCT_HASH_SIZE)]
        self.lock_expand = Lock()  # ノード展開を排他処理するためのLock
        self.lock_po_info = Lock()

        # キュー
        self.current_queue_index = 0
        self.features = [[], []]
        self.hash_index_queues = [[], []]
        self.current_features = self.features[self.current_queue_index]
        self.current_hash_index_queue = self.hash_index_queues[self.current_queue_index]

        # スレッド数
        self.thread_num = THREAD_NUM

        if my_side == 1:
            self.my_side = Winner.plus
            self.other_side = Winner.minus
        elif my_side == -1:
            self.other_side = Winner.plus
            self.my_side = Winner.minus
        else:
            raise ValueError('Invalid my_side')

    def load_model(self, model_config_path, weight_path) -> None:
        self.model = ModelZero(config=Config())
        success_load = self.model.load(model_config_path, weight_path)
        if not success_load:
            raise FileNotFoundError(
                f"{model_config_path} {weight_path}が読み込めませんでした")

    def initialize_model(self) -> None:
        self.model = ModelZero(config=Config())
        self.model.build()

    def select_max_ucb_child(self, gs: GameState, current_node: UctNode):
        """UCB値が最大の手を求める"""
        child_num = current_node.child_num
        child_win = current_node.child_win
        child_move_count = current_node.child_move_count

        q = np.divide(child_win, child_move_count, out=np.repeat(
            np.float32(0.5), child_num), where=child_move_count != 0)
        u = np.sqrt(np.float32(current_node.move_count)) / \
            (1 + child_move_count)
        ucb = q + C_PUCT * current_node.nnrate * u

        return np.argmax(ucb)

    def expand_node(self, gs: GameState):
        """ノードの展開"""
        index = self.node_hash.find_same_hash_index(
            gs.board_hash(), gs.turn, gs.n_turns)

        # 合流先が検知できれば, それを返す
        if index != UCT_HASH_SIZE:
            return index

        # 空のインデックスを探す
        index = self.node_hash.search_empty_index(
            gs.board_hash(), gs.turn, gs.n_turns)

        # 現在のノードの初期化
        current_node = self.uct_nodes[index]
        current_node.move_count = 0
        current_node.win = 0.0
        current_node.child_num = 0
        current_node.evaled = False
        current_node.value_win = 0.0

        # 候補手の展開
        current_node.child_move = list(gs.generate_legal_moves())
        # print('new')
        # if 243 in current_node.child_move:
        #     print(gs)
        # print(current_node.child_move)
        child_num = len(current_node.child_move)
        current_node.child_index = [NOT_EXPANDED] * child_num
        current_node.child_move_count = np.zeros(child_num, dtype=np.int32)
        current_node.child_win = np.zeros(child_num, dtype=np.float32)

        # 子ノードの個数を設定
        current_node.child_num = child_num

        # ノードを評価
        if child_num > 0:
            self.eval_node(gs, index)
        else:
            current_node.value_win = 0.0
            current_node.evaled = True

        return index

    def interruption_check(self):
        """探索を打ち切るか確認"""
        child_num = self.uct_nodes[self.current_root].child_num
        child_move_count = self.uct_nodes[self.current_root].child_move_count
        rest = self.po_info.halt - self.po_info.count

        # 探索回数が最も多い手と次に多い手を求める
        second, first = child_move_count[np.argpartition(
            child_move_count, -2)[-2:]]

        # 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
        if first - second > rest:
            return True
        else:
            return False

    # 並列処理で呼び出す関数
    def parallel_uct_search(self):
        # 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
        while True:
            # 探索回数を1回増やす
            with self.lock_po_info:
                self.po_info.count += 1
            # 盤面のコピー
            gs = copy.deepcopy(self.gs)
            # 1回プレイアウトする
            self.uct_search(gs, self.current_root)
            # 探索を打ち切るか確認
            with self.lock_po_info:
                with self.lock_node[self.current_root]:
                    if self.po_info.count >= self.po_info.halt or self.interruption_check() or not self.node_hash.enough_size:
                        return

    def uct_search(self, gs: GameState, current):
        """UCT探索"""
        current_node = self.uct_nodes[current]

        # 詰みのチェック
        winner = gs.get_winner()
        if winner == self.my_side:
            return 1.0  # 反転して値を返すため1を返す
        if winner == self.other_side:
            return 0.0
        child_move = current_node.child_move
        child_move_count = current_node.child_move_count
        child_index = current_node.child_index

        # ニューラルネットワークが計算されるのを待つ
        while not current_node.evaled:
            time.sleep(0.000001)

        # 現在のノードをロック
        self.lock_node[current].acquire()

        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(gs, current_node)
        # print('push')
        # if child_move[next_index] == 243:
        #     print(gs)
        # print(child_move[next_index])
        # print(np.unravel_index(child_move[next_index], (7, 5, 9)))
        # print(list(gs.generate_legal_moves()))
        # 選んだ手を着手
        gs.move_with_id(child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        child_move_count[next_index] += VIRTUAL_LOSS

        # ノードの展開の確認
        if child_index[next_index] == NOT_EXPANDED:
            with self.lock_expand:
                # ノードの展開(ノード展開処理の中でノードをキューに追加する)
                index = self.expand_node(gs)
            child_index[next_index] = index

            # 現在見ているノードのロックを解除
            self.lock_node[current].release()

            # valueが計算されるのを待つ
            child_node = self.uct_nodes[index]
            while child_node.evaled == False:
                time.sleep(0.000001)

            # valueを勝敗として返す
            result = child_node.value_win
        else:
            # 現在見ているノードのロックを解除
            self.lock_node[current].release()
            # 手番を入れ替えて1手深く読む
            result = self.uct_search(gs, child_index[next_index])

        # 探索結果の反映
        with self.lock_node[current]:
            current_node.win += result
            current_node.move_count += 1 - VIRTUAL_LOSS
            current_node.child_win[next_index] += result
            current_node.child_move_count[next_index] += 1 - VIRTUAL_LOSS

        # 手を戻す
        # print('pop')
        # gs.pop()

        return result

    # キューをクリア
    def clear_eval_queue(self):
        self.current_queue_index = 0
        for i in range(2):
            self.features[i].clear()
            self.hash_index_queues[i].clear()
        self.current_features = self.features[self.current_queue_index]
        self.current_hash_index_queue = self.hash_index_queues[self.current_queue_index]

    # TODO: 未実装
    def eval_node(self, gs: GameState, index):
        """ノードを評価"""
        enough_batch_size = False
        while True:
            if not self.running:
                break

            self.lock_expand.acquire()
            if len(self.current_hash_index_queue) == 0:
                self.lock_expand.release()
                time.sleep(0.000001)
                continue

            if self.running and not enough_batch_size and len(self.current_hash_index_queue) < self.thread_num * 0.5:
                self.lock_expand.release()
                # キューが溜まるのを1回待つ
                time.sleep(0.000001)
                enough_batch_size = True
                continue

            enough_batch_size = False
            # 現在のキューを保存
            eval_features = self.current_features
            eval_hash_index_queue = self.current_hash_index_queue
            # カレントキューを入れ替える
            self.current_queue_index = self.current_queue_index ^ 1
            self.current_features = self.features[self.current_queue_index]
            self.current_hash_index_queue = self.hash_index_queues[self.current_queue_index]
            self.current_features.clear()
            self.current_hash_index_queue.clear()
            self.lock_expand.release()
            x = gs.to_inputs(flip=self.gs.turn == -1)

            logits, value = self.model.model.predict(x)
            if self.gs.turn == -1:
                logits[0] = GameState.flip_turn_outputs(logits[0])
            # logits = np.zeros(315)
            # value = 0.3
            for index, logits, value in zip(eval_hash_index_queue, logits_batch, values_batch):
                self.lock_node[index].acquire()
                current_node = self.uct_nodes[index]
                child_num = current_node.child_num
                child_move = current_node.child_move
                color = self.node_hash[index].color

                # 合法手でフィルター
                legal_move_labels = []
                for i in range(child_num):
                    legal_move_labels.append(
                        child_move[i])

                # Boltzmann分布
                probabilities = softmax_temperature_with_normalize(
                    logits[0, legal_move_labels], self.temperature)

                # ノードの値を更新
                current_node.nnrate = probabilities
                current_node.value_win = float(value)
                current_node.evaled = True
                self.lock_node[index].release()

    def usi(self):
        print('id name mcts_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('option name playout type spin default ' +
              str(self.playout) + ' min 100 max 10000')
        print('option name temperature type spin default ' +
              str(int(self.temperature * 100)) + ' min 10 max 1000')
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]
        elif option[1] == 'playout':
            self.playout = int(option[3])
        elif option[1] == 'temperature':
            self.temperature = int(option[3]) / 100

    # def isready(self):
    #     # モデルをロード
    #     if self.model is None:
    #         self.model = PolicyValueResnet()
    #         self.model.to_gpu()
    #     serializers.load_npz(self.modelfile, self.model)
    #     # ハッシュを初期化
    #     self.node_hash.initialize()
    #     print('readyok')

    def go(self):
        if self.gs.is_game_over():
            print('bestmove resign')
            return

        # 探索情報をクリア
        self.po_info.count = 0

        # 古いハッシュを削除
        self.node_hash.delete_old_hash(self.gs, self.uct_nodes)

        # 探索開始時刻の記録
        begin_time = time.time()

        # 探索回数の閾値を設定
        self.po_info.halt = self.playout

        # ルートノードの展開
        self.current_root = self.expand_node(self.gs)

        # 候補手が1つの場合は、その手を返す
        current_node = self.uct_nodes[self.current_root]
        child_num = current_node.child_num
        child_move = current_node.child_move
        if child_num == 1:
            print('bestmove', child_move[0])
            return

         # 探索実行中フラグを設定
        self.running = True

        # ニューラルネットワーク計算スレッド
        th_nn = Thread(target=self.eval_node)
        th_nn.start()
        # 探索スレッド
        threads = []
        for i in range(self.thread_num):
            th = Thread(target=self.parallel_uct_search)
            th.start()
            threads.append(th)

        # 探索スレッドを待機
        for th in threads:
            th.join()

        # 探索実行中フラグを解除
        self.running = False

        # ニューラルネットワーク計算スレッドを待機
        th_nn.join()

        # 探索にかかった時間を求める
        finish_time = time.time() - begin_time

        child_move_count = current_node.child_move_count
        if self.gs.n_turns < 10:
            # 訪問回数に応じた確率で手を選択する
            selected_index = np.random.choice(
                np.arange(child_num), p=child_move_count/np.sum(child_move_count))
        else:
            # 訪問回数最大の手を選択する
            selected_index = np.argmax(child_move_count)

        child_win = current_node.child_win

        # for debug
        for i in range(child_num):
            print('{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}'.format(
                i, child_move[i], child_move_count[i],
                current_node.nnrate[i],
                child_win[i] / child_move_count[i] if child_move_count[i] > 0 else 0))

        # 選択した着手の勝率の算出
        best_wp = child_win[selected_index] / child_move_count[selected_index]

        # 閾値未満の場合投了
        # if best_wp < RESIGN_THRESHOLD:
        #     print('bestmove resign')
        #     return

        bestmove = child_move[selected_index]

        # 勝率を評価値に変換
        # if best_wp >= 1.0 - 1e-7:
        #     cp = 30000
        # else:
        #     cp = int(-math.log(1.0 / best_wp - 1.0) * 600)

        print('info nps {} time {} nodes {} hashfull {} score wp {} pv {}'.format(
            int(current_node.move_count / finish_time),
            int(finish_time * 1000),
            current_node.move_count,
            int(self.node_hash.get_usage_rate() * 1000),
            best_wp, bestmove))

        print('bestmove', bestmove)

        arr = child_move_count_as_output_array_shape(
            child_move, child_move_count)

        return bestmove, best_wp, arr


def child_move_count_as_output_array_shape(child_move, child_move_count):
    arr = np.zeros(315, dtype=int)
    for i, c in zip(child_move, child_move_count):
        # if not plus_turn:
        #     i = GameState.flip_turn_outputs_index(i)
        arr[i] = c
    return arr


if __name__ == "__main__":
    player = MCTSPlayer()
    # player.load_model("results/002_QLearn_guard/2020-02-20-20-17-50-mainQN.json",
    #                   "results/002_QLearn_guard/2020-02-20-20-17-50-mainQN.h5")
    player.go()
