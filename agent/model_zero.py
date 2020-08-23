from typing import *
import time
import datetime
from collections import deque
import hashlib
import json
import os
from logging import getLogger
import numpy as np
from tqdm import tqdm, trange
# noinspection PyPep8Naming
import keras.backend as K
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

from .config import Config, QLearnConfig
from game.game_state import GameState, Winner


logger = getLogger(__name__)

def objective_function_for_policy(y_true, y_pred):
    # can use categorical_crossentropy??
    return categorical_crossentropy(y_true, y_pred) - categorical_crossentropy(y_true, y_true) # for debug
    # return categorical_crossentropy(y_true, y_pred)

def objective_function_for_value(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def update_learning_rate(self, total_steps):
    # The deepmind paper says
    # ~400k: 1e-2
    # 400k~600k: 1e-3
    # 600k~: 1e-4

    lr = self.decide_learning_rate(total_steps)
    if lr:
        K.set_value(self.optimizer.lr, lr)
        logger.debug(f"total step={total_steps}, set learning rate to {lr}")


class ModelZero:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.digest = None

    def build(self) -> None:
        mc = self.config.model
        in_x = x = Input(shape=(7, 5, 3))

        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(mc.l2_reg))(x)
        x = Activation("relu")(x)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)
        x = Activation("relu")(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, data_format="channels_last",
                   kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'

        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
                 activation="relu")(x)
        policy_out = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
                 activation="relu")(x)
        policy_out = Dense(315, kernel_regularizer=l2(mc.l2_reg),
                    activation="softmax", name="policy_out")(policy_out)
        value_out = Dense(32, kernel_regularizer=l2(mc.l2_reg), activation="relu")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg),
                          activation="tanh", name="value_out")(value_out)

        self.model = Model(in_x, [policy_out, value_out], name="slipe_model")
        self.compile_model()
        # self.model.summary()

    def compile_model(self):
        self.optimizer = SGD(lr=self.config.model.learning_rate, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.compile(optimizer=self.optimizer, loss=losses)

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Dropout(1./3)(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(mc.l2_reg))(x)
        x = Add()([in_x, x])
        return x

    # 重みの学習
    def replay(self, wps, pi_mcts, board_logs, plus_turns, weights, batch_size: int, beta: float) -> None:
        inputs = np.zeros((batch_size, 7, 5, 3))
        policy_true = np.zeros((batch_size, 315))
        values_true = np.zeros((batch_size))
        input_weights = np.zeros((batch_size))
        indices = np.random.choice(
            np.arange(len(wps)), size=batch_size, replace=False)
        mini_batch = [(wps[i], pi_mcts[i], board_logs[i], plus_turns[i], weights[i]) for i in indices]

        for i, (winner, pi, board, plus_turn, weight) in enumerate(mini_batch):
            gs = GameState()
            gs.board = board
            inputs[i] = gs.to_inputs(flip=not plus_turn) # shape=(4, 5, 5)
            policy_true[i] = pi ** beta
            values_true[i] = winner
            input_weights[i] = weight

        # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        self.model.fit(inputs, [policy_true, values_true], sample_weight=input_weights, epochs=1, verbose=0, shuffle=True)

    @staticmethod
    def fetch_digest(weight_path: str):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path: str, weight_path: str) -> bool:
        if os.path.exists(weight_path):  # os.path.exists(config_path) and
            logger.debug(f"loading model from {config_path}")
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.compile_model()
            # self.model.summary()
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(
                f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path: str, weight_path: str) -> None:
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
        self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")
