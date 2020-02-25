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
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.regularizers import l2

from .config import Config, QLearnConfig
from game.game_state import GameState, Winner


logger = getLogger(__name__)


class ModelZero:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.digest = None

    def build(self) -> None:
        mc = self.config.model
        in_x = x = Input((2, 7, 5))

        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first",
                   kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        policy_out = Dense(315, kernel_regularizer=l2(mc.l2_reg),
                    activation="softmax", name="policy_out")(x)

        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
                 activation="relu")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg),
                          activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="slipe_model")
        self.model.compile(loss='mse', optimizer=Adam(lr=mc.learning_rate))
        self.model.summary()

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

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
            self.model.compile(loss='mse',
                               optimizer=Adam(lr=self.config.model.learning_rate))
            self.model.summary()
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
