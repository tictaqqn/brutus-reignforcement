from typing import Optional
from enum import IntEnum
from logging import getLogger
import numpy as np
import wx
from wx.core import CommandEvent
from .errors import ChoiceOfMovementError, GameError
from .game_state import GameState, Winner
from agent.model import QNetwork
from agent.config import Config
from uct.mcts import MCTSPlayer
from uct.dfpn import dfpn

logger = getLogger(__name__)

MODEL_CONFIG_PATH = "results/002_QLearn_guard/2020-02-20-20-17-50-mainQN.json"
WEIGHT_PATH = "results/002_QLearn_guard/2020-02-20-20-17-50-mainQN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-02-26-12-17-48-mainNN.json"
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-02-26-12-17-48-mainNN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-03-09-02-13-23-mainNN.json"
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-03-09-02-13-23-mainNN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-03-10-13-21-13-mainNN.json"
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-03-10-13-21-13-mainNN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-03-30-16-12-58-mainNN.json"
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-03-30-16-12-58-mainNN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-04-04-10-32-50-mainNN.json"
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-04-04-10-32-50-mainNN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-04-06-10-48-35-mainNN.json" # lr=0.001 from 4/4
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-04-06-10-48-35-mainNN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-04-06-10-57-07-mainNN.json" # lr=0.0001 from 4/4
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-04-06-10-57-07-mainNN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-04-06-13-55-43-mainNN.json" # lr=0.0001 3rd
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-04-06-13-55-43-mainNN.h5"
# MODEL_CONFIG_PATH_ZERO = "results/bababax/models/2020-04-07-12-29-07-mainNN.json" # lr=0.0001 3rd
# WEIGHT_PATH_ZERO = "results/bababax/models/2020-04-07-12-29-07-mainNN.h5"
MODEL_CONFIG_PATH_ZERO = "results/bekasa/2020-08-22-00-36/models/2020-08-22-15-32-59-mainNN.json"
WEIGHT_PATH_ZERO = "results/bekasa/2020-08-22-00-36/models/2020-08-22-15-32-59-mainNN.h5"


def start() -> None:
    app = wx.App()
    Frame().Show()
    app.MainLoop()


def notify(caption, message) -> None:
    dialog = wx.MessageDialog(None, message=message,
                              caption=caption, style=wx.OK)
    dialog.ShowModal()
    dialog.Destroy()


class GameMode(IntEnum):
    humans_play = 1
    black_human_vs_random = 2
    white_human_vs_random = 3
    black_human_vs_QL = 4
    black_human_vs_Zero = 6
    white_human_vs_Zero = 7

    quit = 9


class Frame(wx.Frame):

    def __init__(self) -> None:
        self.gs = GameState()
        self.logs = []
        self.piece_selected = False
        self.finished = False
        self.CPU_thinking = False
        self.game_mode = GameMode.humans_play
        window_title = 'Brutus'
        window_size = (250, 400)
        wx.Frame.__init__(self, None, -1, window_title, size=window_size)
        # panel
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.try_move)
        self.panel.Bind(wx.EVT_PAINT, self.refresh)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer)

        menu = wx.Menu()
        menu.Append(GameMode.humans_play, u"New Game Humans")
        menu.Append(GameMode.black_human_vs_random,
                    u"New Game (Black) vs random")
        menu.Append(GameMode.white_human_vs_random,
                    u"New Game (White) vs random")
        menu.Append(GameMode.black_human_vs_QL,
                    u"New Game (Black) vs QL")
        menu.Append(GameMode.black_human_vs_Zero,
                    u"New Game (Black) vs Zero")
        menu.Append(GameMode.white_human_vs_Zero,
                    u"New Game (White) vs Zero")
        menu.AppendSeparator()
        # menu.Append(5, u"Flip Vertical")
        # menu.Append(6, u"Show/Hide Player evaluation")
        # menu.AppendSeparator()
        menu.Append(GameMode.quit, u"quit")
        menu_bar = wx.MenuBar()
        menu_bar.Append(menu, u"menu")
        self.SetMenuBar(menu_bar)

        self.Bind(wx.EVT_MENU, self.handle_new_game,
                  id=GameMode.humans_play)
        self.Bind(wx.EVT_MENU, self.handle_new_game,
                  id=GameMode.black_human_vs_random)
        self.Bind(wx.EVT_MENU, self.handle_new_game,
                  id=GameMode.white_human_vs_random)
        self.Bind(wx.EVT_MENU, self.handle_new_game,
                  id=GameMode.black_human_vs_QL)
        self.Bind(wx.EVT_MENU, self.handle_new_game,
                  id=GameMode.black_human_vs_Zero)
        self.Bind(wx.EVT_MENU, self.handle_new_game,
                  id=GameMode.white_human_vs_Zero)
        self.Bind(wx.EVT_MENU, self.handle_quit,
                  id=GameMode.quit)

        # status bar
        self.CreateStatusBar()

    def handle_new_game(self, event) -> None:
        self.game_mode = event.GetId()
        print(self.logs)
        self.gs = GameState()
        self.logs = []
        self.finished = False
        self.CPU_thinking = False
        self.piece_selected = False
        if self.game_mode == GameMode.humans_play or \
                self.game_mode == GameMode.black_human_vs_random:
            self.panel.Refresh()
        elif self.game_mode == GameMode.white_human_vs_random:
            self.gs.random_play()
            self.panel.Refresh()
        elif self.game_mode == GameMode.black_human_vs_QL:
            self.model = QNetwork(config=Config())
            success_load = self.model.load(MODEL_CONFIG_PATH, WEIGHT_PATH)
            if not success_load:
                raise FileNotFoundError(
                    f"{MODEL_CONFIG_PATH} {WEIGHT_PATH}が読み込めませんでした")
            self.panel.Refresh()
        elif self.game_mode == GameMode.black_human_vs_Zero:
            self.player = MCTSPlayer(-1)
            self.player.load_model(MODEL_CONFIG_PATH_ZERO,
                                   WEIGHT_PATH_ZERO)
            self.panel.Refresh()
        elif self.game_mode == GameMode.white_human_vs_Zero:
            self.player = MCTSPlayer(1)
            self.player.load_model(MODEL_CONFIG_PATH_ZERO,
                                   WEIGHT_PATH_ZERO)
            self.panel.Refresh()
            self.OnTimer(0)

    def try_move(self, event):
        if self.finished or self.CPU_thinking:
            return
        event_x, event_y = event.GetX(), event.GetY()
        w, h = self.panel.GetSize()

        if not self.piece_selected:
            x = int(event_x / (w / 5))
            y = int(event_y / (h / 7))
            if self.gs.board[y, x] == self.gs.turn:
                self.selected_x, self.selected_y = x, y
                self.piece_selected = True
                self.panel.Refresh()
            return

        x = int(event_x / (w / 5))
        y = int(event_y / (h / 7))
        if x == self.selected_x and y == self.selected_y:
            self.piece_selected = False
            self.panel.Refresh()
            return

        if not (-1 <= x - self.selected_x <= 1 and
                -2 <= y - self.selected_y <= 2):
            self.piece_selected = False
            self.panel.Refresh()
            return

        d = np.array([y - self.selected_y, x - self.selected_x])
        print(d)
        try:
            state = self.gs.move_d_vec(self.selected_y, self.selected_x, d)
        except ChoiceOfMovementError as e:
            print(e)
            print("入力が不正です。もう一度入力してください。")
            self.piece_selected = False
            self.panel.Refresh()
            return
        # print(self.gs)
        self.logs.append((y, x, d))
        self.check_game_end(state)
        self.piece_selected = False
        self.panel.Refresh()
        if self.finished:
            return
        if self.game_mode != GameMode.humans_play:
            self.timer.Start(500)  # 500ms後OnTimer()が反応
            self.CPU_thinking = True
            # self.gs.random_play()
            # self.panel.Refresh()

    def OnTimer(self, event) -> None:
        if self.game_mode in [
            GameMode.black_human_vs_random,
            GameMode.white_human_vs_random
        ]:
            state, _ = self.gs.random_play()
        elif self.game_mode == GameMode.black_human_vs_QL:
            retTargetQs = self.model.model.predict(self.gs.to_inputs())[0]
            state, _ = self.gs.outputs_to_move_max(retTargetQs)
        elif self.game_mode in [
            GameMode.black_human_vs_Zero,
            GameMode.white_human_vs_Zero
        ]:
            dfpn_r = dfpn(self.gs)
            if dfpn_r is not None:
                best_action = dfpn_r
            else:
                self.player.gs.board = self.gs.board.copy()
                self.player.gs.turn = self.gs.turn
                self.player.gs.n_turns = self.gs.n_turns
                best_action, best_wp, arr = self.player.go()
            state = self.gs.move_with_id(best_action)
        self.check_game_end(state)
        self.panel.Refresh()
        self.timer.Stop()
        self.CPU_thinking = False

    def check_game_end(self, state: Winner) -> None:
        if state == Winner.plus:
            print(self.gs)
            print("先手勝利")
            self.finished = True
            self.SetStatusText("先手勝利")
            print(self.logs)
        elif state == Winner.minus:
            print(self.gs)
            print("後手勝利")
            self.finished = True
            self.SetStatusText("後手勝利")
            print(self.logs)

    def update_status_bar(self) -> None:
        if self.finished:
            return
        if self.CPU_thinking:
            msg = "CPUが考慮中です"
        else:
            msg = "現在の手番: " + (
                "黒" if self.gs.turn == 1 else "白")
        self.SetStatusText(msg)

    def refresh(self, event) -> None:
        dc = wx.PaintDC(self.panel)
        # self.update_status_bar()

        w, h = self.panel.GetSize()
        # background
        dc.SetBrush(wx.Brush("light gray"))
        dc.DrawRectangle(0, 0, w, h)
        # grid
        dc.SetBrush(wx.Brush("black"))
        px, py = w / 5, h / 7
        for y in range(8):
            dc.DrawLine(y * px, 0, y * px, h)
            dc.DrawLine(0, y * py, w, y * py)
        dc.DrawLine(w - 1, 0, w - 1, h - 1)
        dc.DrawLine(0, h - 1, w - 1, h - 1)

        # stones
        brushes = {
            -2: wx.Brush("white"),
            -1: wx.Brush("white"),
            1: wx.Brush("black"),
            2: wx.Brush("black"),
        }

        for i in range(7):
            for j in range(5):
                c = self.gs.board[i, j]
                if c != 0:
                    dc.SetBrush(brushes[c])
                    dc.DrawEllipse(j * px, i * py, px, py)

        king_black_image = wx.Image(
            'game/pictures/king_black.png').Scale(px*0.8, py*0.8)
        king_white_image = wx.Image(
            'game/pictures/king_white.png').Scale(px*0.8, py*0.8)
        king_black_bitmap = king_black_image.ConvertToBitmap()
        king_white_bitmap = king_white_image.ConvertToBitmap()
        dc.DrawBitmap(king_black_bitmap, 2.1*px, 6.1*py, True)
        dc.DrawBitmap(king_white_bitmap, 2.1*px, 0.1*py, True)

        if self.piece_selected:
            j = self.selected_x
            i = self.selected_y
            c = self.gs.board[i, j]
            dc.SetBrush(wx.Brush("grey"))
            dc.DrawRectangle(j * px + px/4,
                             i * py + py/4, px/2, py/2)

            # 隣あう８方向のうち合法手に印
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    if 0 <= i + dx < 7 and 0 <= j + dy < 5 and \
                            self.gs.board[i+dx, j+dy] == 0:
                        dc.DrawRectangle((j + dy) * px + px/4,
                                         (i + dx) * py + py/4, px/2, py/2)

            # 先手初手以外かつ前に自駒がなければ前に２マス進める
            if self.gs.n_turns != 0 and 0 <= i-2*c < 7 and \
                    self.gs.board[i-c, j] != c and self.gs.board[i-2*c, j] == 0:
                dc.DrawRectangle(j * px + px/4,
                                 (i - c*2) * py + py/4, px/2, py/2)

        self.update_status_bar()

    def handle_quit(self, event: CommandEvent) -> None:
        self.Close()


if __name__ == "__main__":
    start()
