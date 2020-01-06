from typing import Optional
from enum import IntEnum
from logging import getLogger
import time
import numpy as np
import wx
from wx.core import CommandEvent
from .errors import *
from .game_state import GameState

logger = getLogger(__name__)


def start():
    app = wx.App()
    Frame().Show()
    app.MainLoop()


def notify(caption, message):
    dialog = wx.MessageDialog(None, message=message,
                              caption=caption, style=wx.OK)
    dialog.ShowModal()
    dialog.Destroy()


class GameMode(IntEnum):
    humans_play = 1
    black_human_vs_random = 2
    white_human_vs_random = 3


class Frame(wx.Frame):

    def __init__(self) -> None:
        self.gs = GameState()
        self.logs = []
        self.piece_selected = False
        self.finished = False
        self.game_mode = GameMode.humans_play
        window_title = 'Brutus'
        window_size = (250, 400)
        wx.Frame.__init__(self, None, -1, window_title, size=window_size)
        # panel
        self.panel = wx.Panel(self)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.try_move)
        self.panel.Bind(wx.EVT_PAINT, self.refresh)

        menu = wx.Menu()
        menu.Append(1, u"New Game Humans")
        menu.Append(2, u"New Game (Black) vs random")
        menu.Append(3, u"New Game (White) vs random")
        menu.AppendSeparator()
        # menu.Append(5, u"Flip Vertical")
        # menu.Append(6, u"Show/Hide Player evaluation")
        # menu.AppendSeparator()
        menu.Append(9, u"quit")
        menu_bar = wx.MenuBar()
        menu_bar.Append(menu, u"menu")
        self.SetMenuBar(menu_bar)

        self.Bind(wx.EVT_MENU, self.handle_new_game, id=GameMode.humans_play)
        self.Bind(wx.EVT_MENU, self.handle_new_game,
                  id=GameMode.black_human_vs_random)
        self.Bind(wx.EVT_MENU, self.handle_new_game,
                  id=GameMode.white_human_vs_random)
        self.Bind(wx.EVT_MENU, self.handle_quit, id=9)

        # status bar
        self.CreateStatusBar()

    def handle_new_game(self, event):
        self.game_mode = event.GetId()
        print(self.logs)
        self.gs = GameState()
        self.logs = []
        self.finished = False
        self.piece_selected = False
        if self.game_mode == GameMode.humans_play or \
            self.game_mode == GameMode.black_human_vs_random:
            self.panel.Refresh()
        elif self.game_mode == GameMode.white_human_vs_random:
            self.gs.random_play()
            self.panel.Refresh()

    def try_move(self, event):
        if self.finished:
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
        except GameError as e:
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
        if self.game_mode == GameMode.black_human_vs_random or \
                self.game_mode == GameMode.white_human_vs_random:
            time.sleep(2)
            self.gs.random_play()
            self.panel.Refresh()

    def check_game_end(self, state: Optional[int]):
        if state == 1:
            print(self.gs)
            print("先手勝利")
            self.finished = True
            self.SetStatusText("先手勝利")
            print(self.logs)
        elif state == -1:
            print(self.gs)
            print("後手勝利")
            self.finished = True
            self.SetStatusText("後手勝利")
            print(self.logs)

    def update_status_bar(self):
        if self.finished:
            return
        msg = "current player is " + (
            "Black" if self.gs.turn == 1 else "White")
        self.SetStatusText(msg)

    def refresh(self, event):
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
        # TODO: キングのデザイン
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
                    # TODO: デザイン改善の余地
                    if self.piece_selected and \
                            j == self.selected_x and i == self.selected_y:
                        dc.SetBrush(wx.Brush("grey"))
                        dc.DrawRectangle(j * px + px/4,
                                         i * py + py/4, px/2, py/2)
        self.update_status_bar()

    def handle_quit(self, event: CommandEvent):
        self.Close()


if __name__ == "__main__":
    start()
