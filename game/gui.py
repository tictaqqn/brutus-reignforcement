from logging import getLogger
import wx
from wx.core import CommandEvent

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


class Frame(wx.Frame):

    def __init__(self):
        self.gs = GameState()
        window_title = 'Brutus'
        window_size = (400, 440)
        wx.Frame.__init__(self, None, -1, window_title, size=window_size)
        # panel
        self.panel = wx.Panel(self)
        # self.panel.Bind(wx.EVT_LEFT_DOWN, self.try_move)
        # self.panel.Bind(wx.EVT_PAINT, self.refresh)

        menu = wx.Menu()
        menu.Append(1, u"New Game(Black)")
        menu.Append(2, u"New Game(White)")
        menu.AppendSeparator()
        menu.Append(5, u"Flip Vertical")
        menu.Append(6, u"Show/Hide Player evaluation")
        menu.AppendSeparator()
        menu.Append(9, u"quit")
        menu_bar = wx.MenuBar()
        menu_bar.Append(menu, u"menu")
        self.SetMenuBar(menu_bar)

        # status bar
        self.CreateStatusBar()


if __name__ == "__main__":
    start()
