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
        window_size = (250, 400)
        wx.Frame.__init__(self, None, -1, window_title, size=window_size)
        # panel
        self.panel = wx.Panel(self)
        # self.panel.Bind(wx.EVT_LEFT_DOWN, self.try_move)
        self.panel.Bind(wx.EVT_PAINT, self.refresh)

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


if __name__ == "__main__":
    start()
