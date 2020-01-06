class GameError(Exception):
    """GameStateの送出するエラー"""
    pass


class ChoiceOfMovementError(GameError):
    """コマの移動が適切でない時のエラー"""
    pass
