from .registry import MINERS

# 2. Импортируем вашу логику (например, функцию подсчёта score, классы AL)
#    Не забудьте, что в miners.py у вас лежат алго для вычисления score,
#    JsonWriter, или другие классы для AL.
from .miners import al_scores_single_gpu, JsonWriter

# 3. Опционально сформируем список "публичных" сущностей, 
#    которые будут доступны при `from al_mining import *`
__all__ = [
    'MINERS',
    'al_scores_single_gpu',
    'JsonWriter',
]