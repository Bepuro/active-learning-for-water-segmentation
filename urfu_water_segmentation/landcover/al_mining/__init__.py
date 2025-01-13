from .registry import MINERS

# 2. ����������� ���� ������ (��������, ������� �������� score, ������ AL)
#    �� ��������, ��� � miners.py � ��� ����� ���� ��� ���������� score,
#    JsonWriter, ��� ������ ������ ��� AL.
from .miners import al_scores_single_gpu, JsonWriter

# 3. ����������� ���������� ������ "���������" ���������, 
#    ������� ����� �������� ��� `from al_mining import *`
__all__ = [
    'MINERS',
    'al_scores_single_gpu',
    'JsonWriter',
]