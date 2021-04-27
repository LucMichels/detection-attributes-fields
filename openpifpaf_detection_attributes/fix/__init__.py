import openpifpaf

from .cocokp import CocoKp
from .dataset import CocoDataset


def register():
    openpifpaf.DATAMODULES['cocokp'] = CocoKp
