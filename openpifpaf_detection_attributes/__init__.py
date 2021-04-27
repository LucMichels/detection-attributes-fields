from . import datasets
from . import models
from . import fix


def register():
    datasets.register()
    models.register()
    fix.register()
