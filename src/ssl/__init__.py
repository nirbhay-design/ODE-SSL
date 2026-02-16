from .simsiam import SimSiamLoss
from .simclr import SimCLR
from .byol import BYOLLoss 
from .vicreg import VICRegLoss
from .barlow_twins import BarlowTwinLoss

loss_dict = {
    "simsiam": SimSiamLoss,
    "simclr": SimCLR,
    "byol": BYOLLoss,
    "vicreg": VICRegLoss,
    "bt": BarlowTwinLoss
}