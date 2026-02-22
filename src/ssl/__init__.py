from .simsiam import *
from .simclr import *
from .byol import * 
from .vicreg import *
from .barlow_twins import *

loss_dict = {
    "simsiam": SimSiamLoss,
    "simclr": SimCLR,
    "byol": BYOLLoss,
    "vicreg": VICRegLoss,
    "bt": BarlowTwinLoss,
    "simsiam-sc": SimSiamLoss,
    "scalre": SimCLR,
    "byol-sc": BYOLLoss,
    "vicreg-sc": VICRegLoss,
    "bt-sc": BarlowTwinLoss,
    "lema": SimCLR
}

proj_dict = {
    "simsiam": simsiam_proj,
    "simclr": BYOL_mlp,
    "byol": BYOL_mlp,
    "bt": bt_proj,
    "vicreg": bt_proj,
    "simsiam-sc": simsiam_proj,
    "scalre": BYOL_mlp,
    "byol-sc": BYOL_mlp,
    "bt-sc": bt_proj,
    "vicreg-sc": bt_proj,
    "lema": BYOL_mlp
}

pred_dict = {
    "simsiam": simsiam_pred, 
    "byol": BYOL_mlp,
    "simsiam-sc": simsiam_pred, 
    "byol-sc": BYOL_mlp
}

pretrain_algo = {
    "simsiam": train_simsiam,
    "simclr": train_simclr,
    "byol": train_byol,
    "vicreg": train_vicreg,
    "bt": train_bt,
    "simsiam-sc": train_simsiam_sc,
    "scalre": train_scalre,
    "byol-sc": train_byol_sc,
    "vicreg-sc": train_vicreg_sc,
    "bt-sc": train_bt_sc,
    "lema": train_lema
}