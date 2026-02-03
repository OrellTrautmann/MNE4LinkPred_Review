import pandas as pd
import numpy as np
import torch
import networkx

from models.mell import mell
from models.liamne import liamne
from models.rmne import rmne

if __name__ == "__main__":

    data = pd.read_csv("Vickers-Chan-7thGraders_multiplex.txt", sep=" ", header=None)

    model = rmne({"weighted": False, "directed": True, "emb_size": 16, "node_num": 29, "layer_num": 3, "dataset": "Vickers", "run": 1, "target_layer":1})

    model.fit(data)

    print(model.model_return())

    print("fin")