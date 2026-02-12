

import numpy as np
import math

from features_config.features import phoneme_mapping

import yaml
from pathlib import Path
from typing import Any, Dict
    
CONFIG_PATH = Path("config.yml")

def load_training_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)

if __name__ == "__main__":

    # phonemes = ["sil", "d", "ow", "n", "ae", "s", "sil"]
    # starts = [0, 2150, 2600, 3800, 4344, 7080, 8080]
    # stops = [2150, 2600, 3800, 4344, 7080, 8080, 8920]

    # for phoneme, start, stop in zip(phonemes, starts, stops):

    #     start_index = math.floor((start / 16000) / 0.020)
    #     stop_index = math.ceil((stop / 16000) / 0.020)

    #     print(np.arange(start_index, stop_index))

    #     print(f"Phoneme: {phoneme}, Start: {start}, End: {stop}, Start Index: {start_index}, Stop Index: {stop_index}")


    
    config = load_training_config()

    test = config['use_freezing_callback']
    if test:
        print("Freezing callback will be used.")


        




    

