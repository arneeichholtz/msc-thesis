

import numpy as np
import os

# from features_config.features import phoneme_mapping

# import yaml
from pathlib import Path
from typing import Any, Dict

if __name__ == "__main__":

    test_path = Path("/projects/0/prjs1921/thesis-speech/output_test")
    os.makedirs(test_path, exist_ok=True)

    test = np.random.rand(500, 512, 600)
    np.save(test_path / "test.npy", test)




        




    

