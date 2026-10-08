import os
import sys

ROOT = "/root/autodl-tmp/MMRec"
SRC = os.path.join(ROOT, "src")

os.chdir(SRC)
sys.path.insert(0, SRC)

from utils.quick_start import quick_start

if __name__ == "__main__":
    quick_start(
        model="MGCN",
        dataset="sports",
        config_dict={
            "cl_loss": 0.0,
            "hyper_parameters": []
        },
        save_model=True
    )
