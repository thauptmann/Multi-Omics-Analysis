import subprocess
from io import BytesIO
import pandas as pd


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.free'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    return gpu_df['memory.free'].idxmax()
