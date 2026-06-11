import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent

gpu_sls_dir = ROOT / "gpu_sls" / "src"
sys.path.insert(0, str(gpu_sls_dir))

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig

gpu_sls = types.SimpleNamespace()
gpu_sls.ADMMConfig = ADMMConfig
gpu_sls.SLSConfig = SLSConfig
gpu_sls.SQPConfig = SQPConfig
gpu_sls.GenericMPC = GenericMPC
gpu_sls.MPCConfig = MPCConfig
