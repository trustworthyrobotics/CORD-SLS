import argparse
import os
import time
import json
import pickle
import rich
import jax
import jax.numpy as jnp
from pathlib import Path
from datetime import datetime

from methods import *
from tasks import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="Method to benchmark")
    parser.add_argument("--task", type=str, required=True, help="Task to benchmark on")

    parser.add_argument("--evaluations", type=int, default=1, help="Number of evaluations to run")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed for the evaluations run")

    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path, perform training if empty")

    parser.add_argument("--visualize", type=str, default=True, help="Visualize simulation")
    parser.add_argument("--save_traj", type=bool, default=False, help="Save the state control trajectory")

    parser.add_argument("--jax_preallocate", type=bool, default=False, help="JAX preallocate GPU memory")
    parser.add_argument("--jax_x64", type=bool, default=True, help="JAX enable float64")

    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(args.jax_preallocate).lower()
    jax.config.update('jax_enable_x64', args.jax_x64)

    method_str = args.method.upper()
    if method_str == "SLS":
        run_method = run_SLS
    elif method_str == "MPPI":
        run_method = run_MPPI
    elif method_str == "APG":
        run_method = (run_APG, train_APG)
    elif method_str == "PPO":
        run_method = (run_PPO, train_PPO)
    else:
        raise RuntimeError(f"Unknown method '{args.method}'")

    task_str = args.task.lower().replace('_', '').replace('-', '')
    if task_str == "liftrope":
        task_class = LiftRopeTask
    elif task_str == "dragrope":
        task_class = DragRopeTask
    elif task_str == "foldcloth":
        task_class = FoldClothTask
    elif task_str == "flattencloth":
        task_class = FlattenClothTask
    else:
        raise RuntimeError(f"Unknown task '{args.task}'")

    server = None
    if args.visualize:
        import viser
        server = viser.ViserServer()

    timestamp = datetime.now().strftime("%m%d-%H%M")
    metrics = []
    for k in range(args.seed, args.seed + args.evaluations):
        task = task_class(seed=k)
        if method_str == "APG" or method_str == "PPO":
            if args.checkpoint:
                execution_time = run_method[0](task, server, args.checkpoint)
            else:
                traj = load_demo(task_str)
                execution_time = run_method[1](task, server, traj)
        else:
            execution_time = run_method(task, server)

        metric, traj = task.eval_metrics()
        metric["seed"] = k
        metric["execution_time"] = execution_time

        metrics.append(metric)
        rich.print(metric)

        summary = {
            "method": method_str,
            "task": task_str,
            "metrics": metrics,
        }
        path = Path(__file__).resolve().parent / "results" / f"{method_str}_{task_str}_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        if not args.save_traj:
            continue
        path = Path(__file__).resolve().parent / "results" / f"{method_str}_{task_str}_{timestamp}_traj{k}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(traj, f)


def load_demo(task_str, k=0):
    path = Path(__file__).resolve().parent / "demos" / f"{task_str}_traj{k}.pkl"
    with open(path, "rb") as f:
        traj = pickle.load(f)
    return traj


if __name__ == "__main__":
    main()
