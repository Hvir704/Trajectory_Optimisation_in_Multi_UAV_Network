"""make_drl_configs.py -- one DRL-UCS training config per (deployment, K).

usage:
  python make_drl_configs.py --out configs/run --layout paper --M 50 --Emax 1.5e6 --seeds 1 2 3 --K 2 3 4 5 6 7 \
         --nb-step 1e7 --workers 4 --envs 8 --logdir ~/drl_logs
Hyperparameters are the authors' released Beijing settings (adept/config/bj.json: lr 7e-4, discount 0.99, entropy
0.01, rollout 20, GTrXL 2 layers, intrinsic reward on with coefficient 0.3, shared parameters); only the
environment block, the agent count, the step budget and the machine layout differ.
"""
import argparse, json, os, copy

BASE = {
  "actor_host": "ImpalaHostActor", "actor_worker": "ImpalaWorkerActor", "ceil": 1, "custom_network": "RealNetwork",
  "discount": 0.99, "entropy_weight": 0.01, "env": "EnvDyn-v0", "epoch_len": 1000000, "eval": False, "exp": "Rollout",
  "floor": -1, "fourconv_norm": "bn", "frame_stack": False, "grad_norm_clip": "0.5", "head1d": "Identity1D",
  "head2d": "Identity2D", "head3d": "Identity3D", "head4d": "Identity4D", "learner": "ImpalaLearner",
  "load_network": None, "load_optim": None, "lr": 0.0007, "lstm_nb_hidden": 256, "lstm_normalize": True,
  "manager": "SubProcEnvManager", "max_episode_length": 100000, "minimum_importance_policy": 1.0,
  "minimum_importance_value": 1.0, "nb_learners": 1, "net1d": "Identity1D", "net2d": "Identity2D",
  "net3d": "FourConv", "net4d": "Identity4D", "netbody": "Linear", "nb_layer": 2, "linear_nb_hidden": 256,
  "noop_max": 30, "profile": False, "prompt": False, "ray_addr": None, "resume": None, "rollout_len": 20,
  "rollout_queue_size": 20, "rwd_norm": "Clip", "skip_rate": 4, "summary_freq": 10,
  "learner_cpu_alloc": 2, "learner_gpu_alloc": 0.5, "worker_cpu_alloc": 2, "worker_gpu_alloc": 0.1,
  "cuda_visible_device": "", "independent_reward": True, "shared_params": True, "concat_obs": True, "bg": 2,
  "n_layers": 2, "use_transformer": True, "use_intrinsic": True, "intrinsic_coef": 0.3, "dataset": "dyn",
  "test_mode": False, "description": "envdyn",
  # --- EnvDyn (registered choices; see DRL_REGISTRATION.md) ---
  "dyn_slot": 5.0, "dyn_collect_radius": 200.0, "dyn_obs_radius": -1.0, "dyn_horizon_h": 12.0,
  "dyn_burnin_h": 3.0, "dyn_reward_ref": 3600.0, "dyn_eps": 0.0, "dyn_aoi_th": 3600.0, "dyn_mode": "train",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", required=True); ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--Emax", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--K", type=int, nargs="+", required=True)
    ap.add_argument("--nb-step", type=float, default=1e7)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--logdir", default="~/drl_logs")
    ap.add_argument("--set", nargs="*", default=[], help="extra key=value overrides (JSON values)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.seeds:
        for K in a.K:
            c = copy.deepcopy(BASE)
            c.update(nb_agent=K, nb_workers=a.workers, nb_learn_batch=a.workers, nb_env=a.envs, nb_step=a.nb_step,
                     logdir=os.path.expanduser(a.logdir), dyn_layout=a.layout, dyn_M=a.M, dyn_Emax=a.Emax, dyn_seed=s,
                     tag=f"{a.layout}_M{a.M}_E{a.Emax:.1e}_s{s}_K{K}")
            for kv in a.set:
                k, v = kv.split("=", 1); c[k] = json.loads(v)
            fn = os.path.join(a.out, c["tag"] + ".json")
            json.dump(c, open(fn, "w"), indent=2); print(fn)
