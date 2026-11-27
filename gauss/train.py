"""PPO on CPUs for walking, in the central form (one MLP policy for all 12 joints) or the neighbours-only split form
(one controller per leg, the same weights for the four legs, with messages between neighbouring legs; gauss.env).

    python -m gauss.train --out runs/a2-s0 --seed 0 --minutes 75
    python -m gauss.train --out runs/a2-split-s0 --form split --hidden 64,64 --seed 0 --minutes 75
    python -m gauss.train --out runs/a2-s0 --resume          # continue from runs/a2-s0/last.pt (same options)

Both forms use the same algorithm and the same critic: a value network on the whole-body observation, used only in
training. In the split form the four legs' Gaussian log probabilities are summed into the robot's, so PPO's ratio,
entropy and KL are the robot's, as in the central form (as in erg). --hidden sets the actor's hidden layers (the
whole policy in the central form, one leg's controller in the split form); the critic keeps PPO["critic_hidden"].

Follows rsl_rl's PPO (clipped objective, clipped value loss, GAE, adaptive learning rate from the KL,
bootstrapping on time-outs). Writes <out>/config.json (settings, versions, the machine, the commit) and
one JSON line per update to <out>/log.jsonl (env steps, wall time, CPU time, losses, reward terms).

Identical results need the same seed, the same --threads, the same torch version and the same kind of
CPU (float reductions in the update depend on the thread count). The number of workers does not matter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from . import machine
from .env import CONFIG, LEG_ACT, LEG_JOINTS, LEG_OBS, MSG_SIZE, N_LEGS, REWARD_WEIGHTS
from .vecenv import VecEnv

PPO = {
    "num_envs": 512,
    "num_workers": 4,
    "steps_per_env": 24,
    "hidden": [256, 128, 64],  # the central actor's hidden layers (default of --hidden for --form central)
    "split_hidden": [64, 64],  # one leg's controller (default of --hidden for --form split)
    "critic_hidden": [256, 128, 64],  # the same critic for both forms
    "activation": "elu",
    "init_noise_std": 1.0,
    # Split form, message values (#28): they are clipped to [-1, 1] before being sent, so larger noise costs the
    # policy nothing and an entropy bonus on them pushes their std up without limit (it reached about 31 in the
    # first run). Their std starts small (as erg's stds do) and they are left out of the entropy bonus.
    "msg_init_noise_std": 0.1,
    "msg_in_entropy": False,
    "gamma": 0.99,
    "lam": 0.95,
    "epochs": 5,
    "minibatches": 4,
    "clip": 0.2,
    "value_coef": 1.0,
    "entropy_coef": 0.01,
    "lr": 1e-3,
    "schedule": "adaptive",  # or "fixed"
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
    "checkpoint_every": 50,
}


def mlp(sizes, out):
    layers, n = [], sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(n, h), nn.ELU()]
        n = h
    layers.append(nn.Linear(n, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Central form: the actor and the critic both see the whole-body observation."""

    def __init__(self, obs_dim, act_dim, hidden, init_std, critic_hidden=None):
        super().__init__()
        self.actor = mlp([obs_dim] + hidden, act_dim)
        self.critic = mlp([obs_dim] + (hidden if critic_hidden is None else critic_hidden), 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))

    def mean(self, obs):
        return self.actor(obs)

    def dist(self, obs):
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class SplitActorCritic(nn.Module):
    """Split form: one actor network shared by the four legs, applied to each leg's own observation (which holds
    its leg identifier); the critic sees the whole-body observation at the start of the env's observation. The
    robot's action is the four legs' actions in a row (N_LEGS * LEG_ACT), with one learned standard deviation per
    leg action value shared by the legs, so the robot's log probability is the sum over legs."""

    def __init__(self, critic_dim, hidden, init_std, critic_hidden, msg_init_std=None):
        super().__init__()
        self.critic_dim = critic_dim
        self.actor = mlp([LEG_OBS] + hidden, LEG_ACT)
        self.critic = mlp([critic_dim] + critic_hidden, 1)
        log_std = torch.full((LEG_ACT,), float(np.log(init_std)))
        if msg_init_std is not None:
            log_std[LEG_JOINTS:] = float(np.log(msg_init_std))
        self.log_std = nn.Parameter(log_std)

    def mean(self, obs):
        legs = obs[:, self.critic_dim:].reshape(obs.shape[0] * N_LEGS, LEG_OBS)
        return self.actor(legs).reshape(obs.shape[0], N_LEGS * LEG_ACT)

    def dist(self, obs):
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp().repeat(N_LEGS).expand(obs.shape[0], -1))

    def value(self, obs):
        return self.critic(obs[:, :self.critic_dim]).squeeze(-1)


def make_model(form, obs_dim, act_dim, critic_dim, hidden, critic_hidden, init_std, msg_init_std=None):
    if form == "split":
        return SplitActorCritic(critic_dim, hidden, init_std, critic_hidden, msg_init_std)
    return ActorCritic(obs_dim, act_dim, hidden, init_std, critic_hidden)


def mlp_size(sizes, out) -> dict:
    """Parameters and multiply-adds per evaluation of an MLP (the activations are not counted)."""
    dims = list(sizes) + [out]
    macs = sum(a * b for a, b in zip(dims[:-1], dims[1:]))
    return {"layers": dims, "params": macs + sum(dims[1:]), "macs": macs}


def computation(form, obs_dim, act_dim, critic_dim, hidden, critic_hidden) -> dict:
    """Size of the policy per controller and for the whole robot, as run on the robot (the mean action; the
    critic and the standard deviations are used only in training, reported separately)."""
    if form == "split":
        c = mlp_size([LEG_OBS] + hidden, LEG_ACT)
        n = N_LEGS
        # Every control step each leg sends its message (foot contact and learned values) to its two neighbours.
        msgs = {"message_values_sent_per_controller": 2 * MSG_SIZE,
                "message_values_sent_robot": N_LEGS * 2 * MSG_SIZE}
    else:
        c = mlp_size([obs_dim] + hidden, act_dim)
        n = 1
        msgs = {"message_values_sent_per_controller": 0, "message_values_sent_robot": 0}
    # For the robot, every controller holds its own copy of the (shared) weights.
    return {"controllers": n, "per_controller": c,
            "robot": {"params": n * c["params"], "macs": n * c["macs"], "distinct_params": c["params"]},
            **msgs, "critic_training_only": mlp_size([critic_dim] + critic_hidden, 1)}


def param_digest(model) -> str:
    h = hashlib.sha256()
    for p in model.state_dict().values():
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="a2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=None, help="stop after this much wall time")
    ap.add_argument("--updates", type=int, default=None, help="stop after this many updates (total)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num-envs", type=int, default=PPO["num_envs"])
    ap.add_argument("--workers", type=int, default=PPO["num_workers"])
    ap.add_argument("--steps-per-env", type=int, default=PPO["steps_per_env"])
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="torch threads for the update")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--form", choices=("central", "split"), default="central")
    ap.add_argument("--hidden", default=None,
                    help="widths of the actor's hidden layers, e.g. 64,64 (default: 256,128,64 central, 64,64 split)")
    ap.add_argument("--foot-contact", action="store_true",
                    help="central form: add foot contact to the observation (the split form always has it)")
    ap.add_argument("--msg-loss", type=float, default=0.0, help="split form: probability that a message is lost")
    ap.add_argument("--msg-delay", type=int, default=0, help="split form: extra message delay in control steps")
    ap.add_argument("--msg-jitter", type=int, default=0,
                    help="split form: a uniform integer in 0..this many control steps added to the delay")
    args = ap.parse_args(argv)
    if args.hidden is None:
        hidden = list(PPO["hidden"] if args.form == "central" else PPO["split_hidden"])
    else:
        hidden = [int(x) for x in args.hidden.split(",") if x]
    hp = dict(PPO, num_envs=args.num_envs, num_workers=args.workers, steps_per_env=args.steps_per_env,
              form=args.form, hidden=hidden)
    del hp["split_hidden"]
    env_cfg = dict(CONFIG, form=args.form, obs_foot_contact=args.foot_contact, msg_loss=args.msg_loss,
                   msg_delay=args.msg_delay, msg_jitter=args.msg_jitter)
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    gen = torch.Generator().manual_seed(args.seed)
    venv = VecEnv(args.robot, hp["num_envs"], hp["num_workers"], args.seed, env_cfg)
    N, T = hp["num_envs"], hp["steps_per_env"]
    obs_dim, act_dim, critic_dim = venv.dims()
    ac = make_model(args.form, obs_dim, act_dim, critic_dim, hidden, hp["critic_hidden"], hp["init_noise_std"],
                    hp["msg_init_noise_std"])
    opt = torch.optim.Adam(ac.parameters(), lr=hp["lr"])
    lr = hp["lr"]
    update, env_steps, wall_prev, cpu_prev = 0, 0, 0.0, 0.0
    ckpt = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ckpt):
        s = torch.load(ckpt, weights_only=False)
        if s["robot"] != args.robot or s["seed"] != args.seed:
            raise SystemExit(f"{ckpt} is for robot {s['robot']}, seed {s['seed']}")
        same = ("form", "hidden", "critic_hidden", "num_envs", "steps_per_env")  # the number of workers may change
        if args.form == "split":
            same += ("msg_init_noise_std", "msg_in_entropy")
        if any(s["hp"].get(k) != hp[k] for k in same) or s["env_config"] != env_cfg:
            raise SystemExit(f"{ckpt} was made with other settings; give the same options as the first run")
        ac.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        gen.set_state(s["gen"])
        lr, update, env_steps = s["lr"], s["update"], s["env_steps"]
        wall_prev, cpu_prev = s["wall_s"], s["cpu_s"]
        venv.set_state(s["envs"])
        if not args.quiet:
            print(f"resumed at update {update}, {env_steps} env steps", flush=True)
    else:
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump({"robot": args.robot, "seed": args.seed, "form": args.form, "env": env_cfg,
                       "reward_weights": REWARD_WEIGHTS, "ppo": hp, "obs_dim": obs_dim, "act_dim": act_dim,
                       "critic_obs_dim": critic_dim,
                       "computation": computation(args.form, obs_dim, act_dim, critic_dim, hidden,
                                                  hp["critic_hidden"]),
                       "torch": torch.__version__, "threads": args.threads, "commit": git_commit(),
                       "machine": machine.describe()}, f, indent=1)
    obs = torch.from_numpy(venv.obs())

    t0, c0 = time.perf_counter(), time.process_time()
    _, cpu_workers0 = venv.stats()
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    buf_obs = torch.zeros(T, N, obs_dim)
    buf_act = torch.zeros(T, N, act_dim)
    buf_logp = torch.zeros(T, N)
    buf_val = torch.zeros(T, N)
    buf_rew = torch.zeros(T, N)
    buf_done = torch.zeros(T, N)
    buf_mu = torch.zeros(T, N, act_dim)
    buf_sigma = torch.zeros(T, N, act_dim)

    def save(path):
        torch.save({"robot": args.robot, "seed": args.seed, "model": ac.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(), "lr": lr,
                    "update": update, "env_steps": env_steps, "envs": venv.get_state(),
                    "wall_s": wall_prev + time.perf_counter() - t0,
                    "cpu_s": cpu_prev + (time.process_time() - c0) + (venv.stats()[1] - cpu_workers0),
                    "hp": hp, "env_config": env_cfg, "reward_weights": REWARD_WEIGHTS}, path + ".tmp")
        os.replace(path + ".tmp", path)

    while True:
        if args.updates is not None and update >= args.updates:
            break
        if args.minutes is not None and time.perf_counter() - t0 > args.minutes * 60:
            break
        t_roll = time.perf_counter()
        torch.set_num_threads(1)  # workers own the cores while stepping
        ep = {"ret": [], "len": [], "fell": 0, "ended": 0}
        lin_err = yaw_err = 0.0
        fwd_v, fwd_n = 0.0, 0
        with torch.no_grad():
            for t in range(T):
                d = ac.dist(obs)
                a = d.mean + d.stddev * torch.randn(d.mean.shape, generator=gen)
                buf_obs[t], buf_act[t] = obs, a
                buf_logp[t] = d.log_prob(a).sum(-1)
                buf_val[t] = ac.value(obs)
                buf_mu[t], buf_sigma[t] = d.mean, d.stddev
                o, r, done, info = venv.step(a.numpy().astype(np.float64))
                r = torch.from_numpy(r)
                timeout = torch.from_numpy(info["time_out"].astype(np.float32))
                buf_rew[t] = r + hp["gamma"] * buf_val[t] * timeout  # bootstrap on time-out (rsl_rl)
                buf_done[t] = torch.from_numpy(done.astype(np.float32))
                obs = torch.from_numpy(o)
                lin_err += info["lin_err"].mean() / T
                yaw_err += info["yaw_err"].mean() / T
                fwd = info["cmd_vx"] >= 0.5
                fwd_v += info["vx"][fwd].sum()
                fwd_n += int(fwd.sum())
                if done.any():
                    ep["ret"] += info["ep_return"][done].tolist()
                    ep["len"] += info["ep_len"][done].tolist()
                    ep["fell"] += int(info["fell"][done].sum())
                    ep["ended"] += int(done.sum())
            last_val = ac.value(obs)
        t_upd = time.perf_counter()
        torch.set_num_threads(args.threads)

        # GAE
        adv = torch.zeros(T, N)
        gae = torch.zeros(N)
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + hp["gamma"] * nv * nonterm - buf_val[t]
            gae = delta + hp["gamma"] * hp["lam"] * nonterm * gae
            adv[t] = gae
        ret = adv + buf_val
        B = T * N
        f_obs, f_act, f_logp, f_val = buf_obs.view(B, -1), buf_act.view(B, -1), buf_logp.view(B), buf_val.view(B)
        f_ret, f_adv = ret.view(B), adv.view(B)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        f_mu, f_sigma = buf_mu.view(B, -1), buf_sigma.view(B, -1)
        mb = B // hp["minibatches"]
        stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "kl": 0.0}
        n_mb = 0
        for _ in range(hp["epochs"]):
            perm = torch.randperm(B, generator=gen)
            for i in range(hp["minibatches"]):
                idx = perm[i * mb:(i + 1) * mb]
                d = ac.dist(f_obs[idx])
                logp = d.log_prob(f_act[idx]).sum(-1)
                ent_all = d.entropy()
                if args.form == "split" and not hp["msg_in_entropy"]:  # joint targets only (#28)
                    ent_all = ent_all.reshape(-1, N_LEGS, LEG_ACT)[..., :LEG_JOINTS]
                ent = ent_all.reshape(ent_all.shape[0], -1).sum(-1).mean()
                v = ac.value(f_obs[idx])
                with torch.no_grad():  # KL(old || new) for diagonal Gaussians, as rsl_rl
                    s_old, s_new = f_sigma[idx], d.stddev
                    kl = (torch.log(s_new / s_old) + (s_old ** 2 + (f_mu[idx] - d.mean) ** 2) / (2 * s_new ** 2)
                          - 0.5).sum(-1).mean()
                    if hp["schedule"] == "adaptive":
                        if kl > 2 * hp["desired_kl"]:
                            lr = max(1e-5, lr / 1.5)
                        elif 0 < kl < hp["desired_kl"] / 2:
                            lr = min(1e-2, lr * 1.5)
                        for g in opt.param_groups:
                            g["lr"] = lr
                ratio = torch.exp(logp - f_logp[idx])
                a_ = f_adv[idx]
                l_pi = -torch.min(ratio * a_, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * a_).mean()
                v_clip = f_val[idx] + (v - f_val[idx]).clamp(-hp["clip"], hp["clip"])
                l_v = torch.max((v - f_ret[idx]) ** 2, (v_clip - f_ret[idx]) ** 2).mean()
                loss = l_pi + hp["value_coef"] * l_v - hp["entropy_coef"] * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), hp["max_grad_norm"])
                opt.step()
                stats["loss_pi"] += l_pi.item()
                stats["loss_v"] += l_v.item()
                stats["entropy"] += ent.item()
                stats["kl"] += kl.item()
                n_mb += 1
        update += 1
        env_steps += B
        t_end = time.perf_counter()
        rec = {
            "update": update, "env_steps": env_steps,
            "wall_s": round(wall_prev + t_end - t0, 2),
            "rollout_s": round(t_upd - t_roll, 3), "update_s": round(t_end - t_upd, 3),
            "mean_step_reward": round(float(buf_rew.mean()), 5),
            "ep_return": round(float(np.mean(ep["ret"])), 3) if ep["ret"] else None,
            "ep_len": round(float(np.mean(ep["len"])), 1) if ep["len"] else None,
            "episodes_ended": ep["ended"],
            "fall_rate": round(ep["fell"] / ep["ended"], 3) if ep["ended"] else None,
            "falls_per_1k_steps": round(1000 * ep["fell"] / B, 3),
            "lin_vel_err": round(float(lin_err), 4), "yaw_rate_err": round(float(yaw_err), 4),
            "vx_when_cmd_ge_0.5": round(fwd_v / fwd_n, 3) if fwd_n else None,
            "lr": lr, "std": round(float(ac.log_std.detach().exp().mean()), 4),
            **{k: round(v / n_mb, 6) for k, v in stats.items()},
        }
        if args.form == "split":  # the joint targets' and the learned message values' noise, separately
            leg_std = ac.log_std.detach().exp()
            rec["std_joints"] = round(float(leg_std[:LEG_JOINTS].mean()), 4)
            rec["std_messages"] = round(float(leg_std[LEG_JOINTS:].mean()), 4)
        sums, cpu_w = venv.stats()
        rec["cpu_s"] = round(cpu_prev + (time.process_time() - c0) + (cpu_w - cpu_workers0), 1)
        rec["reward_terms_per_step"] = {k: round(v / B, 6) for k, v in sums.items()}
        rec["param_digest"] = param_digest(ac)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if not args.quiet and (update % 10 == 0 or update == 1):
            print(f"u{update} steps={env_steps} wall={rec['wall_s']:.0f}s cpu={rec.get('cpu_s')} "
                  f"R={rec['ep_return']} L={rec['ep_len']} lin={rec['lin_vel_err']} yaw={rec['yaw_rate_err']} "
                  f"fall={rec['fall_rate']} vx={rec['vx_when_cmd_ge_0.5']} lr={lr:.2e} std={rec['std']} "
                  f"roll={rec['rollout_s']} upd={rec['update_s']}", flush=True)
        if update % hp["checkpoint_every"] == 0:
            save(ckpt)
            save(os.path.join(args.out, f"model_{update:05d}.pt"))
    save(ckpt)
    venv.close()
    log.close()
    if not args.quiet:
        print("done", update, env_steps, param_digest(ac), flush=True)
    return param_digest(ac)


if __name__ == "__main__":
    main()
