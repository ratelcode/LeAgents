# LeAgents — Phase 2: Residual RL for the Self-Improvement Flywheel

**Design document · July 2026 · design + implementation-plan only (no code shipped by this doc)**

This document designs **step 2 of the DexFlyWheel cycle** ([arXiv:2509.23829](https://arxiv.org/abs/2509.23829)) for LeAgents: a **residual RL** stage whose exploration reaches success on initial states and task variations the frozen base policy fails, so that rolling out the *composed* (base + residual) policy harvests success on genuinely **new** situations — the actual source of new coverage the flywheel needs.

It follows the parent `DESIGN.md` conventions. Claims marked **[verified]** were read from a primary source (arXiv full text, installed lerobot 0.5.1 source, or LIBERO source) during the July 2026 research pass; **[recommendation — unverified]** are design choices to validate during implementation.

> **Why Phase 2 exists (the Phase-1 result that motivates it).** A Phase-1 autonomous run climbed LIBERO-Spatial 63% → 69% → 72% purely by **data growth** (200 → 320 → 432 episodes). At 432 the expert-data well is **dry** — that is every `libero_spatial` episode in `HuggingFaceVLA/libero` (~43/task). Phase-1 also made the DexFlyWheel harvest work: it collects **success-filtered** rollouts and merges them into a growing, seed-compatible "self-play" mix. But success-filtering keeps only episodes the policy **already** solves, on the **same 10 tasks** — so the harvest adds ~no new coverage over the expert data, and blending it yields only ±1–2%. Tasks the base systematically fails (e.g. `task_id 5`, "black bowl on the ramekin", ~2–5/10) are exactly the ones success-filtering **excludes**. Residual RL is the mechanism that changes what the harvest *can contain*: it produces success on situations the base fails, so the harvest finally carries new information.

---

## 0. TL;DR — the recommended design in one screen

| Axis | Decision | Grounding |
|---|---|---|
| **Residual policy** | Small SAC actor: 3-layer MLP (256 hidden), input `(proprio_state 8-d, base_action 7-d, [optional] DrQ image features)`, output a **per-step** 7-d delta `Δa`, bounded `α·tanh(·)`. | PLD [2511.00091], ResFiT [2509.19301], A2C2 [2509.23224] |
| **Composition** | Per-step over the cached base chunk: `a_t = a_base_{t+i} + α·tanh(f_θ(o_t, a_base_{t+i}))`, with a **progressive ε-ramp** (base-only → composed) and **α≈0.5** (PLD's LIBERO value). Base weights **frozen**; residual never becomes a promoted checkpoint. | Policy Decorator [2412.13630], PLD, DexFlyWheel |
| **Algorithm** | **Off-policy SAC + RLPD**: 50/50 online/demo symmetric sampling, LayerNorm critic, critic ensemble min-over-subset, moderate UTD. **Reuse lerobot 0.5.1 `SACPolicy` + `ReplayBuffer`** in a **single process**. | RLPD [2302.02948], HIL-SERL [2410.21845], lerobot source |
| **Reward** | LIBERO's **native per-step sparse binary** success (1.0 iff `_check_success()`), terminate-on-success. **No reward classifier in sim.** Demo-anchored RLPD makes sparse tractable; light dense shaping is a documented fallback. | LIBERO source, PLD, SimpleVLA-RL [2509.09674] |
| **CVE-2026-25874 safety** | **Never import** `lerobot.rl.actor` / `learner` / `learner_service` / `lerobot.transport.*` (the gRPC/pickle surface). Single-process loop reuses only the transport-free SAC math. Constitution gains an explicit deny rule. | lerobot source (import audit), `DESIGN.md` §6 |
| **Loop slot** | A **tool the IMPROVE stage invokes** after a promotion, *before* the harvest. Residual is **ephemeral** (a data generator, discarded after harvest). Promote/rollback stays a pure function of eval deltas on the standard gate. | `orchestrator/loop.py`, `orchestrator/decision.py` |
| **Sample budget** | ~**100k–250k** env steps per targeted task (off-policy), **not** the millions a PPO residual needs. Fits 16 GB (~4–6 GB used); **wall-clock**, not VRAM, is the constraint. | ResFiT (200k vs 40M), PLD (250k), SmolVLA latency |

**The one-sentence honest caveat:** this exact combination — residual **RL** on a **frozen 450M VLA** — is **novel** (no verified publication does it on SmolVLA), but it is **tightly bracketed** by verified precedent on every individual axis (PLD does it on π0/OpenVLA; A2C2 does the SmolVLA/chunk-50/per-step-residual plumbing but supervised; ResFiT does frozen-chunked-base + off-policy + sparse reward). The dominant risk is **wall-clock on one GPU**, and the honest limit is that **a residual cannot create a behavior the base never attempts** (§6, §Risks).

---

## 1. Architecture — the residual policy and how it composes with frozen SmolVLA

### 1.1 What SmolVLA gives us (verified facts that constrain the design)

From the installed `lerobot 0.5.1` source (`lerobot/policies/smolvla/`) and the `HuggingFaceVLA/smolvla_libero` checkpoint config:

- **[verified]** `SmolVLAConfig`: `chunk_size=50`, `n_action_steps=50` by default; **but the LIBERO checkpoint ships `n_action_steps=1`** (re-plans every env step). `n_action_steps` is overridable at load (`--policy.n_action_steps=N`).
- **[verified]** LIBERO action space is `Box(-1,1, shape=(7,))` — 6-DoF end-effector delta (OSC) + 1-DoF gripper. Internally padded to `max_action_dim=32`, unpadded to 7 on output.
- **[verified]** Proprio `observation.state` is **8-d** = `[eef_pos(3), eef axis-angle(3), gripper_qpos(2)]` (LIBERO's nested robot_state flattened by `LiberoProcessorStep`). Two cameras `observation.images.image` (agentview) + `observation.images.image2` (wrist), rendered **256×256** (the config's 360 is not forwarded — a known lerobot quirk).
- **[verified]** Action expert = **flow matching**, 10-step Euler integration at inference; the image/language/state prefix is encoded **once** (KV-cache) then 10 cheap expert-only steps.
- **[verified]** Clean programmatic hook: `SmolVLAPolicy.predict_action_chunk(batch) -> Tensor (B, 50, action_dim)` (`@torch.no_grad`, accepts a `noise` kwarg); `select_action(batch)` pops one action from an internal `deque(maxlen=n_action_steps)` and refills by calling the model only when empty; `policy.reset()` clears the queue (must be called on env reset).
- **[verified]** SmolVLA ~450M params (~100M in the action expert); weights `model.safetensors` = **907 MB** (bf16 VLM). Paper gives **no** official inference-VRAM number (the "~0.9 GB" figure circulating is the *checkpoint size*, not measured VRAM).

### 1.2 The residual policy

**Recommendation.** A small **SAC actor** — a 3-layer MLP with 256 hidden units — matching PLD's residual spec (**[verified]** PLD uses "a 3-layer MLP Gaussian policy, 256 hidden"). We reuse lerobot's `Policy`/`MLP`/`CriticEnsemble` classes from `lerobot/policies/sac/modeling_sac.py` rather than reimplement them.

**Inputs (actor):** `(s_t, a_base)` where
- `s_t` = the 8-d proprioceptive state (**[verified]** cheap, always available), and
- `a_base` = the base policy's action for the current step (the t-th action of the cached chunk).

Conditioning the residual on the **base action** is the ResFiT/PLD choice and is **more important for a chunked base**: mid-chunk, `a_base_{t+i}` carries information about the base's (possibly stale) open-loop plan that the current proprio observation alone does not. **[verified]** ResFiT: residual observation is `[s_{t+i}, a_base_{t+i}]`; **[verified]** PLD conditions the residual on `(s, a_b)`. (Contrast: **[verified]** Silver 2018 and Policy Decorator's *actor* use state only — but Policy Decorator's *critic* still evaluates the summed action, and DexFlyWheel's state-only residual had privileged object state that we deliberately avoid.)

**Vision (progressive, configurable).** A pure proprio+base-action state is **not strictly Markov** for a pick task (it does not localize the bowl). Two mitigations, in order of preference:
1. **Rely on `a_base` as the visual channel (default v1).** `a_base` is SmolVLA's vision-and-language-informed output, so a large amount of scene information reaches the residual implicitly. Start here — it is the cheapest and matches the fact that the residual is a *local corrector*, not a from-scratch policy.
2. **Add a small image encoder to the critic (and optionally actor) if v1 underperforms.** **[verified]** lerobot's `SACObservationEncoder` + `DefaultImageEncoder` (scratch CNN) or `PretrainedImageEncoder` (frozen `helper2424/resnet10`, ~5M params) + DrQ random-shift augmentation is already built and composable. This is the recommended escalation for spatial-config-dependent failures (LIBERO-Spatial's entire point is that only the spatial layout varies — **[verified]** "place a bowl … there are two identical bowls that differ only in their location").

**[recommendation — unverified]** *A future optimization — reusing SmolVLA's own encoded prefix features as the residual's visual input — is deferred: the prefix is a token sequence, not a fixed vector, and wiring it cleanly is more integration than v1 warrants.*

**Output:** a **per-step** 7-d delta `Δa_t`, bounded:

```
a_t = a_base_{t+i} + α · tanh(f_θ(s_t, a_base_{t+i}))
```

- **[verified]** Bounding via `tanh × α` is Policy Decorator's recipe; PLD scales to `[−ξ, ξ]` with a scheduler and **recommends ξ=0.5 for LIBERO** (ξ=0.1 for SimplerEnv). **Recommendation:** start `α=0.5`, treat it as the primary tuning knob.
- **[verified]** Zero/small init so the composed policy **starts equal to the base**: Silver 2018 zero-inits the last layer (`f_θ(s)=0 ∀s`); ResiP uses orthogonal init with small final-layer gain. **Recommendation:** small-gain final layer + the exploration ramp below.

### 1.3 Composition at action time — per-step residual over a cached chunk

This is the crux for an action-chunking base. **[verified]** the literature converges on a **per-step residual inside the chunk** (ResiP, ResFiT, A2C2 all do this; nobody applies one macro-residual to a whole 50-step chunk and reports success).

```
On env reset:            base_policy.reset(); residual MDP step counter = 0
Every K = n_action_steps: chunk = base_policy.predict_action_chunk(obs)   # 450M forward, cached
                          (recommend K small enough for reactivity, large enough for wall-clock)
Every env step t (i = t mod K):
    a_base   = chunk[i]                                   # cached — no base forward
    a_res    = α · tanh(actor(s_t, a_base))               # ~cheap MLP forward
    a_exec   = clip(a_base + a_res, -1, 1)                # executed in LIBERO
    with prob (1 − ε): a_exec = a_base                    # progressive-exploration gate
```

- **Progressive exploration ε-ramp.** **[verified]** Policy Decorator ramps ε linearly 0→1 over H steps; "too-small H → complete failure, large H is a safe choice." **[verified]** DexFlyWheel ramps ε from 0→1 between global steps 1.5k and 10k. **Recommendation:** linear ramp over the first ~10–20k env steps; start conservative (long ramp) because aggressive early residual is a documented cause of total failure.
- **Chunk-caching is what makes this affordable.** **[verified]** A2C2 measures SmolVLA forward ≈ 101 ms vs 4.7 ms for a 32M residual head — the base dominates wall-clock. Calling `predict_action_chunk` **once per K steps** and correcting per-step with the cheap MLP is the difference between a feasible and an infeasible run. There is a **reactivity/compute trade-off** in K: `n_action_steps=1` (the LIBERO checkpoint default) re-plans every step (max reactivity, ~50× the base forwards); a larger K caches longer (cheap, but the residual must correct a staler open-loop plan). **Recommendation:** start with a moderate `K` (e.g. 5–10) and tune; the residual exists precisely to restore closed-loop reactivity to the open-loop chunk (**[verified]** ResiP's framing: chunked BC policies "function more like trajectory planners than reactive controllers").

### 1.4 The critic

**[verified]** The clean parameterization (ResFiT, PLD, Policy Decorator all agree) is a critic over the **full executed action**:

```
Q_φ(s_t, a_exec)   where a_exec = a_base + a_res
```

- **[verified]** LayerNorm on the critic is "crucial … bounds Q-values … without constraining the policy to stay near the data" (RLPD, ResFiT). lerobot's `CriticHead` supports this.
- **[verified]** REDQ/RLPD-style **ensemble with min-over-random-subset** TD targets (lerobot exposes `num_critics`, `num_subsample_critics`).
- **Buffer bookkeeping.** Because the critic and actor loss need `a_base(s)` for **every** sampled transition (including demo transitions), store `a_base` **alongside** each transition: `(s, a_base, a_exec, r, s', done)`. Online transitions already computed `a_base` during rollout; for the offline demo buffer, **precompute `a_base` once** by running the frozen base over the demo states (a one-time offline pass), so the RL loop never pays a 450M forward per sampled batch. This bookkeeping is the one genuinely new piece beyond lerobot's buffer and is called out again in the plan (P2.0).

### 1.5 Why *not* DexFlyWheel's literal residual recipe

**[verified]** DexFlyWheel's residual is a **state-only** SAC MLP `[256,256,256]` taking **privileged object state + proprioception** (no vision, no base action), trained with **dense hand-designed per-task shaped rewards** (their Eq. 2/3: grasp/lift/pour/handover shaping), **1.5M timesteps per task per iteration** on OmniGibson, on top of a **small per-task diffusion policy that is fully retrained every cycle**. Three of those assumptions **do not hold** for LeAgents and are the reason we adopt the PLD/ResFiT recipe instead:

1. **Privileged object state** — unavailable/undesirable; we condition on `a_base` (which carries vision) + proprio, matching PLD/ResFiT.
2. **Dense per-task hand-shaped reward** — writing bespoke shaping for 10 LIBERO tasks is brittle and defeats autonomy; **[verified]** PLD and SimpleVLA-RL show LIBERO's **sparse binary** reward is tractable *given demo-anchored RLPD*, so we start sparse (§2).
3. **A small, cheaply-retrained base + PPO/SAC at 1.5M steps** — our base is a frozen 450M VLA and we have one 16 GB GPU; **[verified]** ResFiT's off-policy residual converges at **~200k steps vs ~40M for PPO (~200×)**, so off-policy is mandatory and the sample budget must be PLD-scale (250k), not DexFlyWheel-scale (1.5M/task).

The **mechanism** (frozen base + α-scaled residual + ε-ramp + success-filtered harvest of the composed policy) is DexFlyWheel-faithful; the **algorithm/reward/scale** follow the frozen-VLA residual-RL line (PLD/ResFiT) that is actually costed for our hardware.

---

## 2. RL algorithm & reward

### 2.1 Algorithm — off-policy SAC + RLPD (reusing lerobot's stack)

**Decision: off-policy SAC with the RLPD recipe.** Justification is the single most decision-relevant number in the research: **[verified]** ResFiT reports its off-policy residual converging at **~200k env steps vs ~40M for the on-policy (PPO) residual — a ~200× sample-efficiency gap.** On-policy residuals (ResiP, and DexFlyWheel's PPO-flavored SAC at 1.5M steps/task) assume Isaac-Gym-scale parallel simulation. On one CPU-bound LIBERO MuJoCo host, only off-policy is viable.

The **RLPD** ingredients, all **[verified]** present in lerobot 0.5.1's `SACPolicy`/`ReplayBuffer`/learner math (`lerobot/policies/sac/`, `lerobot/rl/buffer.py`, `lerobot/rl/learner.py:393-554`):

1. **Symmetric 50/50 sampling** — `batch_size//2` from the online buffer, `batch_size//2` from an offline **demo** buffer (`ReplayBuffer.from_lerobot_dataset` + `concatenate_batch_transitions`). **[verified]** RLPD: "50% from replay, 50% from offline data — no tuning required." This is what anchors sparse-reward learning to the expert distribution.
2. **LayerNorm critic** — **[verified]** RLPD/ResFiT "crucial" for taming overestimation without constraining exploration.
3. **Critic ensemble, min-over-subset targets** — **[verified]** RLPD uses E=10, subset Z∈{1,2}; lerobot exposes `num_critics`/`num_subsample_critics`. **Recommendation:** start smaller (E=2–4) for wall-clock and raise if unstable.
4. **Moderate UTD** — **[verified]** RLPD uses G=20 (state) / 10 (pixels); ResFiT UTD≈4. **Recommendation:** start UTD≈4 (wall-clock-bound here, not sample-bound) and tune up only if learning stalls.
5. **Critic burn-in / warmup** — **[verified]** Silver 2018 documents "good base + untrained critic → performance degradation"; ResFiT runs a warmup executing `a_base + uniform noise` to fill the buffer and warm the critic **before** actor updates. **Recommendation:** replicate — hold the actor fixed (residual≈0) for the first N steps while the critic learns.
6. **DrQ image augmentation** — **[verified]** already in `ReplayBuffer` (`random_shift`, pad=4, `use_drq`) if/when the residual uses images.

**Gripper.** **[verified]** lerobot's SAC has a `DiscreteCritic` (DQN-style) for the gripper (open/close/stay) used by HIL-SERL. **Recommendation:** for v1, keep the residual **continuous over all 7 dims** (simpler; `a_base` already produces a sensible gripper signal and the residual only nudges it); adopt the discrete gripper head only if gripper timing proves to be the failure mode.

### 2.2 Reward — LIBERO's native sparse binary success (no classifier in sim)

**[verified]** LIBERO's env, through lerobot's factory, already provides a proper sparse-reward MDP:
- `LiberoEnv.step()` returns per-step `reward = 1.0 iff _check_success()` (robosuite base, `reward_scale=1.0`), with `_check_success()` evaluating the BDDL goal conjunction **every step**;
- `info["is_success"]` is available **every step** (not just terminal), and `terminated = done or is_success`.

So on **sim we do NOT need a trained reward classifier** — the sparse binary reward and a terminate-on-success signal come free. This is a meaningful simplification over the real-robot HIL-SERL recipe (where the reward classifier exists precisely because real robots have no success predicate).

**Two verified env quirks the RL loop MUST handle:**
- **[verified]** On termination LIBERO **auto-resets itself inside `step()`** (returns the terminal obs, but the sim is already in a fresh episode and the init-state pointer has advanced). → Use `done`-masked bootstrapping and **never use the post-terminal `next_state`**; drive resets explicitly.
- **[verified]** `truncated` is always `False` from the inner env; the 280-step `libero_spatial` limit (`TASK_SUITE_MAX_STEPS`) is enforced by the caller. → Our loop owns the time limit and must distinguish time-limit truncation (bootstrap) from success termination (don't bootstrap).

**Sparse-reward tractability — why this can work despite binary reward:** **[verified]** PLD gets ~99% on LIBERO-90 with **sparse reward and no shaping** on a frozen VLA + residual SAC; **[verified]** SimpleVLA-RL solves LIBERO-Spatial from binary reward. Both rely on **demo-anchored** training — the 50/50 demo buffer keeps the actor near a region where the sparse reward is reachable, and the residual restricts the search to a neighborhood of the (already-decent) base. This is exactly our setting.

**Documented fallbacks if sparse alone stalls (in priority order), each [verified] as a technique:**
1. **Demo-guided / targeted resets** — reset the residual's training episodes to the **failing init states** (§3.2, `set_init_state`) and, optionally, to states drawn from partway through expert demos (bringing the agent near reward). This is the cheapest and most-aligned fix.
2. **Light dense shaping** — a distance-to-object or grasp-height term (à la DexFlyWheel Eq. 2). Kept as a fallback, **off by default**, because per-task shaping is brittle and anti-autonomy.
3. **Reward densification via a process model** (**[verified]** VLA-RL's RPRM) — heavier; out of scope for Phase 2, noted for completeness.

---

## 3. LeRobot integration — CVE-safe, single-process

### 3.1 The safety decision: reuse HIL-SERL's *machinery*, not its *transport*

**Constraint (`DESIGN.md` §6):** CVE-2026-25874 (CVSS 9.3) is an **unauthenticated pickle RCE over lerobot's plaintext gRPC**, fixed only in lerobot ≥ 0.6.0. **[verified from lerobot source]** the same pickle primitive is on the **RL** transport, not just async-inference: `lerobot/transport/utils.py:135` does raw `pickle.load` on interaction messages the **learner** receives via its `add_insecure_port` gRPC server (`learner.py:664`). The constitution already denies `lerobot-async-inference`; the HIL-SERL **actor-learner is the same gRPC family** and must be treated identically.

**Decision: a self-contained, in-process residual-SAC loop that reuses lerobot's SAC *math and modules* but never touches the transport.** **[verified by import audit in the venv]** `SACPolicy`, `SACConfig`, `ReplayBuffer`, the reward `Classifier`, `lerobot.rl.gym_manipulator`, `lerobot.rl.eval_policy`, and `lerobot.envs.factory` import **zero** `grpc*`/`lerobot.transport*` modules (`lerobot/rl/__init__.py` is empty, so importing submodules does not drag in `actor`/`learner`). The gRPC coupling is only at the learner's edges; **[verified]** the entire SAC update step is ~150 lines of pure torch (`learner.py:393-554`) plus optimizer setup (`learner.py:761-810`) — copy those out (they live in the gRPC-importing module but contain no transport code).

**Concrete import allow/deny list for Phase 2 code:**

| Reuse (transport-free) | **Never import (CVE surface)** |
|---|---|
| `lerobot.policies.sac.modeling_sac` (`SACPolicy`, encoders, `MLP`, `CriticEnsemble`, `Policy`) | `lerobot.rl.actor` |
| `lerobot.policies.sac.configuration_sac` (`SACConfig`) | `lerobot.rl.learner` |
| `lerobot.rl.buffer` (`ReplayBuffer`, `concatenate_batch_transitions`, `random_shift`) | `lerobot.rl.learner_service` |
| `lerobot.envs.factory.make_env` (LIBERO) | `lerobot.transport.*` |
| `lerobot.policies.smolvla.modeling_smolvla` (frozen base, `predict_action_chunk`) | (async-inference `PolicyServer`) |

### 3.2 Env driving and targeted init states

**[verified]** `make_env(env_type="libero", ...)` returns `dict[suite][task_id] -> gym.vector.{Sync,Async}VectorEnv`. For targeted residual RL:
- **[verified]** `get_task_init_states(suite, task_id)` loads the suite's `.init` file — **(100, 92)**, i.e. **100 fixed MuJoCo init states per task**; `env.set_init_state(state_vec)` sets any exact state. This is the lever for **targeting the init states the base fails**: split the 100 states into a **train** subset (residual RL trains here) and a **held-out** subset (harvest + eval here) so the residual cannot memorize eval states (§5).
- **[verified quirks to fix before real runs]** (a) `~/.libero/config.yaml` on this machine points at the **stale** `leagent` (no "s") venv — resolve to the current install first; (b) rendered images are **256×256** (config's 360 is not forwarded); (c) LIBERO needs `MUJOCO_GL=egl` (the machine had an EGL blocker — verify headers per project memory); (d) reset burns 10 no-op settle steps (not counted against the limit).

### 3.3 Reward classifier — deferred to M3 (real robot)

**[verified]** lerobot ships a `reward_classifier` policy (`Classifier`, frozen ResNet-10 → BCE head, trainable via `lerobot-train --policy.type=reward_classifier`, deployed via `RewardClassifierProcessorStep`). We **do not need it in sim** (§2.2). It is the natural bridge to M3 (real robot, post-CVE lerobot ≥ 0.6.0), where no success predicate exists — noted here so the sim design does not accidentally depend on it.

---

## 4. Loop integration — where residual RL slots in, staying deterministic

### 4.1 It is a tool the IMPROVE stage invokes — not a control-flow change

The parent design's iron rule (`DESIGN.md` §2, §4; `orchestrator/decision.py`): **promote/iterate/escalate/rollback is a pure function of eval deltas, never an LLM, never a clever stage.** Phase 2 respects this by making residual RL a **tool inside the existing `ImproveAgent`**, invoked on the same trigger the harvest already uses (`loop.py`: on `Decision.PROMOTE` with `success_rate > 0`). No change to `decide()`; no change to the state machine.

**Current harvest path (Phase-1, in `improve_agent.py`/`collect_rollouts.py`):**
```
PROMOTE → collect_rollouts(blessed base) → success-filter → merge into seed-compatible mix
```

**Phase-2 path (residual RL inserted before the harvest):**
```
PROMOTE → [NEW] train_residual(frozen base, target tasks/init-states)   # in-process SAC, §1–3
        → [NEW] compose(base + residual)                                # a data-generator policy
        → collect_rollouts(COMPOSED policy) → success-filter → merge into mix   # existing path
        → residual DISCARDED
```

The composed policy is passed to the **existing** `collect_rollouts` (which already produces a seed-compatible harvest — image dtype, fps 10, `panda`, real LIBERO language strings, via `--match-dataset`). The only new collector flag is `--residual-path` (load a residual and wrap the base). Everything downstream (merge, co-train) is unchanged.

### 4.2 The residual is ephemeral — a data generator, never a promoted artifact

This is the design choice that keeps the deterministic controller honest and **directly prevents the Phase-1 failure mode**:

- The **residual is discarded after the harvest.** It is tied to one frozen base checkpoint and (optionally) specific tasks; it is a means to produce **new-coverage data**, not a policy to bless.
- What flows into the flywheel is the **harvested data**, co-trained into the **next base** (DexFlyWheel step 5 = co-training, which `loop.py` already implements as a blend into the next cycle's main dataset — **not** a separate fine-tune of the just-trained checkpoint).
- Therefore **eval still judges a normally-trained base checkpoint on the standard gate.** The residual never becomes the thing `decide()` scores. This is the structural guard against the **63% → 20% catastrophic-forgetting** rollback from Phase-1 (**[verified]** in `knowledge/tasks/libero_spatial.md`: "adapting on rollouts ALONE for many steps … the decision function correctly rolled back"). The lesson — *never let an adaptation on a tiny self-generated set be the checkpoint eval/promote judges* — is preserved by construction.

### 4.3 Config surface (extends `ImproveConfig`, off by default)

```yaml
improve:
  enabled: true
  episodes: 20                 # success-filtered episodes to harvest (existing)
  residual_rl:                 # NEW, default enabled: false
    enabled: false
    target_tasks: auto         # 'auto' = tasks below a success floor in the last eval report;
                               # or an explicit list of task_ids
    train_init_states: [0, 60) # subset of the 100 init states to TRAIN on
    harvest_init_states: [60, 100)  # DISJOINT subset the composed policy harvests on
    env_steps: 150000          # per targeted task (off-policy budget; PLD≈250k, ResFiT≈75-200k)
    alpha: 0.5                 # residual scale (PLD LIBERO value)
    exploration_ramp_steps: 15000
    n_action_steps: 5          # base re-plan cadence (chunk cache); reactivity/compute trade-off
    utd: 4
    use_images: false          # v1 default: proprio+base_action; escalate to true if it stalls
    min_composed_gain: 0.05    # gate: only harvest if composed beats base on held-out init states
```

### 4.4 Which tasks the residual targets

**Recommendation:** target the tasks the **last eval report** shows below a floor — the proposer/Improve agent already has `report.per_task`. This focuses the expensive RL on the **stuck** tasks (`task_id 5` "on the ramekin" at ~2–5/10 is the clearest; `task_id 8` also lags) rather than burning GPU on tasks the base already solves. It also aligns with the M1 knowledge layer: the task pages record which tasks stay stuck, and that context can drive `target_tasks` (advisory only — never control flow).

> **Note on task identity (flagged for verification).** The Phase-2 brief names "task 5 = pick up the black bowl on the cookie box". **[verified]** in LIBERO's default `libero_spatial` order, `task_id 3` = "…black bowl **on the cookie box**…" and `task_id 5` = "…black bowl **on the ramekin**…". The discrepancy is likely 1-based indexing or a non-default `task_order_index`. **RESOLVED (verified against eval_info per_task + the LIBERO benchmark order):** the eval harness's `per_task` `task_id` matches `benchmark.get_task(i)`, and the persistently-stuck task is `task_id 5` = "black bowl **on the ramekin**" (2→5→4 /10 across the run). `task_id 3` = "…on the cookie box…" was NOT stuck (9→9→8). Earlier project notes that called the stuck task "cookie box" were a naming error; the `task_id` (5) and trajectory were right. Target by `report.per_task` success (robust to the label).

---

## 5. Eval — how we know the residual helped (and can't cheat)

Two independent signals, matching the parent design's "don't let a clever stage overfit" discipline:

### 5.1 Inner gate (was the residual worth harvesting?)

Before spending harvest budget, check the **composed** policy beats the **base** on **held-out** init states of the targeted task:
```
success(base+residual, harvest_init_states) − success(base, harvest_init_states) ≥ min_composed_gain
```
- Trained on `train_init_states`, measured on the **disjoint** `harvest_init_states` — so the residual cannot win by memorizing the states it trained on.
- If the gate fails, **skip the harvest** for that task and log it (the residual didn't help — an honest, common outcome for the hardest tasks, per §6). This prevents burning harvest/co-train budget on a residual that overfit or never learned.

### 5.2 Outer gate (did the flywheel actually improve? — the one that matters)

The **only** promotion signal remains the existing decision function on the **standard `libero_spatial` gate** (`eval_agent.py`, 10 tasks × N episodes, fixed seeds), scoring a **normally-trained base** co-trained on the enlarged dataset:
- **[verified]** `decide()` promotes on `Δ ≥ promote_delta`, **rolls back on `Δ ≤ −regression_delta`** — it already vetoes regressions, including a residual-harvest that overfits or narrows the distribution. Phase-2 does not touch it.
- **Guard against narrow overfitting** (the Phase-1 failure): the eval report is **per-task** (`report.per_task`). A residual-harvest that lifts the targeted task but **regresses others** shows up as a per-task drop and, if it pulls the aggregate down, triggers rollback. **Recommendation:** additionally log per-task deltas and flag any per-task regression > `regression_delta` for the knowledge layer even when the aggregate passes — an early-warning signal that the harvest is narrowing the policy.
- **Eval/harvest/train init-state hygiene:** the standard eval uses its own fixed seeds/protocol; keep the residual's `train_init_states` **disjoint** from the states the standard eval samples, so improvement is genuine generalization, not eval-state leakage.

### 5.3 What "success" looks like end-to-end

A Phase-2 cycle **worked** if: (inner) the composed policy lifts a stuck task on held-out init states → (harvest) success-filtered rollouts of that task now exist and merge → (outer) the next base co-trained on them **clears `promote_delta` on the standard gate without per-task regressions**. If any link breaks, the deterministic controller does the safe thing (skip harvest / iterate / rollback) with no manual intervention.

---

## 6. Feasibility — honest assessment (what could make this NOT work in sim on LIBERO)

**VRAM is fine; wall-clock is the real constraint.** **[verified/computed]** frozen SmolVLA (~2–3 GB) + SAC learner (<1 GB) + a few EGL render processes (~0.5–1.5 GB) ≈ **4–6 GB of 16 GB**. But **[verified]** the SmolVLA forward (~101 ms) dominates: at 150k env steps with `n_action_steps=5` that is ~30k base forwards **plus** MuJoCo stepping (CPU-bound, tens of Hz per worker) **plus** UTD×gradient steps. A realistic estimate is **~10–20+ GPU-hours per targeted task** — the binding budget on one GPU. Mitigations are all in the design: chunk-caching, `AsyncVectorEnv` workers, moderate UTD, and **targeting only stuck tasks** (not all 10).

Even where a piece is impractical, the pragmatic alternative is named inline. The residual **mechanism** is the fixed point; the sample budget, vision usage, and reward shaping are the tunable/fallback axes.

---

## Risks (ranked, with mitigations)

1. **The base is too weak on the stuck task → the residual has no reward to learn from.** **[verified]** the sharpest limit in the literature: ResiP — "the local nature of our residual policies is not well-suited for … large-scale deviations"; performance saturates (~70%) when base coverage is poor. **A residual cannot create a behavior (e.g. a grasp) the base never attempts.** If SmolVLA is ~0 on a task's init states, sparse-reward RL will likely see no reward and not learn. *Mitigation:* targeted/demo-guided resets that start episodes near reward (§2.2 fallback 1); if still zero, this task is **out of residual-RL's reach** and the honest answer is to route it to data curation / policy escalation (π0.5) — not to pretend the residual will fix it. **This is the top risk and the most likely reason a specific stuck task stays stuck.**
2. **Wall-clock blows the GPU budget** (§6). *Mitigation:* chunk-cache, async envs, moderate UTD, target few tasks; if still too slow, reduce `env_steps` and accept a weaker residual (harvest whatever new coverage it does produce).
3. **Chunked-base / per-step-residual mismatch.** A stale open-loop chunk the residual can't correct locally. *Mitigation:* smaller `n_action_steps` (more re-planning) trades compute for reactivity; **[verified]** this is the standard fix (ResiP/ResFiT/A2C2).
4. **Sparse reward defeats SAC despite demos.** *Mitigation:* RLPD 50/50 demo anchoring + warmup are the primary defense (**[verified]** the reason PLD/SimpleVLA-RL work on sparse LIBERO); dense-shaping fallback exists but is off by default.
5. **Harvest narrows the policy → co-training regresses other tasks** (the Phase-1 failure recurring at the data level). *Mitigation:* structural — residual is ephemeral, eval judges a co-trained base, `decide()` vetoes regressions, per-task deltas logged (§4.2, §5.2).
6. **Novelty risk.** **[verified]** no published residual **RL** on SmolVLA specifically (A2C2 is supervised; PLD is π0/OpenVLA). Bracketed by precedent on every axis but unproven for this exact combo. *Mitigation:* the phased plan front-loads a cheap single-task go/no-go (P2.1) before any loop integration.
7. **LIBERO env quirks** (auto-reset-in-step, EGL blocker, stale `~/.libero/config.yaml`, 256-vs-360 resolution, task-id ordering). *Mitigation:* each is **[verified]** and enumerated (§2.2, §3.2, §4.4) so P2.0 handles them explicitly.

---

## Phased implementation plan (each milestone validatable on one 16 GB GPU)

Milestones are ordered to **front-load the cheap go/no-go** and to keep the loop untouched until the RL core is proven. Every step is sim-only and CVE-safe by construction.

**P2.0 — Offline scaffolding (no GPU-heavy work, mostly CPU + unit tests).** — ✅ **core landed** (`leagents/rl/`): `ComposedPolicy` (per-step α·tanh residual over a cached chunk + ε-ramp, base/residual injected as callables so the math is CPU-testable), `ResidualRLConfig` (wired into `ImproveConfig`, off by default), and the CVE guard — `leagents.rl.safety.assert_cve_safe` (always-on) + a constitution `forbidden_imports` deny rule. 14 unit tests cover the composition invariants (residual=0 ⇒ composed==base, α·tanh bounding, clip, one base-forward per `n_action_steps`, ε-ramp, exploration gate) and import-safety, and it is **verified against the installed lerobot** that the SAC math + buffer import zero transport modules while `lerobot.rl.learner` drags in the pickle/gRPC surface the guard forbids. Also landed: `leagents/rl/env_adapter.py` — the verified LIBERO episode logic as pure functions (`classify_step`: success-terminates-without-bootstrap, time-limit-truncates-with-bootstrap, ignore the inner env's unreliable flags around auto-reset-in-step; `init_state_split`: disjoint train/harvest ranges) plus `ResidualRollout`, which drives a `ComposedPolicy` through an injected env, collecting `(s, a_base, a_exec, r, s', done, bootstrap)` transitions and resetting explicitly on episode end — 9 tests with a fake env cover the terminal bookkeeping and init-state cycling. **Still pending** (need lerobot + a GPU, the P2.1 RL core): the `ResidualActor`/`ResidualCritic` wrappers over `lerobot.policies.sac`, the single-process `residual_sac_train()` loop (copied SAC update), the real LIBERO driver that wraps lerobot's env behind the `ResidualRollout` interface and extracts the 8-d proprio state, and the offline `a_base` precompute over demo states.
- New module `leagents/agents/improve/residual/` (or `leagents/rl/`): `ResidualActor`/`ResidualCritic` wrappers over lerobot's `sac` modules; a `ComposedPolicy` that wraps a frozen `SmolVLAPolicy` (`predict_action_chunk` cache + per-step `α·tanh` residual + ε-ramp gate); a single-process `residual_sac_train()` loop that **copies** the ~150-line update step out of `learner.py` (no transport imports).
- Buffer bookkeeping: extend the transition to `(s, a_base, a_exec, r, s', done)`; offline pass that precomputes `a_base` over the demo (seed) episodes for the target task.
- LIBERO env adapter: per-step sparse reward, `set_init_state` train/harvest split, done-masked bootstrapping around the auto-reset-in-step quirk, explicit time limit.
- **Constitution:** add a deny rule for `lerobot.rl.actor|learner|learner_service` and `lerobot.transport` (import-level or process-level), extending the existing `forbidden_commands`/`forbidden_args`.
- **Validate:** unit tests with a **fake base** (returns fixed chunks) and a **fake/tiny env** — the full loop runs under `--dry-run`, composition math is correct (residual=0 ⇒ composed==base), buffer mixing is 50/50, no forbidden imports (a test asserts `sys.modules` is gRPC-free). *No LIBERO needed.*

**P2.1 — Single-task go/no-go (the feasibility gate). ← ✅ PASSED (2026-07-06).**
- Task 4 (base ~65% at n=5), frozen v6 cycle-2 base, RLPD demo-anchored (6257 task-4 demo transitions), 150k env-steps (~3.7 GPU-h). **Result: composed 100% vs base 65% on the held-out `harvest_init_states` → gain +0.35, gate PASS.** Residual RL produces a policy that succeeds where the base fails — the flywheel's missing new-coverage ingredient, proven.
- **The decisive finding: α is the primary knob (as §1.2/§0 said), and the design's cited α=0.5 is WRONG for SmolVLA@n=5.** At α=0.5 the residual saturates to a ~0.32/dim constant *override* (measured: 27.7% of dims at the ±α bound) and collapses the policy to 0% (self-reinforcing: success→0 ⇒ reward→0 ⇒ Q→0 ⇒ no un-saturating gradient); demo anchoring did not prevent it. At **α=0.1** the residual is a gentle corrector that cannot override a good base action, and it learned a real +0.35 improvement. **Config default is now α=0.1.** Process lesson: judge by the deterministic gate, not the noisy stochastic-collection successes (mid-run the exploration policy looked degraded at ~16% while the learned deterministic policy was 100%).

**P2.2 — Target a stuck task (the real test).**
- Point the same machinery at the stuck black-bowl task (**resolve the task_id first**, §4.4). Honestly log whether the residual reaches success on init states the base fails, or hits Risk #1 (can't create a grasp the base never attempts). Either outcome is a **valid, publishable result** and feeds the knowledge layer.

**P2.3 — Harvest integration.**
- Add `--residual-path` to `collect_rollouts` so it wraps the base with the trained residual; verify the composed-policy harvest is **seed-compatible** (reuses the existing `--match-dataset` path — image dtype, fps 10, `panda`, real language strings) and merges into the mix. *Reuses Phase-1's proven merge path end-to-end.*

**P2.4 — Full loop integration + one autonomous cycle.**
- `ImproveAgent` invokes `residual_rl` on promotion → compose → harvest → merge → co-train next base → **standard eval gate**. The inner gate (§5.1) skips unhelpful harvests; the outer gate (§5.2) + `decide()` govern promotion/rollback unchanged.
- **Validate:** one end-to-end cycle where a residual-harvested stuck task measurably lifts the co-trained base on the standard gate **without per-task regression** — or the controller safely skips/iterates/rolls back. Compare against a Phase-1 baseline (data-growth-only) on the same gate to isolate the residual's contribution.

---

## Appendix — key references (with verification status)

**DexFlyWheel template** — [arXiv:2509.23829](https://arxiv.org/abs/2509.23829) (NeurIPS 2025 Spotlight) **[verified, primary]**: residual = state-only SAC MLP `[256,256,256]` on privileged object+proprio state, dense per-task shaped reward, `π_combined = π_base + α·π_res` with prob ε, **α=0.1**, ε-ramp 1.5k→10k steps, 1.5M steps/task on OmniGibson (single RTX 4090, ~6.5 h/iteration), K=20/100/500 harvest/task over 3 iterations, base is a **small per-task diffusion policy**, not a VLA. Ablation: removing the residual is "the most significant drop."

**Frozen-VLA + residual RL (the recipe we adopt)** —
- **PLD** [arXiv:2511.00091](https://arxiv.org/abs/2511.00091) **[verified]**: frozen π0/OpenVLA + 3-layer-MLP(256) residual conditioned on `(s, a_b)`, **off-policy SAC + Cal-QL warmstart**, RLPD two-buffer, residual `[−ξ,ξ]` scheduled, **ξ=0.5 for LIBERO**, sparse reward, ~99% LIBERO-90, **250k online steps**.
- **ResFiT** [arXiv:2509.19301](https://arxiv.org/abs/2509.19301) **[verified]**: frozen chunked base + **per-step** residual on `(s_t, a_base_t)`, off-policy RLPD-flavored, LayerNorm critic, REDQ ensemble, UTD≈4, warmup; **off-policy ~200k vs PPO ~40M steps (~200×)**.
- **A2C2** [arXiv:2509.23224](https://arxiv.org/abs/2509.23224) **[verified]**: **frozen SmolVLA-450M on LIBERO** with a 32M **per-step residual over the chunk** — *supervised*, not RL, but validates the exact plumbing; SmolVLA fwd ≈101 ms vs 4.7 ms residual; LIBERO 84.2% vs 64.4% at horizon 40/delay 10.
- **Policy Decorator** [arXiv:2412.13630](https://arxiv.org/abs/2412.13630) **[verified]**: bounded residual `tanh×α` (α≈0.03–0.5), progressive ε-ramp (too-small H → total failure), critic on summed action; frozen BeT/Diffusion base.

**Classic residual RL** — Silver 2018 [arXiv:1812.06298](https://arxiv.org/abs/1812.06298) **[verified]** (`π=π_base+f_θ`, zero-init last layer, critic burn-in, `∇_θπ=∇_θf_θ` ⇒ base can be black-box); Johannink 2019 [arXiv:1812.03201](https://arxiv.org/abs/1812.03201) **[verified]** (residual TD3, ~8k real samples, base confines exploration).

**Sample-efficient sparse RL from demos** — RLPD [arXiv:2302.02948](https://arxiv.org/abs/2302.02948) **[verified]** (50/50 symmetric sampling, LayerNorm critic, E=10/subset, UTD 20); HIL-SERL [arXiv:2410.21845](https://arxiv.org/abs/2410.21845) **[verified]** (RLPD + binary success classifier + human interventions; ~100% in 1–2.5 h real).

**Direct RL on LIBERO (context)** — SimpleVLA-RL [arXiv:2509.09674](https://arxiv.org/abs/2509.09674) **[verified]** (full-param GRPO, binary reward, LIBERO-Spatial 98.7%, **8×A800**); VLA-RL [arXiv:2505.18719](https://arxiv.org/abs/2505.18719) **[verified]** (LoRA PPO + process-reward densification).

**lerobot 0.5.1 (installed source)** **[verified]** — reusable transport-free: `lerobot/policies/sac/modeling_sac.py` (`SACPolicy`, encoders, `CriticEnsemble`), `lerobot/rl/buffer.py` (`ReplayBuffer`, `from_lerobot_dataset`, `random_shift`, `concatenate_batch_transitions`), SAC update `lerobot/rl/learner.py:393-554`. **CVE surface (never import):** `lerobot/rl/actor.py`, `learner.py`, `learner_service.py`, `lerobot/transport/*` (pickle at `transport/utils.py:135`). LIBERO env: `lerobot/envs/libero.py` (per-step sparse reward, `info["is_success"]`, auto-reset-in-step, `get_task_init_states` (100 states/task), `set_init_state`, `TASK_SUITE_MAX_STEPS["libero_spatial"]=280`). SmolVLA: `lerobot/policies/smolvla/` (`chunk_size=50`, LIBERO ckpt `n_action_steps=1`, 7-d action, 8-d state, flow-matching 10-step, `predict_action_chunk` hook).

**Project internals** — `DESIGN.md` §3.5 (flywheel), §6 (CVE-2026-25874); `orchestrator/decision.py` (`decide()` — the pure promote/rollback function Phase-2 must not touch); `orchestrator/loop.py` (harvest trigger, co-training-not-adapt note); `knowledge/tasks/libero_spatial.md` (63→69→72 arc; the 63→20 catastrophic-forgetting rollback; the dry-well and suite-ordering landmines).
