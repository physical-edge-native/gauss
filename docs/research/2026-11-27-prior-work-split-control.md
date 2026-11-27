# Prior work on split versus central control policies

Date: 2026-11-27. Issue: #36. Research note written by an AI assistant (Claude Code) in a scheduled night run.
It summarises published work as reported in the papers' abstracts and main results. Nothing here was rerun, and
the note does not change the [scope record](../design/2026-11-27-scope-and-comparison.md) or the
[expected differences](../design/2026-11-27-expected-differences.md).

## Why this note

gauss compares one small controller per leg, joint or limb, which sees only its own part and its neighbours'
messages, with one central policy. Several lines of work have trained similar decentralized or modular policies
with reinforcement learning. This note records what they compared and found, and what that means for gauss's
measures (M1 computation on microcontrollers, M2 training computation, M3 tracking and stability, M4 faults and
messages, M5 ease of learning).

## 1. Decentralized deep RL for legged locomotion

**Schilling, Melnik, Ohl, Ritter, Hammer (2021).** "Decentralized control and local information for robust and
adaptive decentralized Deep Reinforcement Learning." *Neural Networks* 144: 699–725.
doi:10.1016/j.neunet.2021.09.017. Earlier hexapod version: arXiv:2005.11164.

- Compared one central controller with decentralized ones (one controller per leg, with different amounts of
  information from neighbouring legs) on a four-legged MuJoCo agent, trained with deep RL.
- Reported: learning was more robust in the decentralized cases (less sensitive to hyperparameters, less often
  stuck in poor local minima), and the authors see it as a route to better generalisation towards adaptive
  behaviour.
- For gauss: the closest published comparison to gauss's quadruped stages. It is evidence that the split form
  need not learn worse (M5), against the expectation in the scope record that split may need more steps. gauss's
  measurement should report the spread over seeds and the share of seeds reaching Q, which is where that
  paper's main difference appeared. The agent there is a simulated "Ant", not a model of a real robot, and no
  messages were learned; gauss adds learned messages, real robot models and sizes.

**Sartoretti, Paivine, Shi, Wu, Choset (2019).** "Distributed Learning of Decentralized Control Policies for
Articulated Mobile Robots." *IEEE Transactions on Robotics* 35(5): 1109–1122. doi:10.1109/TRO.2019.2922493.
arXiv:1901.08537.

- Treated each independently controlled portion of a snake robot and a hexapod as an A3C agent sharing one
  reward. Controllers learned offline (snake) and online (hexapod), and transferred to the real robots.
- For gauss: shows that per-part agents with a shared reward can be trained and run on hardware; it supports
  gauss's choice of one reward for all split controllers. It does not report a matched central baseline under an
  equal budget, which gauss does.

## 2. Robots split into agents: multi-agent MuJoCo

**de Witt, Peng, Kamienny, Torr, Böhmer, Whiteson (2020).** "Deep Multi-Agent Reinforcement Learning for
Decentralized Continuous Cooperative Control." arXiv:2003.06709 (later versions present FACMAC; cite the version
used).

- Introduced Multi-Agent MuJoCo: a MuJoCo robot split into agents (for example one per leg), each seeing only
  nearby joints.
- Found that how the value function is factorised over agents mattered most for performance.
- For gauss: `gauss/train.py` already follows the "centralized training, decentralized execution" pattern of
  this literature. Both forms use the same critic on the whole-body observation, and it is used in training only.
  The split form's actors see only their own leg. So only the actors have to fit on microcontrollers (M1), and
  `computation()` reports the critic separately as training-only. The design record could name this pattern
  explicitly.

## 3. Modular policies that pass messages along the body

**Wang, Liao, Ba, Fidler (2018).** "NerveNet: Learning Structured Policy with Graph Neural Networks." ICLR 2018.

- The policy is a graph network over the body's joints and links, passing messages along the body graph.
- For gauss: the idea behind "neighbours only". The main claim there is transfer between bodies, which gauss
  does not measure.

**Huang, Mordatch, Pathak (2020).** "One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic
Control." ICML 2020.

- One small module, shared by every joint, with messages passed up and down the body tree. One policy controlled
  many different bodies.
- For gauss: closest in form to gauss's split controllers with learned messages. Shared weights over identical
  parts cut the number of weights to train. `gauss/train.py` already shares one actor among the four legs,
  each leg's observation holding its leg identifier, and every controller holds its own copy of the weights for M1.

**Kurin, Igl, Rocktäschel, Böhmer, Whiteson (2021).** "My Body is a Cage: the Role of Morphology in Graph-Based
Incompatible Control." ICLR 2021. arXiv:2010.01856.

- Ablations found that encoding the body's structure in the message graph did not improve performance. A
  transformer over all parts (Amorpheus), which lets every part attend to every other, did better than graph
  networks that pass messages along the body.
- For gauss: direct evidence for an expected disadvantage of neighbours-only messages. Information that has to
  travel several hops along the body arrives late and compressed. The rung "all parts" of gauss's ladder
  (messages from every part) is the controlled test of this effect, and is worth keeping in the budget for at
  least one robot.

**Gupta, Fan, Ganguli, Fei-Fei (2022).** "MetaMorph: Learning Universal Controllers with Transformers." ICLR 2022.

- A transformer over all limbs, pretrained on many bodies.
- For gauss: like Amorpheus, it uses global attention. That is a central computation, so it is a reference point
  for the "body-structured central model" rung, not for the split form.

## 4. Decentralized gait control without learning

**Owaki and Ishiguro (2017).** "A quadruped robot exhibiting spontaneous gait transitions from walking to
trotting to galloping." *Scientific Reports* 7: 277.

- One oscillator per leg, coupled only through each leg's own load sensing (no explicit communication between
  legs), produced walk, trot and gallop transitions on a real quadruped as speed changed.
- For gauss and erg: an existence proof that coordination can come through the body itself (forces felt at the
  feet) rather than through messages. gauss's "no messages" rung tests whether learned per-leg controllers find
  this. It also links to poise's question of computation by the body.

## What this means for gauss

| Point | Prior work | Possible change for gauss (to be decided in a later issue) |
|---|---|---|
| Split may need more training steps (expected disadvantage, M2/M5) | Schilling et al. found decentralized learning more robust over seeds | Keep the expectation, but also report the share of seeds that reach Q and the spread of final quality, where the reported difference was |
| Neighbours-only messages vs global information | Kurin et al.: morphology-shaped message passing did not help, global attention did | Keep the "all parts" rung for at least one robot, so the cost of neighbours-only is measured |
| Shared critic during training | Multi-agent MuJoCo work: factorisation of the value function matters | Already so in the code (one whole-body critic, training only); name it in the design record. A per-leg or factorised critic is a possible later variation |
| Shared weights between identical legs | Huang et al.: one module shared by all parts | Already so in the code (one actor shared by the four legs, with a leg identifier); name it in the design record. On the arm and the humanoid, parts differ, so sharing needs a decision there |
| No messages at all | Owaki and Ishiguro: coordination through load feedback alone | The "no messages" rung is a meaningful baseline, not only a lower bound |

None of these works measured whether the policies fit on microcontrollers (gauss's M1), and none compared sizes
of the same body plan from about 1.5 kg to 40 kg, which is gauss's main addition.

## Not yet read

- Work on hierarchical or modular quadruped controllers on real commercial robots (for example per-leg policies
  on ANYmal- or Unitree-class robots), if any reports a matched central baseline.
- Multi-agent PPO variants (shared vs separate actors) applied to legged robots, for the choice of gauss's
  training algorithm.
