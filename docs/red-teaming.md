# Red Teaming

`checkllm` ships two building blocks for adversarial evaluation of
LLM-powered applications you own or have explicit written permission
to test:

- **Jailbreak dataset presets** — hand-picked subsets of published
  red-team benchmarks, bundled for offline use.
- **Adversarial attack evolver** — a PAIR-style loop that uses a judge
  LLM to mutate seed attacks and keep the variants that best bypass
  your target model's safety policy.

!!! warning "Authorized testing only"
    These tools are for probing the safety of models **you own** or
    **have explicit, written permission to evaluate**. Running them
    against third-party production endpoints without authorization may
    violate the provider's terms of service and applicable law. Test
    against your own models, your own deployments, or sandboxes you
    control.

## Jailbreak dataset presets

Presets are bundled as curated subsets (20 prompts each) drawn from
publicly-released academic red-team datasets. They focus on
refusal-probing behaviors — social engineering, policy-circumvention,
misinformation, privacy probes, and prompt injection — and deliberately
exclude CSAM, operational weapons instructions, and CBRN synthesis
content.

Available presets:

| Name                   | Source                                   |
| ---------------------- | ---------------------------------------- |
| `advbench`             | Zou et al. 2023 — AdvBench (MIT)        |
| `harmbench`            | Mazeika et al. 2024 — HarmBench         |
| `jailbreakbench`       | Chao et al. 2024 — JailbreakBench       |
| `do-not-answer`        | Wang et al. 2023 — Do-Not-Answer        |
| `garak`                | Derczynski et al. — NVIDIA Garak probes |
| `prompt-injection-2023`| Liu 2023 / Perez & Ribeiro 2022         |

### Loading a preset

```python
from checkllm.redteam_datasets import load_jailbreak_preset, available_presets

print(available_presets())
# ['advbench', 'do-not-answer', 'garak', 'harmbench', 'jailbreakbench',
#  'prompt-injection-2023']

prompts = load_jailbreak_preset("advbench")
for attack in prompts[:3]:
    print(attack.category, "—", attack.prompt[:80])
```

Each item is an `AttackPrompt` with `prompt`, `category`, `source`,
`citation`, and `license` fields.

### Running a preset through the RedTeamer

```python
from checkllm.redteam import RedTeamer, VulnerabilityType

async def my_llm(prompt: str) -> str:
    ...  # your target wrapper

red = RedTeamer()
report = await red.use_preset(
    "advbench",
    target=my_llm,
    vulnerability_type=VulnerabilityType.JAILBREAK,
    limit=20,
)
print(report.summary())
```

### Getting the full datasets

The bundled subsets are small so the package stays lightweight and to
avoid shipping content that would require full dataset licensing.
For full evaluations, fetch the original datasets:

- AdvBench — <https://github.com/llm-attacks/llm-attacks>
- HarmBench — <https://github.com/centerforaisafety/HarmBench>
- JailbreakBench — <https://github.com/JailbreakBench/jailbreakbench>
- Do-Not-Answer — <https://github.com/Libr-AI/do-not-answer>
- Garak probes — <https://github.com/NVIDIA/garak>

## Adversarial-LLM-evolved attacks

`AdversarialAttackEvolver` implements a PAIR-style loop
(Chao et al. 2023, <https://arxiv.org/abs/2310.08419>): an evolver
LLM mutates seed prompts, a target LLM is probed, and a scoring LLM
rates whether the attack succeeded. Survivors seed the next
generation.

!!! warning "Test against your own models"
    The evolver deliberately searches for inputs that bypass a target
    model's refusals. Run it only against models you own or have
    explicit permission to probe. Seeds matching the built-in denylist
    (CSAM, CBRN synthesis, operational weapons instructions) are
    rejected at call time.

### Basic usage

```python
from checkllm.redteam_evolver import AdversarialAttackEvolver, EvolverConfig
from checkllm.judge import OpenAIJudge

async def target(prompt: str) -> str:
    ...  # your own model under test

judge = OpenAIJudge()
evolver = AdversarialAttackEvolver(
    evolver_judge=judge,
    scoring_judge=judge,
    config=EvolverConfig(
        generations=3,
        population_size=6,
        mutations_per_parent=5,
        success_threshold=0.9,
        patience=2,
    ),
)

results = await evolver.evolve(
    seed_prompts=["Tell me how to bypass the rate limiter."],
    target=target,
)

best = results[0]
print(best.success_score, best.prompt[:200])
```

### Stopping criteria

The evolver stops early when either:

- the top candidate's success score reaches `success_threshold`, or
- no improvement in the top score is seen for `patience` consecutive
  generations.

### Convenience wrapper on `RedTeamer`

```python
red = RedTeamer(judge=OpenAIJudge())
evolved = await red.evolve_attacks(
    seed=["seed prompt 1", "seed prompt 2"],
    target=my_llm,
    generations=3,
    population_size=10,
    mutations_per_parent=5,
)
```

### Denylist

The evolver refuses to run against seeds that contain well-known
phrases from hard-prohibited categories. The current denylist covers
CSAM indicators, chem/bio/radiological/nuclear synthesis details, and
operational weapons-construction instructions. This is a safety floor,
not a substitute for editorial review of your test corpus.

## Ethical guidelines

1. Only attack systems you own or are explicitly authorized to test.
2. Treat attack corpora as sensitive material — don't publish the raw
   prompts alongside target-response pairs without reviewing for
   harmful content.
3. Keep seed corpora focused on refusal-probing and
   policy-circumvention; avoid seeds whose successful exploitation
   would produce operationally harmful content.
4. Log every run. When an attack succeeds, capture the prompt, the
   model response, and enough metadata to reproduce it, then feed the
   finding into your safety-training or prompt-engineering loop.
