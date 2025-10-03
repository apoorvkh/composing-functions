# How Do Language Models Compose Functions?

[[Paper]](https://arxiv.org/abs/2510.01685)

**Apoorv Khandelwal & Ellie Pavlick**

**Abstract:** While large language models (LLMs) appear to be increasingly capable of solving compositional tasks, it is an open question whether they do so using compositional mechanisms. In this work, we investigate how feedforward LLMs solve two-hop factual recall tasks, which can be expressed compositionally as $g(f(x))$. We first confirm that modern LLMs continue to suffer from the "compositionality gap": i.e. their ability to compute both $z = f(x)$ and $y = g(z)$ does not entail their ability to compute the composition $y = g(f(x))$. Then, using logit lens on their residual stream activations, we identify two processing mechanisms, one which solves tasks *compositionally*, computing $f(x)$ along the way to computing $g(f(x))$, and one which solves them *directly*, without any detectable signature of the intermediate variable $f(x)$. Finally, we find that which mechanism is employed appears to be related to the embedding space geometry, with the idiomatic mechanism being dominant in cases where there exists a linear mapping from $x$ to $g(f(x))$ in the embedding spaces.

### Installation

```bash
# Install the uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync --frozen

cp .env.local.example .env.local
# Update .env.local with user-specific variables
```

### Activate Environment

```bash
source .venv/bin/activate
source .env
source .env.local
```

## Usage

This codebase offers tools to [prompt](./src/composing_functions/prompting.py) and perform [mechanistic analyses](./src/composing_functions/lens.py) on a selection of [models](`./src/composing_functions/models/__init__.py`) and [compositional tasks](`src/composing_functions/tasks/__init__.py`).

Beyond this, we implement our [experiments](./src/composing_functions/experiments) as modular pipelines, run using the [AI2 Tango](https://github.com/allenai/tango) library. Each `Experiment` implements a `step_dict` method that defines that pipeline's steps (which can depend on other steps, like a graph). The outputs of steps are automatically cached in `./tango_workspace` by our scripts.

To replicate the experiments and plots (output to `artifacts/`) in our paper, one can run:

- Generate datasets for every task

```bash
python configs/generate_data.py
python plotting/tasks_table.py  # Tables 1-2
```

- Evaluate all models on a few tasks

```bash
python configs/evaluation_by_model.py
python plotting/compositionality_gap_by_model.py  # Fig. 2
python plotting/compositionality_gap_by_size.py  # App. C
```

- Evaluate Llama 3 (3B) on all tasks

```bash
python configs/llama_3_3b/evaluation.py
python plotting/compositionality_gap.py  # Fig. 1
```

- Logit lens analyses on all tasks

```bash
python configs/llama_3_3b/lens.py
python plotting/logit_lens_overall.py  # Fig. 3(a-b)
python plotting/logit_lens_per_task.py  # Fig. 3(c-f), App. D-E
python plotting/intermediate_var_distribution.py  # Fig. 4(b)
```

- Correlation between task linearity and intermediate variables

```bash
python configs/llama_3_3b/linear_task_embedding.py
python configs/llama_3_3b/lens.py
python plotting/task_logit_lens_corr.py  # Fig. 4(a)
python plotting/hop_logit_lens_corr.py  # App. H
```

- Token identity patchscope analyses

```bash
python configs/llama_3_3b/lens_token_identity.py
python plotting/token_identity_per_task.py  # App. F, Fig. 10-11
python configs/llama_3_3b/linear_task_embedding.py
python plotting/task_token_identity_correlation.py  # App. F, Fig. 12
```

- Causality of intermediate variable representations

```bash
python configs/llama_3_3b/patching_across_tasks_comp.py
python configs/llama_3_3b/patching_across_tasks_direct.py
python plotting/patching_across_tasks.py  # App. G
```

## Citation

```bibtex
@misc{khandelwal2025:compose,
      title={How Do Language Models Compose Functions?}, 
      author={Apoorv Khandelwal and Ellie Pavlick},
      year={2025},
      eprint={2510.01685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.01685}, 
}
```
