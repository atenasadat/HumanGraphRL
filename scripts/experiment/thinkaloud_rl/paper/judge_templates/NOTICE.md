# Native judge templates

Templates retain their upstream wording, under Apache License 2.0 (see LICENSE).
Task rubrics are supplied separately by HumanGraphRL.

- `prometheus_absolute.txt`: ABS_SYSTEM_PROMPT followed by ABSOLUTE_PROMPT_WO_REF,
  from [Prometheus Eval](https://github.com/prometheus-eval/prometheus-eval/blob/dcfb44272d5d0414832f5dbb8c2a05ebc2614234/libs/prometheus-eval/prometheus_eval/prompts.py).
  Copyright the Prometheus Eval contributors. Combined into one user turn for
  the model's Mistral conversation template.
- `selene_absolute.txt`: user_template from
  [Atla Selene Mini](https://github.com/atla-ai/selene-mini/blob/418c5ce2d8c4d89c5bd971ce60c7589eb8f9d8de/prompt-templates/absolute-scoring.yaml).
  Copyright Atla and the Selene Mini contributors. YAML indentation removed;
  rendered with the model's Llama 3 conversation template.
