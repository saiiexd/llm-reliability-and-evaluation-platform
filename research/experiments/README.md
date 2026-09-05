# Research Experiments

This directory contains the definitions, configurations, observations, results, and failure analyses for all research experiments run in this project.

## What Belongs Here

Each experiment is documented as a self-contained directory or file containing:

- **Experiment definition**: The research question, hypothesis, and scope.
- **Experimental design**: What is being tested, what is held constant, and what varies.
- **Controlled variables**: Configuration, dataset version, model version, evaluation method version.
- **Dataset reference**: The dataset used, its version, and how to obtain it.
- **Baseline**: The reference configuration against which results are compared.
- **Results**: Measured outcomes in a structured, reproducible format.
- **Analysis**: Statistical or qualitative interpretation of results.
- **Failure analysis**: What went wrong, what was unexpected, what the results do not explain.
- **Conclusion**: What was learned.
- **Limitations**: What the experiment cannot conclude.
- **Next steps**: What the next experiment should address.

## What Does Not Belong Here

- Raw, unreviewed output dumps from models or evaluation scripts.
- Partially run experiments without a documented result and conclusion.
- Experiment configurations without a documented research question.

## Directory Structure Convention

```
experiments/
  README.md                     This file.
  EXP-001_experiment_name/      One directory per experiment.
    definition.md               Research question, hypothesis, design, variables.
    config.yaml                 Exact configuration used (when applicable).
    results.md                  Measurements, analysis, and conclusion.
    failure_analysis.md         What failed or was unexpected.
```

## Experiment Numbering

Experiments are numbered sequentially (EXP-001, EXP-002, ...). The number is permanent and does not change if the experiment order is reorganized. Numbers are assigned when an experiment is formally defined, not when it is run.

## Reproducibility Requirement

Every experiment in this directory must be fully reproducible from its documented definition. If an experiment cannot be reproduced without undocumented steps, it is not considered complete.
