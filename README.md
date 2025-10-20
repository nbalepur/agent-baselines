# agent-baselines

The repo contains baseline implementations of a variety of agents to as reported in [AstaBench](https://github.com/allenai/asta-bench).

These agents are implemented as InspectAI solvers and can be run on the AstaBench suit with the `astabench eval` command.

## Usage

```bash
git clone https://github.com/allenai/agent-baselines.git
```

The `solvers` directory has descriptions of each of the baseline agents, along with sample scripts to set up the environment required by that solver and
run against a limit number of eval samples.

If you have trouble setting up an environment on your local system, a Docker image is provided.  For a given `<solver_name>` (e.g. `react`):

```commandline
make shell SOLVER=<solver_name>
```
This should put you in a Docker shell like `root:/agent-baselines# `; then you can set up your environment (either via `.env` for Docker or `export` in the shell; see [asta-bench](https://github.com/allenai/asta-bench) for more details and how to get HF and Asta keys) and run the solver:

```commandline
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>
export GOOGLE_API_KEY=<your-google-key>
export ASTA_TOOL_KEY=<your-asta-tool-key>
export HF_TOKEN=<your-huggingface-key>

./solvers/<solver_name>/demo.sh
```

See documentation in [asta-bench](https://github.com/allenai/asta-bench) for more details on how to run the suite.


## Available Agents

AstaBench includes several built-in solvers. Look in the `solvers` directory for setup and demo scripts.

### General agent baselines

- [**Basic ReAct agent**](/solvers/react/): a simple ReAct implementation that uses LLM tool-calling in a loop. It supports all [AstaBench tool options](https://github.com/allenai/asta-bench?tab=readme-ov-file#tools-and-utilities) and configurable `max_steps`.
- [**Smolagents coder**](/solvers/smolagents/): integrates HuggingFace's [CodeAgent](https://github.com/huggingface/smolagents), which takes actions by writing Python code rather than using JSON tool-calling. Requires a sandbox and uses `SandboxToolManager` to access tools from sandboxed code.

### Asta Science Agents
- [**Asta-v0**](/solvers/asta-v0/): an agent that routes tasks to specialized agents based on the detected task type.
- [**Asta ScholarQA**](/solvers/sqa/): Long-form question answering system with retrieval and reranking, designed for the SQA task.  This is a variant of the system that powers [Asta Summarize Literature](https://asta.allen.ai/synthesize).
- [**Asta DataVoyager**](/solvers/datavoyager/): Agent for data analysis and exploration tasks, designed for the DiscoveryBench task
- [**Asta Paper Finder**](/solvers/paper_finder): Agent for paper search and recommendation, designed for the PaperFinder task.   This is a variant of the system that powers [Asta Find Papers](https://asta.allen.ai/discover).
- [**Asta Tables**](/solvers/arxivdigestables/): Agent for building lit-review tables, designed for the ArxivDIGESTables task
- [**Asta Code**](/solvers/super/): ReAct-style agent designed specifically for SUPER-style repository reproduction tasks.
- [**Asta Panda**](/solvers/e2e_discovery/): Agent specialized for the E2E-discovery and E2E-discovery-hard tasks.


### Other baselines
- [**STORM**](/solvers/storm/): Knowledge curation system for comprehensive report generation.  Works on the `sqa_dev` and `sqa_test` tasks.
- [**FutureHouse**](/solvers/futurehouse/): Literature review and scientific writing agent.  Works on the `sqa_dev` and `sqa_test` tasks.

Each solver directory contains `setup.sh` for installation and `demo.sh` with example commands.

## Running ScholarQA evaluations with dvc
[`dvc`](dvc.org) is used to perform "release" runs of the SQA evaluation. If you are just debugging, there is no need to use `dvc`. The evalution suite is tracked in `dvc.yaml` where the stages of the pipeline are defined along with the inputs and outputs of each stage. These are cached on s3 and shared across systems.

### Getting started

First, install `dvc`:

``` bash
pip install 'dvc[s3]'
```

The main command for dvc is `dvc repro`. This checks to see which stages in your pipeline need to be run and runs them. It uses a cache to avoid rerunning stages unnecessarily. Please check the dvc documentation for more details if you are interested.

To force the pipeline to run (ignoring the cachhe):

``` bash
dvc repro --force
```

If you just want to run a single stage (the "@" specifies the loop value to run):

``` bash
dvc repro solve_sqa@claude-3.7
```

### When to force a stage to run
`dvc repro` only executes stages for which:
* Any of the `dependencies` changed. (This is determined by comparing the hash with the hash stored in `dvc.lock`)
* The definition of the stage changed (ie the `dependencies`, the `outputs` and the `cmd` as defined in `dvc.yaml`)

Sometimes we need to manually force a stage to run since the code for a stage can change (for example a bugfix or improvement to a solver or scorer) without the `cmd` for that stage changing. For example, I might change the prompt used in the scorer for SQA, in which case I should manually force the scoring stage (and all its descendants) to rerun:

``` bash
dvc repro score_sqa --force --downstream
```

I could also just rerun the `score_sqa` stage if I knew that the downstream stages shouldn't be affected by this:
``` bash
dvc repro score_sqa --force --single-item
```

You can check if the `dependencies` of any stages have changed by running `dvc status`:

``` bash
summarize_scores:
  changed deps:
    modified:           dvc_logs/scored/task_sqa_solver_sqa_claude-3.7.eval
```
This tells me that the scores for the claude 3.7 solver have changed and I need to rerun the stage which aggregates the results into a table (`summarize_scores`). You can do this by running `dvc repro`.

### Collaborating with others on the same project
When someone pushes changes to the lockfile `dvc.lock` then you can checkout those changing by pulling from the cache:

``` bash
dvc pull
```

This updates your local files with the versions pushed by your collaborator.

You can push your local changes using:

``` bash
dvc push
```
then your collaborators can pull your changes by running `dvc pull`.
