# Experiment Workbench Layout

`experiments.py` is now a stable compatibility facade. It preserves historical
imports and the command:

```bash
python -m expert_game_lab.experiments ...
```

The current implementation lives in `experiments_impl.py`. That file is still a
large research workbench, but moving it behind a facade makes it safe to split
incrementally without breaking old scripts or tests.

Recommended next splits:

1. `experiment_models.py`
   Dataclasses and small typed result rows.

2. `experiment_formatting.py`
   Formatting helpers such as action strings, optional floats, interval output,
   and report table utilities.

3. `experiment_libraries.py`
   Restricted action libraries, edge signatures, interval helpers, and strategy
   class lookup.

4. `time_policy.py`
   Explicit time-dependent policy registry, reachable evaluator, cutoff/mask
   policies, and boundary-loss utilities.

5. `k5_proof.py`
   Chase5, random-playout, softmax, scalar-exchangeability, boundary-correction,
   and relaxation-certificate diagnostics.

6. `k6_candidates.py`
   k=6 alive, interval, barycentric, greedy patch, and long-horizon diagnostics.

7. `k9_replay.py`
   two-run replay, skeleton/template mining, cutoff/mask sweeps, and flow-role
   reports.

Keep `experiments.py` as the public facade until downstream scripts stop
importing private helpers from it. New code should prefer importing from the
smaller modules once they exist.
