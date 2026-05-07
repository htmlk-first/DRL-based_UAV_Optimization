# Common Experiment Utilities

This folder contains shared helpers used by the per-algorithm experiment
folders.  Each algorithm folder should remain runnable on its own, while this
package keeps repeated experiment plumbing consistent.

## Standard Folder Shape

Each algorithm folder is expected to follow this shape:

```text
<algorithm>/
  env/
    __init__.py
    config.py
    uav_env.py
  <algorithm>_agent.py
  train.py
  test.py
  visualize.py
  results/
```

Algorithm-specific learning logic stays in the folder-local agent and training
loop.  Shared path handling, log reading, success detection, and evaluation
summary formatting should use `common.experiment`.

