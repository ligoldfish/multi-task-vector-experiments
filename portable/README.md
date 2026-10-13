# Experiment artifacts

`use_task_vector.py` loads the ordinary tensor dictionaries stored in exported
representations and applies them to a matching base model. `--inspect FILE`
prints a representation's structure without loading model weights.

`export_portable.py --code PROJECT --experiment RUN --config CONFIG --output DIR`
exports completed representations, their training evidence, and the experiment
materials needed for relocation. It creates `vectors-TIMESTAMP.zip` and
`experiment-TIMESTAMP.zip`. Neither contains Qwen base weights or optimizer/RNG
checkpoints. New files are written without altering the source experiment.

`portable_import.py` restores an extracted experiment archive to a new directory.
Use `portable_run.py --destination DIRECTORY inventory`, then `preflight`,
`smoke`, and `all --scope mvp --split test_formal` to inspect and continue it.
Exported representations from a different documented source revision can be
attached to the current implementation with `tools/attach_vectors.py`.

See `docs/VECTORS.md` and `docs/RESUME.md` in the repository for complete examples.
