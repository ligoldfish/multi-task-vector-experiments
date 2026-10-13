"""Compatibility alias for run_modelmate_full.py; identical complete workflow."""
from ops.modelmate_job import launch

if __name__ == "__main__":
    raise SystemExit(launch("full-formal"))
