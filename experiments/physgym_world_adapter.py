"""
Adapter that wraps a PhyEnv as a v2/v3-style WorldFnSimulation.

Exposes:
  input_spec / output_spec  -- dict[name -> {"dtype": "float", "shape": []}]
  query(x: dict) -> dict    -- run env_function; raise on error / complex / non-finite
  sample_input() -> dict    -- log-uniform on [10^lo, 10^hi] by default; reject on query failure

Lets the v2-style interactive prediction harness drive PhysGym environments without
any modification to recipes/simulation. Optional Level 4 ("no_description_anonymous")
mode renames inputs to var_1, var_2, ... and the output to var_obs.

Dummy variables (defined in the env's input_variables but unused by env_function)
are exposed to the model as input fields so the model has to discover that they
don't matter; they're silently dropped before calling env_function.

Per-variable sampling overrides come from physgym_env_overrides.json (see that file
for format). They let us encode "this input must be an integer" (env 457: N polygon
sides) or "this input must be negative" (env 443: metal permittivity) which
log-uniform-on-positives can't otherwise produce. Envs without overrides fall back
to the default log-uniform sampler; envs whose constraints are unsatisfiable
(e.g. env 284) will raise from sample_input — that's the intended signal.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from physgym.phyenv import PhyEnv

from recipes.simulation.world_env import _round_floats

_DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "physgym_env_overrides.json"


@lru_cache(maxsize=8)
def load_env_overrides(path: str | None = None) -> dict:
    """Read the per-env overrides JSON; return {env_id_str: entry_dict}."""
    p = Path(path) if path else _DEFAULT_OVERRIDES_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


class PhysGymWorldAdapter:
    """v2/v3-compatible wrapper over a PhyEnv."""

    def __init__(
        self,
        env_id,
        mode: str = "no_description_anonymous",
        sample_log10_range: tuple[float, float] = (-2.0, 2.0),
        max_sample_attempts: int = 200,
        samples_file: str | Path | None = None,
        overrides_path: str | Path | None = None,
    ):
        # env_id may be int/str (loads full_samples.json) or a pre-loaded sample dict
        # (skips disk I/O — preferred when constructing many adapters in a loop).
        self._env = PhyEnv(env_id, samples_file=str(samples_file) if samples_file else None)
        self.id = self._env.id
        self.mode = mode
        self.sample_log10_range = sample_log10_range
        self.max_sample_attempts = max_sample_attempts

        all_overrides = load_env_overrides(str(overrides_path) if overrides_path else None)
        self._var_overrides: dict = all_overrides.get(str(self.id), {}).get("_overrides", {})

        # Public input names = input vars + dummy vars (both shown to the model).
        # env_function only accepts the real input vars; dummies get dropped at query time.
        real_inputs = list(self._env.input_variables_des.keys())
        dummies = list(self._env.dummy_variables_des.keys())
        orig_inputs_all = real_inputs + dummies

        anonymous = "anonymous" in mode
        if anonymous:
            self._public_to_orig = {f"var_{i+1}": p for i, p in enumerate(orig_inputs_all)}
            self._public_output_name = "var_obs"
        else:
            self._public_to_orig = {p: p for p in orig_inputs_all}
            self._public_output_name = next(iter(self._env.output_variable_des.keys()))

        # Real-input set in original-name space; query() must only forward these to env_function.
        self._real_input_orig = set(real_inputs)

        self.input_spec: dict = {
            name: {"dtype": "float", "shape": []}
            for name in self._public_to_orig.keys()
        }
        self.output_spec: dict = {
            self._public_output_name: {"dtype": "float", "shape": []}
        }

        # Description shown by the system prompt. For levels that strip context, this is empty.
        if anonymous or "no_context" in mode or "no_description" in mode:
            self.description = ""
        else:
            self.description = self._env.problem_content

    # ------------------------------------------------------------------
    # Core interface used by the harness
    # ------------------------------------------------------------------

    def query(self, x: dict) -> dict:
        """Evaluate env_function on the public input dict. Raises on any failure."""
        kwargs = {}
        for pub_name, val in x.items():
            orig = self._public_to_orig[pub_name]
            if orig in self._real_input_orig:
                ov = self._var_overrides.get(orig)
                if ov is not None and ov.get("dtype") == "int":
                    # env_function may guard with isinstance(N, int); honor that.
                    kwargs[orig] = int(val)
                else:
                    kwargs[orig] = float(val)

        missing = self._real_input_orig - kwargs.keys()
        if missing:
            raise ValueError(f"missing required inputs (orig names): {sorted(missing)}")

        y = self._env.execute(**kwargs)
        if isinstance(y, str) or isinstance(y, complex):
            raise ValueError(f"non-real output: {y!r}")
        y = float(y)
        if not math.isfinite(y):
            raise ValueError(f"non-finite output: {y}")
        return {self._public_output_name: _round_floats(y)}

    def _sample_one_var(self, orig_name: str) -> float:
        """Sample a single input value, honoring per-variable overrides."""
        ov = self._var_overrides.get(orig_name)
        if ov is None:
            lo, hi = self.sample_log10_range
            return float(10.0 ** np.random.uniform(lo, hi))

        dtype = ov.get("dtype", "float")
        rng = ov.get("value_range")
        if dtype == "int":
            assert rng is not None, f"int override for {orig_name} needs value_range"
            return int(np.random.randint(int(rng[0]), int(rng[1]) + 1))
        # float
        if ov.get("log_uniform") and rng is not None:
            lo, hi = rng
            same_sign = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
            if not same_sign:
                raise ValueError(f"log_uniform override on {orig_name} requires same-sign bounds")
            sign = 1.0 if lo > 0 else -1.0
            mag_lo, mag_hi = sorted([abs(lo), abs(hi)])
            return float(sign * 10.0 ** np.random.uniform(math.log10(mag_lo), math.log10(mag_hi)))
        if rng is not None:
            return float(np.random.uniform(rng[0], rng[1]))
        lo, hi = self.sample_log10_range
        return float(10.0 ** np.random.uniform(lo, hi))

    def sample_input(self) -> dict:
        """Sample one valid input dict, honoring per-var overrides; reject on env-side failures.

        Only ValueError / ArithmeticError from env_function are treated as "this draw was
        invalid, try again." Anything else propagates — it's a bug in the adapter or override.
        """
        for _ in range(self.max_sample_attempts):
            x = {
                pub_name: self._sample_one_var(self._public_to_orig[pub_name])
                for pub_name in self._public_to_orig.keys()
            }
            try:
                _ = self.query(x)
            except (ValueError, ArithmeticError):
                continue
            return _round_floats(x)
        raise RuntimeError(
            f"PhysGym env {self.id}: failed to sample valid input in "
            f"{self.max_sample_attempts} attempts (default range 10^{self.sample_log10_range})"
        )
