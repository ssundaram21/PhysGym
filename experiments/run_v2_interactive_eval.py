#!/usr/bin/env python3
"""
v2/v3-style interactive evaluation harness over PhysGym environments.

Protocol (matches recipes/simulation/v2 + v3 prompting and scoring; NOT PhysGym's
canonical equation-discovery eval):

  1. System prompt describes the function I/O schema and the query budget.
  2. Each turn, the model proposes up to --batch-size queries as a single
     ###QUERIES: [...] JSON array. Results are fed back in the next turn.
  3. Loop until the total --budget of queries is spent OR --max-turns reached.
  4. Eval phase: both modes are scored on --k-eval held-out inputs (mean of
     per-point tier rewards), so program vs numerical results are apples-to-apples.
       numerical (v2-style): K independent prediction turns; each turn asks for
         ###ANSWER: {...} for one x.
       program (v2_program-style): one Python predict(x) function; run in a
         sandboxed subprocess on the K inputs.

By default this runs PhysGym's Level 4 ("no_description_anonymous") mode: inputs
are renamed to var_1, var_2, ..., and the output to var_obs. Inputs/exemplars are
log-uniform on 10^[-2, 2] with reject-on-failure.

Outputs one JSON per env to <output-dir>/<group>/<mode>/<model-slug>/.
"""

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

# --- path setup: this script lives at PhysGym/experiments/; we need both the
# PhysGym package (../) and the verl repo (../../) on sys.path.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PHYSGYM_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _PHYSGYM_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_PHYSGYM_ROOT))

from physgym.utils.llm_providers import get_recommended_provider, load_api_key  # noqa: E402

from recipes.simulation.v2.interact_and_eval import (  # noqa: E402
    ANSWER_FORMAT,
    _render_schema,
    compute_rl_reward,
    parse_prediction,
)
from recipes.simulation.v2_program.interact_and_eval import (  # noqa: E402
    EVAL_PHASE_INSTRUCTIONS,
    evaluate_program,
    extract_predict_code,
)
from recipes.simulation.v3.interact_and_eval import (  # noqa: E402
    QUERIES_FORMAT,
    parse_queries,
)

from experiments.physgym_world_adapter import PhysGymWorldAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Chat helper: thin wrapper that talks directly to chat-completions endpoints,
# unlike physgym.utils.llm_providers.generate_with_provider which is single-turn.
# Supports OpenAI-compatible APIs (vllm, openrouter, openai) and Anthropic.
# ---------------------------------------------------------------------------

_OPENAI_COMPAT = {"openrouter", "openai", "vllm", "ollama"}

_DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}


def chat_completion(
    messages: list[dict],
    *,
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float = 600.0,
) -> tuple[str, str | None]:
    """Returns (content, reasoning_content). reasoning_content is None if the
    server didn't surface a separate reasoning trace."""
    provider = provider.lower()
    # Strip any non-standard fields (e.g. reasoning_content we stash for logging)
    # before sending the message list back to the API.
    clean = [{"role": m["role"], "content": m["content"]} for m in messages]
    if provider in _OPENAI_COMPAT:
        url_root = base_url or _DEFAULT_BASE_URLS.get(provider)
        if url_root is None:
            raise ValueError(f"--base-url required for provider '{provider}'")
        url = f"{url_root.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": clean,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or None
        return content, reasoning
    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        system = ""
        chat: list[dict] = []
        for m in clean:
            if m["role"] == "system":
                system = m["content"] if not system else system + "\n\n" + m["content"]
            else:
                chat.append({"role": m["role"], "content": m["content"]})
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=chat,
        )
        text_parts, think_parts = [], []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                think_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
        return "\n".join(text_parts), ("\n".join(think_parts) or None)
    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DONE_SENTINEL = "###DONE"


def build_system_prompt(
    adapter: PhysGymWorldAdapter, budget: int, batch_size: int, eval_mode: str,
    allow_early_stop: bool, solver_memory: str | None = None,
) -> str:
    input_str = _render_schema(adapter.input_spec)
    output_str = _render_schema(adapter.output_spec, incl_range=False)
    if eval_mode == "numerical":
        tail = (
            "After your query budget is exhausted, you will be given a single new input "
            f"and must predict the function's output."
        )
    elif eval_mode == "program":
        tail = (
            "After your query budget is exhausted, you will be asked to write a Python "
            "function that implements the unknown function."
        )
    else:
        raise ValueError(f"unknown eval_mode: {eval_mode}")
    early_stop_line = (
        f"If you are confident in the function's behavior and want to end the query phase early, "
        f"emit `{DONE_SENTINEL}` anywhere in your response (with or without a final {QUERIES_FORMAT} batch). "
        f"You will then be moved directly to the prediction phase.\n\n"
        if allow_early_stop else ""
    )
    # Solver Memory.MD block, mirroring recipes/in_memory_learning/prompts.py so a
    # checkpoint trained there carries the same framing into the eval prompt.
    memory_block = (
        "# Solver Memory.MD\n\n"
        "You maintain a `Memory.MD` across episodes. It contains heuristics, anti-patterns,\n"
        "and domain observations distilled from past episodes. Read it carefully at the\n"
        "start of each episode — it is the only thing that persists.\n\n"
        "Current Solver Memory.MD:\n"
        "<<<MEMORY\n"
        f"{solver_memory}\n"
        "MEMORY>>>\n\n"
        if solver_memory else ""
    )
    return (
        "You are exploring an unknown function to figure out its behavior.\n\n"
        "The function maps typed input dictionaries to typed output dictionaries. "
        "It is deterministic: the same input always produces the same output. "
        "The function is governed by hidden parameters that you must identify through querying.\n\n"
        f"{memory_block}"
        f"Input fields:\n{input_str}\n\n"
        f"Output fields:\n{output_str}\n\n"
        "NOTE: All floats must be rounded to <=3 significant figures.\n\n"
        f"You have a budget of {budget} total queries to spend across multiple turns. "
        f"On each turn, you may propose up to {batch_size} queries as a single JSON array in the format:\n"
        f"{QUERIES_FORMAT}\n\n"
        f"{early_stop_line}"
        "Results of your queries will be returned before the next turn. "
        f"{tail}\n\n"
        "Choose queries strategically to identify the function's behavior with as few queries as possible.\n"
    )


def render_query_results(queries: list[dict], results: list) -> str:
    if not queries:
        return "(You did not provide any parseable queries.)"
    lines = []
    for i, (q, r) in enumerate(zip(queries, results), 1):
        if isinstance(r, str) and r.startswith("ERROR:"):
            lines.append(f"  Query {i}: {json.dumps(q)} -> {r}")
        else:
            lines.append(f"  Query {i}: {json.dumps(q)} -> {json.dumps(r)}")
    return "\n".join(lines)


def build_query_turn_message(
    turn_idx: int, remaining: int, batch_size: int, prior_results_text: str | None,
    allow_early_stop: bool,
) -> str:
    this_batch = min(batch_size, remaining)
    prefix = (
        f"Results of your previous queries:\n{prior_results_text}\n\n"
        if prior_results_text is not None
        else ""
    )
    early_stop_line = (
        f" Or emit `{DONE_SENTINEL}` to end the query phase early and proceed to prediction."
        if allow_early_stop else ""
    )
    return (
        f"{prefix}"
        f"Turn {turn_idx}: You have {remaining} queries remaining. "
        f"You may propose up to {this_batch} queries this turn. "
        f"Format:\n{QUERIES_FORMAT}"
        f"{early_stop_line}"
    )


def build_prediction_message(prior_results_text: str | None, x_eval: dict) -> str:
    prefix = (
        f"Results of your final queries:\n{prior_results_text}\n\n"
        if prior_results_text is not None
        else ""
    )
    return (
        f"{prefix}"
        "Your query budget is now exhausted. "
        f"Given input {json.dumps(x_eval)}, predict the function's output.\n"
        f"You may reason about the answer. Your final answer MUST follow the format: {ANSWER_FORMAT}"
    )


def build_program_message(prior_results_text: str | None) -> str:
    prefix = (
        f"Results of your final queries:\n{prior_results_text}\n\n"
        if prior_results_text is not None
        else ""
    )
    return f"{prefix}Your query budget is now exhausted.\n\n{EVAL_PHASE_INSTRUCTIONS}"


# ---------------------------------------------------------------------------
# Per-env eval
# ---------------------------------------------------------------------------

# Distinct primes so the exemplar/eval seed streams don't alias.
_EVAL_SEED_PRIME = 104729


def _sample_eval_inputs(adapter: PhysGymWorldAdapter, k: int) -> tuple[list[dict], list[dict]]:
    """Draw k held-out (x, y) pairs from the adapter, retrying internal sample failures."""
    xs, ys = [], []
    for _ in range(k):
        x = adapter.sample_input()
        y = adapter.query(x)
        xs.append(x)
        ys.append(y)
    return xs, ys


def _log(env_id, label: str, text: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n[env {env_id}] {label}\n{sep}\n{text}")


def _run_numerical_eval(
    messages: list[dict],
    eval_inputs: list[dict],
    eval_outputs: list[dict],
    output_spec: dict,
    prior_results_text: str | None,
    llm_call,
    threshold_strict: float,
    threshold_partial: float,
    env_id=None,
) -> dict:
    """K independent prediction calls, fired concurrently; reward = mean of per-point tiers.

    Each call's prompt is base_trajectory + one `###ANSWER: {...}` ask for one x.
    Because the calls are independent (no model sees another's prediction), we send
    them in parallel — vLLM batches them server-side and remote APIs parallelize
    them too. Sequential would be K× slower for no information gain.
    """
    pred_message_lists = [
        messages + [{"role": "user", "content": build_prediction_message(prior_results_text, x)}]
        for x in eval_inputs
    ]
    for i, pred_msgs in enumerate(pred_message_lists):
        _log(env_id, f"EVAL PREDICTION {i+1}/{len(pred_message_lists)} USER", pred_msgs[-1]["content"])
    with ThreadPoolExecutor(max_workers=len(pred_message_lists)) as ex:
        responses = list(ex.map(llm_call, pred_message_lists))
    for i, (content, reasoning) in enumerate(responses):
        if reasoning:
            _log(env_id, f"EVAL PREDICTION {i+1}/{len(responses)} REASONING", reasoning)
        _log(env_id, f"EVAL PREDICTION {i+1}/{len(responses)} ASSISTANT", content)

    per_preds = [parse_prediction(c, output_spec) for c, _ in responses]
    per_rewards = [
        float(compute_rl_reward(
            p, y, output_spec,
            threshold_strict=threshold_strict, threshold_partial=threshold_partial,
        ))
        for p, y in zip(per_preds, eval_outputs)
    ]

    # Append all K (user, assistant) pairs to the transcript for audit. Order is
    # input order, not call-completion order — the calls were independent.
    for pred_msgs, (content, reasoning) in zip(pred_message_lists, responses):
        messages.append(pred_msgs[-1])
        asst_msg = {"role": "assistant", "content": content}
        if reasoning:
            asst_msg["reasoning_content"] = reasoning
        messages.append(asst_msg)

    return {
        "preds": per_preds,
        "per_eval_rewards": per_rewards,
        "parsed": all(p is not None for p in per_preds),
        "reward": float(np.mean(per_rewards)) if per_rewards else 0.0,
    }


def run_one_env(
    adapter: PhysGymWorldAdapter,
    *,
    budget: int,
    batch_size: int,
    max_turns: int,
    llm_call,
    eval_seed: int,
    eval_mode: str,
    threshold_strict: float,
    threshold_partial: float,
    k_eval: int,
    per_call_timeout: float,
    allow_early_stop: bool,
    solver_memory: str | None = None,
) -> dict:
    # Sample k_eval held-out (x, y) pairs deterministically per env so reruns are stable.
    # Both modes use the same K so program-vs-numerical scores are apples-to-apples.
    np.random.seed(eval_seed + int(adapter.id) * _EVAL_SEED_PRIME)
    eval_inputs, eval_outputs = _sample_eval_inputs(adapter, k_eval)
    if eval_mode not in ("numerical", "program"):
        raise ValueError(f"unknown eval_mode: {eval_mode}")

    system = build_system_prompt(adapter, budget, batch_size, eval_mode, allow_early_stop, solver_memory)
    _log(adapter.id, "SYSTEM PROMPT", system)
    messages: list[dict] = [{"role": "system", "content": system}]

    remaining = budget
    turn_idx = 0
    all_queries: list[dict] = []
    all_results: list = []
    prior_results_text: str | None = None
    stopped_early = False

    while remaining > 0 and turn_idx < max_turns:
        turn_idx += 1
        user_msg = build_query_turn_message(
            turn_idx, remaining, batch_size, prior_results_text, allow_early_stop,
        )
        messages.append({"role": "user", "content": user_msg})
        _log(adapter.id, f"TURN {turn_idx} USER", user_msg)

        content, reasoning = llm_call(messages)
        asst_msg = {"role": "assistant", "content": content}
        if reasoning:
            asst_msg["reasoning_content"] = reasoning
        messages.append(asst_msg)
        if reasoning:
            _log(adapter.id, f"TURN {turn_idx} REASONING", reasoning)
        _log(adapter.id, f"TURN {turn_idx} ASSISTANT", content)

        this_batch = min(batch_size, remaining)
        queries = parse_queries(content, adapter.input_spec, this_batch)
        results: list = []
        for q in queries:
            try:
                results.append(adapter.query(q))
            except Exception as e:
                results.append(f"ERROR: {type(e).__name__}: {e}")

        all_queries.extend(queries)
        all_results.extend(results)
        remaining -= len(queries)
        prior_results_text = render_query_results(queries, results)
        if queries:
            _log(adapter.id, f"TURN {turn_idx} QUERY RESULTS ({len(queries)} queries)", prior_results_text)

        if allow_early_stop and DONE_SENTINEL in content:
            stopped_early = True
            _log(adapter.id, f"TURN {turn_idx} EARLY STOP", f"Detected {DONE_SENTINEL} in response.")
            break

    # Prediction phase
    out = {
        "env_id": adapter.id,
        "mode": adapter.mode,
        "eval_mode": eval_mode,
        "budget": budget,
        "batch_size": batch_size,
        "eval_inputs": eval_inputs,
        "eval_outputs": eval_outputs,
        "n_queries_spent": len(all_queries),
        "n_turns_used": turn_idx,
        "stopped_early": stopped_early,
        "queries": all_queries,
        "results": [r if not isinstance(r, str) else {"error": r} for r in all_results],
    }

    if eval_mode == "numerical":
        num_out = _run_numerical_eval(
            messages, eval_inputs, eval_outputs, adapter.output_spec,
            prior_results_text, llm_call, threshold_strict, threshold_partial,
            env_id=adapter.id,
        )
        out.update(num_out)
    else:  # program
        prog_msg = build_program_message(prior_results_text)
        messages.append({"role": "user", "content": prog_msg})
        _log(adapter.id, "PROGRAM PHASE USER", prog_msg)
        pred_content, pred_reasoning = llm_call(messages)
        prog_asst = {"role": "assistant", "content": pred_content}
        if pred_reasoning:
            prog_asst["reasoning_content"] = pred_reasoning
        messages.append(prog_asst)
        if pred_reasoning:
            _log(adapter.id, "PROGRAM PHASE REASONING", pred_reasoning)
        _log(adapter.id, "PROGRAM PHASE ASSISTANT", pred_content)

        pred_code = extract_predict_code(pred_content)
        reward, diag = evaluate_program(
            pred_code, eval_inputs, eval_outputs, adapter.output_spec,
            threshold_strict=threshold_strict, threshold_partial=threshold_partial,
            per_call_timeout=per_call_timeout,
        )
        out["pred_code"] = pred_code
        out["parsed"] = diag["parsed"] >= 1.0
        out["reward"] = float(reward)
        out["program_diag"] = diag

    out["messages"] = messages
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", type=int, default=None,
                   help="If set, run only this env. Otherwise iterate --idx-start..--idx-end.")
    p.add_argument("--idx-start", type=int, default=0)
    p.add_argument("--idx-end", type=int, default=None)
    p.add_argument("--mode", default="no_description_anonymous",
                   help="PhysGym mode: default | no_context | no_description | no_description_anonymous")
    p.add_argument("--budget", type=int, default=100, help="Total query budget per env (default: 100).")
    p.add_argument("--batch-size", type=int, default=20, help="Max queries per turn (default: 20).")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Cap on query turns; defaults to ceil(budget/batch_size) + 1.")
    p.add_argument("--allow-early-stop", action="store_true",
                   help=f"If set, model may emit `{DONE_SENTINEL}` to end the query phase early and skip to evaluation.")
    p.add_argument("--sample-log10-range", default="-3,3",
                   help="Comma-separated log10 lo,hi for input sampling. Default '-3,3'.")
    p.add_argument("--threshold-strict", type=float, default=0.05)
    p.add_argument("--threshold-partial", type=float, default=0.20)
    p.add_argument("--eval-seed", type=int, default=0)

    p.add_argument("--eval-mode", choices=["numerical", "program"], default="numerical",
                   help="numerical: K independent ###ANSWER turns (matches v2). program: one predict(x) Python function, evaluated on K inputs (matches v2_program).")
    p.add_argument("--k-eval", type=int, default=5,
                   help="Number of held-out inputs to score on (both modes; mean of per-point tier rewards).")
    p.add_argument("--per-call-timeout", type=float, default=1.0,
                   help="program mode only: SIGALRM timeout per predict() call inside the sandbox.")

    p.add_argument("--llm-model", required=True)
    p.add_argument("--api-provider", default=None)
    p.add_argument("--api-key-file", default="api_keys.env")
    p.add_argument("--base-url", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=8000)

    p.add_argument("--solver-memory-path", default=None,
                   help="Path to a Solver Memory.MD (e.g. one produced by recipes/in_memory_learning). "
                        "If set, its contents are injected into the system prompt the same way they "
                        "are during in_memory_learning training. Use --group-name to keep runs separate.")

    p.add_argument("--output-dir", default="/data/scratch/shobhita/physgym_v2eval")
    p.add_argument("--group-name", default="v2_interactive")
    p.add_argument("--overwrite", action="store_true", help="Re-run envs whose result file already exists.")
    return p.parse_args()


def model_slug(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


_SAMPLES_PATH = _PHYSGYM_ROOT / "physgym" / "samples" / "full_samples.json"


def load_full_samples() -> dict:
    """Load full_samples.json once and return {env_id: sample_dict}."""
    with open(_SAMPLES_PATH) as f:
        return {s["id"]: s for s in json.load(f)}


def main():
    args = parse_args()

    provider = args.api_provider or get_recommended_provider()
    api_key = None
    if provider not in {"ollama", "vllm"}:
        api_key = load_api_key(env_file=args.api_key_file, provider=provider)
        if not api_key:
            print(f"Error: no API key for provider {provider} in {args.api_key_file}")
            sys.exit(1)

    lo_str, hi_str = args.sample_log10_range.split(",")
    sample_log10_range = (float(lo_str), float(hi_str))

    max_turns = args.max_turns
    if max_turns is None:
        max_turns = (args.budget + args.batch_size - 1) // args.batch_size + 1

    solver_memory: str | None = None
    if args.solver_memory_path is not None:
        solver_memory = Path(args.solver_memory_path).read_text()
        print(f"Loaded solver memory from {args.solver_memory_path} ({len(solver_memory)} chars).")

    slug = model_slug(args.llm_model)
    out_root = Path(args.output_dir) / args.group_name / f"{args.mode}_{args.eval_mode}" / slug
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to: {out_root}")
    print(f"Provider: {provider} | Model: {args.llm_model} | Mode: {args.mode} | Eval: {args.eval_mode}")
    print(f"Budget: {args.budget} | batch_size: {args.batch_size} | max_turns: {max_turns} | k_eval: {args.k_eval}")

    def llm_call(messages):
        return chat_completion(
            messages,
            provider=provider,
            model=args.llm_model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    samples_by_id = load_full_samples()
    env_ids = list(samples_by_id.keys())
    if args.env_id is not None:
        if args.env_id not in samples_by_id:
            print(f"Error: env_id {args.env_id} not in full_samples.json")
            sys.exit(1)
        target_ids = [args.env_id]
    else:
        end = args.idx_end if args.idx_end is not None else len(env_ids)
        target_ids = env_ids[args.idx_start:end]

    print(f"Will evaluate {len(target_ids)} environment(s).")
    summary = []
    for env_id in target_ids:
        results_path = out_root / f"experiment_{env_id}_results.json"
        if results_path.exists() and not args.overwrite:
            print(f"[env {env_id}] result file exists, skipping.")
            with open(results_path) as f:
                summary.append(json.load(f))
            continue

        t0 = datetime.datetime.now()
        print(f"\n[env {env_id}] starting at {t0.isoformat(timespec='seconds')}")
        try:
            adapter = PhysGymWorldAdapter(
                env_id=samples_by_id[env_id],
                mode=args.mode,
                sample_log10_range=sample_log10_range,
            )
            result = run_one_env(
                adapter,
                budget=args.budget,
                batch_size=args.batch_size,
                max_turns=max_turns,
                llm_call=llm_call,
                eval_seed=args.eval_seed,
                eval_mode=args.eval_mode,
                threshold_strict=args.threshold_strict,
                threshold_partial=args.threshold_partial,
                k_eval=args.k_eval,
                per_call_timeout=args.per_call_timeout,
                allow_early_stop=args.allow_early_stop,
                solver_memory=solver_memory,
            )
        except Exception as e:
            print(f"[env {env_id}] FAILED: {type(e).__name__}: {e}")
            result = {
                "env_id": env_id,
                "error": f"{type(e).__name__}: {e}",
                "reward": 0.0,
                "parsed": False,
            }

        result["duration_seconds"] = (datetime.datetime.now() - t0).total_seconds()
        result["model"] = args.llm_model
        result["provider"] = provider
        result["solver_memory_path"] = args.solver_memory_path
        result["solver_memory_chars"] = len(solver_memory) if solver_memory else 0
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
        history_path = out_root / f"experiment_{env_id}_history.json"
        with open(history_path, "w") as f:
            json.dump(result.get("messages", []), f, indent=2)
        summary.append(result)
        print(
            f"[env {env_id}] reward={result.get('reward', 0):.3f} "
            f"parsed={result.get('parsed', False)} "
            f"queries={result.get('n_queries_spent', 0)} "
            f"turns={result.get('n_turns_used', 0)} "
            f"({result['duration_seconds']:.1f}s)"
        )

    if summary:
        rewards = [r.get("reward", 0.0) for r in summary]
        parsed = sum(1 for r in summary if r.get("parsed", False))
        print(f"\n=== Aggregate ({len(summary)} envs) ===")
        print(f"  mean reward: {np.mean(rewards):.4f}")
        print(f"  median reward: {np.median(rewards):.4f}")
        print(f"  parse rate: {parsed}/{len(summary)}")


if __name__ == "__main__":
    main()
