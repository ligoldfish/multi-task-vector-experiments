"""Task instructions, surface controls, and full-answer tokenization.

Every source intervention sees exactly the target task's rendered instruction.
The assistant prefix is part of the prompt, including Qwen's non-thinking prefix.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import Sample

TASK_ALIASES = {
    "gsm8k": "math", "math_reasoning": "math", "proofwriter": "logic",
    "logical_deduction": "logic", "iwslt2017": "translation", "iwslt": "translation",
    "qqp": "paraphrase", "paraphrasing": "paraphrase", "mbpp": "code_generation",
    "samsum": "summary", "summarization": "summary", "repair": "code_repair",
}
TASKS = ("math", "logic", "translation", "paraphrase", "code_generation", "code_repair", "summary")
VARIANTS = ("canonical", "prompt_paraphrase", "output_format", "label_recode")
LABEL_MAP = {"True": "A", "False": "B", "Unknown": "C"}


def canonical_task(task: str) -> str:
    task = TASK_ALIASES.get(task.lower(), task.lower())
    if task not in TASKS and task != "joint":
        raise ValueError(f"Unknown task: {task}")
    return task


def task_instruction(task: str, variant: str = "canonical") -> str:
    task = canonical_task(task)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
    canonical = {
        "math": "Solve the mathematical problem. Explain your reasoning and end with '#### ' followed by the final numerical answer.",
        "logic": "Using only the stated facts and rules, determine whether the query is True, False, or Unknown. Unknown means neither the query nor its negation follows. Reply with exactly one of: True, False, Unknown.",
        "translation": "Translate the following English text into German. Output only the German translation.",
        "paraphrase": "Rewrite the following question with the same meaning using different wording. Output only the rewritten question.",
        "code_generation": "Write a complete Python program that solves the described task. Output only Python code, including the requested function definition.",
        "code_repair": "Repair the supplied Python program so it solves the described task. Output only the complete corrected Python program.",
        "summary": "Summarize the following dialogue in English. Preserve the main events, participants and outcomes. Output only the summary.",
        "joint": "Complete both independent subtasks. Use the instructions for each subtask. Return exactly a JSON object with string fields 'task_a' and 'task_b', containing the complete answers to subtasks A and B respectively.",
    }
    rewritten = {
        "math": "Work out the answer to this mathematics question. Show the solution steps, then put the numerical result after '#### ' on the final line.",
        "logic": "Decide the query from the provided premises alone. Answer True if it follows, False if its negation follows, and Unknown if neither follows. Give only that label.",
        "translation": "Render this English passage in German, retaining its meaning. Give the translation without commentary.",
        "paraphrase": "Express this question in another way while preserving what it asks. Give just the new question.",
        "code_generation": "Implement the requested functionality in Python. Provide the entire implementation, with the specified function, and no explanation.",
        "code_repair": "Correct the faulty Python implementation below. Return the full working implementation for the stated requirements, without commentary.",
        "summary": "Give a concise English account of this conversation, including its key people, events, and outcome. Provide the account alone.",
        "joint": canonical["joint"],
    }
    instruction = rewritten[task] if variant == "prompt_paraphrase" else canonical[task]
    if variant == "label_recode" and task == "logic":
        instruction = "Using only the stated facts and rules, classify the query. Use A for True, B for False, and C for Unknown (neither the query nor its negation follows). Output exactly A, B, or C."
    if variant == "output_format" and task != "joint":
        instruction += " Encode the complete answer as the string value in a JSON object with the single key 'answer'. Do not use a Markdown fence."
    return instruction


def target_text(sample: "Sample", variant: str = "canonical") -> str:
    target = sample.target
    if variant == "label_recode" and canonical_task(sample.task) == "logic":
        target = LABEL_MAP[target]
    if variant == "output_format" and canonical_task(sample.task) != "joint":
        target = json.dumps({"answer": target}, ensure_ascii=False)
    return target


def render_prompt(tokenizer: Any, sample: "Sample", variant: str = "canonical") -> str:
    messages = [
        {"role": "system", "content": task_instruction(sample.task, variant)},
        {"role": "user", "content": sample.prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_sample(tokenizer: Any, sample: "Sample", variant: str = "canonical", max_prompt: int = 512, max_target: int = 512) -> dict:
    prompt = render_prompt(tokenizer, sample, variant)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    answer_ids = list(tokenizer.encode(target_text(sample, variant), add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("A defined EOS token is required for full-answer supervision")
    if not answer_ids or answer_ids[-1] != eos:
        answer_ids.append(eos)
    if not prompt_ids or len(prompt_ids) > max_prompt or len(answer_ids) > max_target:
        raise ValueError(f"length_exceeded:{sample.id}:prompt={len(prompt_ids)},target_with_eos={len(answer_ids)}; limits={max_prompt},{max_target}; no truncation")
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "prompt_length": len(prompt_ids), "id": sample.id,
        "target_length": len(answer_ids), "variant": variant,
    }


def parse_json_object(text: str) -> dict:
    """Parse an output object without silently resolving duplicate JSON keys."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON output key: {key}")
            result[key] = value
        return result
    result = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(result, dict):
        raise ValueError("A JSON output object is required")
    return result


def unwrap_prediction(prediction: str, variant: str = "canonical", task: str | None = None) -> str:
    text = prediction.strip()
    if variant == "output_format":
        try:
            obj = parse_json_object(text)
            if isinstance(obj, dict) and set(obj) == {"answer"} and isinstance(obj["answer"], str):
                return obj["answer"]
        except (ValueError, TypeError):
            return ""  # Wrong output format is not silently repaired.
        return ""
    if variant == "label_recode" and task and canonical_task(task) == "logic":
        return {value: key for key, value in LABEL_MAP.items()}.get(text, "")
    return text
