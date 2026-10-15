"""Prompt rendering and A/B classification helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from safe_quant.benchmarks.finguard import SYSTEM_PROMPT


def _messages(user_text: str) -> List[Dict[str, str]]:
    wrapped = (
        "<UNTRUSTED_USER_MESSAGE>\n"
        + user_text
        + "\n</UNTRUSTED_USER_MESSAGE>"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": wrapped},
    ]


def _render(tokenizer: Any, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        _messages(user_text), tokenize=False, add_generation_prompt=True
    )


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def render_prompt(tokenizer: Any, user_text: str, max_length: int) -> Tuple[str, bool]:
    """Render a prompt, preserving both ends of input when truncation is needed."""
    prompt = _render(tokenizer, user_text)
    if token_count(tokenizer, prompt) <= max_length:
        return prompt, False

    empty_prompt = _render(tokenizer, "")
    if token_count(tokenizer, empty_prompt) >= max_length:
        raise ValueError(f"max_length={max_length} cannot fit the FinGuard prompt")

    message_ids = tokenizer.encode(user_text, add_special_tokens=False)
    keep = min(len(message_ids), max_length - token_count(tokenizer, empty_prompt))
    while keep >= 0:
        head = (keep + 1) // 2
        tail = keep // 2
        kept_ids = message_ids[:head]
        if tail:
            kept_ids += message_ids[-tail:]
        shortened = tokenizer.decode(
            kept_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prompt = _render(tokenizer, shortened)
        length = token_count(tokenizer, prompt)
        if length <= max_length:
            return prompt, True
        keep -= max(1, length - max_length)

    raise ValueError(f"max_length={max_length} cannot fit the FinGuard prompt")


def continuation_token_id(tokenizer: Any, prompt: str, candidate: str) -> int:
    """Return a one-token continuation ID, validating the prompt boundary."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    combined_ids = tokenizer.encode(prompt + candidate, add_special_tokens=False)
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Candidate {candidate!r} changes tokenization at the prompt boundary"
        )
    continuation = combined_ids[len(prompt_ids) :]
    if len(continuation) != 1:
        raise ValueError(
            f"Candidate {candidate!r} must be exactly one token; got {continuation}"
        )
    return int(continuation[0])


def prepare_batch(
    tokenizer: Any,
    examples: Sequence[Dict[str, Any]],
    max_length: int,
) -> Tuple[Dict[str, Any], List[bool]]:
    prompts = []
    truncated = []
    for example in examples:
        prompt, was_truncated = render_prompt(tokenizer, example["text"], max_length)
        prompts.append(prompt)
        truncated.append(was_truncated)
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if int(encoded["input_ids"].shape[1]) > max_length:
        raise AssertionError("Rendered prompt exceeded max_length")
    return encoded, truncated
