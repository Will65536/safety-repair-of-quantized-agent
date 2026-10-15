from safe_quant.scoring import continuation_token_id, render_prompt, token_count


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        ) + "<assistant>"


def test_prompt_truncation_fits_and_preserves_ends() -> None:
    tokenizer = CharacterTokenizer()
    text = "abcdefghijklmnopqrstuvwxyz"
    empty_prompt, _ = render_prompt(tokenizer, "", 10_000)

    prompt, truncated = render_prompt(tokenizer, text, len(empty_prompt) + 8)

    assert truncated is True
    assert token_count(tokenizer, prompt) <= len(empty_prompt) + 8
    assert "abcd" in prompt
    assert "wxyz" in prompt


def test_candidate_is_validated_as_one_token() -> None:
    tokenizer = CharacterTokenizer()
    prompt, _ = render_prompt(tokenizer, "hello", 10_000)

    assert continuation_token_id(tokenizer, prompt, "A") == ord("A")
