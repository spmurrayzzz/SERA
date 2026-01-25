"""
Dataset filtering using HuggingFace tokenizers.
Drop-in replacement for filter_dataset_tt.py without torchtune dependency.
"""

import json
import os
import copy
from typing import Any, Literal, Optional, Union

from tqdm import tqdm


# =============================================================================
# Message and Transform classes (replacing torchtune.data imports)
# =============================================================================

Role = Literal["system", "user", "assistant", "ipython", "tool"]


class Message:
    """Replacement for torchtune.data.Message"""

    def __init__(
        self,
        role: Role,
        content: Union[str, list[dict[str, Any]]],
        masked: bool = False,
        ipython: bool = False,
        eot: bool = True,
    ):
        self.role = role
        self.content = self._convert_to_list_of_dict(content)
        self.masked = masked
        self.ipython = ipython
        self.eot = eot

    def _convert_to_list_of_dict(self, content) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "content": content}]
        return content

    @property
    def text_content(self) -> str:
        return "".join(
            c["content"] for c in self.content if c["type"] == "text"
        )


class ShareGPTToMessages:
    """Replacement for torchtune.data._messages.ShareGPTToMessages"""

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[dict[str, str]] = None,
    ):
        self.train_on_input = train_on_input
        self._column_map = column_map or {"conversations": "conversations"}

    def __call__(self, sample: dict) -> dict:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        messages = []
        for msg in sample[self._column_map["conversations"]]:
            role = role_map.get(msg["from"], msg["from"])
            masked = False if self.train_on_input else (role != "assistant")
            if role == "system":
                masked = True
            messages.append(Message(role=role, content=msg["value"], masked=masked))
        return {"messages": messages}


class OpenAIToMessages:
    """Replacement for torchtune.data._messages.OpenAIToMessages"""

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[dict[str, str]] = None,
    ):
        self.train_on_input = train_on_input
        self._column_map = column_map or {"messages": "messages"}

    def __call__(self, sample: dict) -> dict:
        messages = []
        for msg in sample[self._column_map["messages"]]:
            content = msg["content"]
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", item.get("content", ""))
                    for item in content if item.get("type") == "text"
                )
            role = msg["role"]
            masked = False if self.train_on_input else (role != "assistant")
            if role == "system":
                masked = True
            messages.append(Message(role=role, content=content, masked=masked))
        return {"messages": messages}


# =============================================================================
# HuggingFace Tokenizer Wrapper
# =============================================================================

class HFTokenizerWrapper:
    """
    Wrapper that provides the interface expected by check_seq_length/count_seq_length.
    """

    def __init__(self, tokenizer_name_or_path: str, max_seq_len: Optional[int] = None):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )
        self.max_seq_len = max_seq_len or getattr(self.tokenizer, "model_max_length", 4096)
        self.prompt_template = None  # We use HF chat template instead

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _tokenize_header(self, messages: list[Message], idx: int) -> list[int]:
        return []  # Handled by chat template

    def _tokenize_footer(self, messages: list[Message], idx: int) -> list[int]:
        return []  # Handled by chat template


def create_hf_tokenizer(tokenizer_name_or_path: str, max_seq_len: Optional[int] = None) -> HFTokenizerWrapper:
    """Create an HF tokenizer wrapper."""
    return HFTokenizerWrapper(tokenizer_name_or_path, max_seq_len)


# =============================================================================
# Core filtering functions (same signatures as original)
# =============================================================================

def filter_dataset_by_tokens(tokenizer: Union[str, HFTokenizerWrapper], data_file: str, tokens: int,
                              conversation_style: str = "openai",
                              conversation_column: str = "messages",
                              train_on_input: bool = False):
    """Filters dataset, removing any instances over the token limit."""
    with open(data_file, "r") as f:
        messages = [json.loads(line) for line in f.readlines()]
    file_name, file_type = os.path.splitext(data_file)

    if isinstance(tokenizer, str):
        tokenizer_name = os.path.basename(tokenizer)
        tokenizer = create_hf_tokenizer(tokenizer)
    else:
        tokenizer_name = "hf_tokenizer"

    print("Filtering dataset...")
    filtered_messages = filter_messages(
        tokenizer, messages,
        custom_limit=tokens,
        conversation_style=conversation_style,
        conversation_column=conversation_column,
        train_on_input=train_on_input,
    )
    print(len(filtered_messages))
    output_file = f"{file_name}_filtered_len{len(filtered_messages)}_tok{tokens}_{tokenizer_name}{file_type}"
    with open(output_file, "w") as f:
        for fltr_sample in filtered_messages:
            f.write(json.dumps(fltr_sample) + "\n")
    print(f"Saved filtered data to {output_file}")
    return output_file


def filter_dataset(tokenizer: Union[str, HFTokenizerWrapper], messages, truncate=False,
                   return_token_to_data_tuples=False,
                   conversation_style: str = "openai",
                   conversation_column: str = "messages",
                   train_on_input: bool = False,
                   custom_limit: int = -1):
    """Apply filtering to a list of messages."""
    print("Filtering dataset...")
    if isinstance(tokenizer, str):
        tokenizer = create_hf_tokenizer(tokenizer)
    return filter_messages(
        tokenizer, messages,
        truncate=truncate,
        return_token_to_data_tuples=return_token_to_data_tuples,
        conversation_style=conversation_style,
        conversation_column=conversation_column,
        train_on_input=train_on_input,
        custom_limit=custom_limit,
    )


def filter_messages(tokenizer: HFTokenizerWrapper, messages: list, custom_limit: int = -1,
                    truncate: bool = False, return_token_to_data_tuples: bool = False,
                    conversation_style: str = "openai",
                    conversation_column: str = "messages",
                    train_on_input: bool = False):
    """Core filtering logic."""
    assert (not return_token_to_data_tuples) or (return_token_to_data_tuples and truncate)

    filtered_messages = []
    if return_token_to_data_tuples:
        token_to_data_tuples = []

    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": conversation_column},
        )
    elif conversation_style == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=train_on_input,
            column_map={"messages": conversation_column},
        )
    else:
        raise ValueError(f"Unknown conversation_style: {conversation_style}")

    for sample in tqdm(messages):
        transformed_sample = message_transform(sample).pop("messages")
        if not truncate:
            valid_seq_length = check_seq_length(tokenizer, transformed_sample, custom_limit=custom_limit)
        else:
            valid_seq_length = check_seq_length(tokenizer, transformed_sample, custom_limit=custom_limit, truncate=truncate)
            if not isinstance(valid_seq_length, bool):
                old_sample_length = len(sample["messages"])
                sample = truncate_messages(sample, valid_seq_length)
                if sample:
                    kept_sample_proportion = len(sample["messages"]) / old_sample_length
                    valid_seq_length = True
                else:
                    valid_seq_length = False
            elif valid_seq_length:
                kept_sample_proportion = 1
        if valid_seq_length:
            filtered_messages.append(sample)
            if return_token_to_data_tuples:
                token_to_data_tuples.append((kept_sample_proportion, sample))

    print(f"Filtered {len(messages)} to {len(filtered_messages)}")
    if return_token_to_data_tuples:
        return filtered_messages, token_to_data_tuples
    return filtered_messages


def truncate_messages(sample, truncate_idx):
    """Truncate sample at the given index, keeping complete assistant turns."""
    last_assistant_message = -1
    new_sample = copy.deepcopy(sample)
    for i, message in enumerate(sample["messages"]):
        if truncate_idx == i:
            if last_assistant_message < 0:
                return None
            else:
                new_sample["messages"] = sample["messages"][:last_assistant_message+1]
                return new_sample
        if message["role"] == "assistant":
            last_assistant_message = i
    return None


def apply_chatml_template(messages: list[Message]) -> list[Message]:
    """
    Apply ChatML formatting to messages.

    ChatML format:
        <|im_start|>{role}
        {content}<|im_end|>
    """
    CHATML_TEMPLATE = {
        "system": ("<|im_start|>system\n", "<|im_end|>\n"),
        "user": ("<|im_start|>user\n", "<|im_end|>\n"),
        "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        "ipython": ("<|im_start|>ipython\n", "<|im_end|>\n"),
        "tool": ("<|im_start|>tool\n", "<|im_end|>\n"),
    }

    templated = []
    for msg in messages:
        prefix, suffix = CHATML_TEMPLATE.get(msg.role, ("", ""))
        new_content = prefix + msg.text_content + suffix
        templated.append(Message(
            role=msg.role,
            content=new_content,
            masked=msg.masked,
            eot=msg.eot,
        ))
    return templated


def check_seq_length(
    tokenizer: HFTokenizerWrapper,
    messages: list[Message],
    *,
    add_end_tokens: bool = True,
    custom_limit: int = -1,
    truncate: bool = False
) -> Union[bool, int]:
    """
    Check if messages fit within token limit.
    Applies ChatML template before tokenizing for accurate counts.

    Returns:
        bool: True if fits, False if doesn't
        int: If truncate=True and too long, returns message index to truncate at
    """
    limit = custom_limit if custom_limit > 0 else tokenizer.max_seq_len

    # Apply ChatML template to get accurate token counts
    templated_messages = apply_chatml_template(messages)

    # Tokenize each message and accumulate
    total_tokens = 0
    for i, message in enumerate(templated_messages):
        tokens = tokenizer.encode(message.text_content)
        total_tokens += len(tokens)

        if total_tokens >= limit:
            if truncate:
                return i
            else:
                return False

    return True


def count_seq_length(
    tokenizer: HFTokenizerWrapper,
    messages: list[Message],
    *,
    add_end_tokens: bool = True,
    custom_limit: int = -1
) -> tuple[int, int]:
    """
    Count generated (assistant) and prefilled (other) tokens.
    Applies ChatML template before tokenizing for accurate counts.

    Returns:
        tuple[int, int]: (generated_count, prefilled_count)
    """
    # Apply ChatML template for accurate counts
    templated_messages = apply_chatml_template(messages)

    generated_count = 0
    prefilled_count = 0

    for orig_msg, templated_msg in zip(messages, templated_messages):
        tokens = tokenizer.encode(templated_msg.text_content)
        if orig_msg.role == "assistant":
            generated_count += len(tokens)
        else:
            prefilled_count += len(tokens)

    return generated_count, prefilled_count


def count_tokens(tokenizer: Union[str, HFTokenizerWrapper], messages: list, custom_limit: int = -1,
                 conversation_style: str = "openai",
                 conversation_column: str = "messages",
                 train_on_input: bool = False):
    """Count total generated and prefilled tokens across all samples."""
    if isinstance(tokenizer, str):
        tokenizer = create_hf_tokenizer(tokenizer)

    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": conversation_column},
        )
    elif conversation_style == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=train_on_input,
            column_map={"messages": conversation_column},
        )
    else:
        raise ValueError(f"Unknown conversation_style: {conversation_style}")

    total_generated = 0
    total_prefilled = 0
    for sample in tqdm(messages):
        transformed_sample = message_transform(sample).pop("messages")
        generate, prefill = count_seq_length(tokenizer, transformed_sample, custom_limit=custom_limit)
        total_generated += generate
        total_prefilled += prefill

    print(len(messages))
    print(total_generated)
    print(total_prefilled)
    return total_generated, total_prefilled
