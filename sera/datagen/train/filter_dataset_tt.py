import json
import os
import copy
import yaml

from tqdm import tqdm

from omegaconf import OmegaConf
from torchtune import config
from torchtune.data._messages import OpenAIToMessages, ShareGPTToMessages
from torchtune.data import Message

def filter_dataset_by_tokens(cfg: dict, data_file: str, tokens: int):
    # Filters dataset, removing any instances over the token limit
    with open(data_file, "r") as f:
        messages = [json.loads(line) for line in f.readlines()]
    file_name, file_type = os.path.splitext(data_file)
    tokenizer_name = cfg["tokenizer"]["_component_"].split(".")[-1]
    output_file = f"{file_name}_filtered_{tokenizer_name}{file_type}"
    # if os.path.isfile(output_file):
    #     print("Using existing filtered dataset")
    #     return output_file

    # Apply filtering
    print("Filtering dataset...")
    filtered_messages = filter_messages(cfg, messages, custom_limit=tokens)
    print(len(filtered_messages))
    output_file = f"{file_name}_filtered_len{len(filtered_messages)}_tok{tokens}_{tokenizer_name}{file_type}"
    with open(output_file, "w") as f:
        for fltr_sample in filtered_messages:
            f.write(json.dumps(fltr_sample) + "\n")
    print(f"Saved filtered data to {output_file}")
    return output_file

def filter_dataset(cfg: dict, messages, truncate=False, return_token_to_data_tuples=False):
    # Apply filtering
    print("Filtering dataset...")
    return filter_messages(cfg, messages, truncate=truncate, return_token_to_data_tuples=return_token_to_data_tuples)

def filter_messages(cfg: dict, messages: list, custom_limit: int = -1, truncate: bool = False, return_token_to_data_tuples: bool = False):
    assert (not return_token_to_data_tuples) or (return_token_to_data_tuples and truncate)
    filtered_messages = []
    if return_token_to_data_tuples:
        token_to_data_tuples = []
    if cfg["dataset"]["conversation_style"] == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=cfg["dataset"]["train_on_input"],
            column_map={"conversations": cfg["dataset"]["conversation_column"]},
        )
    elif cfg["dataset"]["conversation_style"] == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=cfg["dataset"]["train_on_input"],
            column_map={"messages": cfg["dataset"]["conversation_column"]},
        )
    tokenizer_cfg = cfg["tokenizer"]
    if isinstance(tokenizer_cfg, dict):
        tokenizer_cfg = OmegaConf.create(tokenizer_cfg)

    tokenizer = config.instantiate(tokenizer_cfg)
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
        # else:
        #     if "instance_id" in sample:
        #         print(f"Removed {sample['instance_id']}")
    print(f"Filtered {len(messages)} to {len(filtered_messages)}")
    if return_token_to_data_tuples:
        return filtered_messages, token_to_data_tuples
    return filtered_messages

def truncate_messages(sample, truncate_idx):
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

def check_seq_length(
    tokenizer,
    messages: list[Message],
    *,
    add_end_tokens: bool = True,
    custom_limit: int = -1,
    truncate: bool = False
) -> tuple[list[int], list[bool]]:
    """
    Given a list of messages, return a list of tokens for the concatenated
    and formatted messages.

    Args:
        messages (list[Message]): The message list to tokenize.
        add_end_tokens (bool): Wether to add the tokenizer's end of message
            tokens, such as  eos_id. Default is True.

    Returns:
        tuple[list[int], list[bool]]: The list of token ids and the list of masks.

    Raises:
        RuntimeError: If a message contains non-text content
    """
    templated_messages = (
        tokenizer.prompt_template(messages)
        if tokenizer.prompt_template is not None
        else messages
    )

    tokenized_messages = []
    mask = []
    unmasked_start_token = -1
    unmasked_stop_token = -1
    for i, message in enumerate(templated_messages):
        if not message.masked and unmasked_start_token == -1:
            unmasked_start_token = len(tokenized_messages) - 1
        # message header
        tokens = tokenizer._tokenize_header(templated_messages, i)

        # message content
        for item in message.content:
            if item["type"] == "text":
                tokens.extend(
                    tokenizer.encode(
                        item["content"],
                        add_bos=False,
                        add_eos=False,
                    )
                )
            else:
                raise RuntimeError(
                    f"Unsupported message content type: {item['type']}"
                )

        # message footer
        tokens.extend(tokenizer._tokenize_footer(templated_messages, i))

        tokenized_messages.extend(tokens)
        mask.extend([message.masked] * len(tokens))
        if not message.masked and unmasked_stop_token == -1:
            unmasked_stop_token = len(tokenized_messages) - 1

        # Break out early if we reach max_seq_len
        if custom_limit < 0:
            if tokenizer.max_seq_len and len(tokenized_messages) >= tokenizer.max_seq_len:
                # print(f"ALERT ALERT ALERT, EARLIEST UNMASKED: {unmasked_start_token}-{unmasked_stop_token}, MAX SEQ IS {self.max_seq_len}")
                if truncate:
                    # truncated_msgs = truncate_messages(messages, truncate_idx=i)
                    # if truncated_msgs:
                    #     return truncated_msgs
                    return i
                else:
                    return False
        else:
            if len(tokenized_messages) >= custom_limit:
                return False
    return True


def count_seq_length(
    tokenizer,
    messages: list[Message],
    *,
    add_end_tokens: bool = True,
    custom_limit: int = -1
) -> tuple[list[int], list[bool]]:
    """
    Given a list of messages, return a list of tokens for the concatenated
    and formatted messages.

    Args:
        messages (list[Message]): The message list to tokenize.
        add_end_tokens (bool): Wether to add the tokenizer's end of message
            tokens, such as  eos_id. Default is True.

    Returns:
        tuple[list[int], list[bool]]: The list of token ids and the list of masks.

    Raises:
        RuntimeError: If a message contains non-text content
    """
    templated_messages = (
        tokenizer.prompt_template(messages)
        if tokenizer.prompt_template is not None
        else messages
    )

    tokenized_messages = []
    mask = []
    unmasked_start_token = -1
    unmasked_stop_token = -1
    generated_count = 0
    prefilled_count = 0
    for i, message in enumerate(templated_messages):
        if not message.masked and unmasked_start_token == -1:
            unmasked_start_token = len(tokenized_messages) - 1
        # message header
        tokens = tokenizer._tokenize_header(templated_messages, i)

        # message content
        for item in message.content:
            if item["type"] == "text":
                tokens.extend(
                    tokenizer.encode(
                        item["content"],
                        add_bos=False,
                        add_eos=False,
                    )
                )
            else:
                raise RuntimeError(
                    f"Unsupported message content type: {item['type']}"
                )

        # message footer
        tokens.extend(tokenizer._tokenize_footer(templated_messages, i))

        if message.role == "assistant":
            generated_count += len(tokens)
        else:
            prefilled_count += len(tokens)
    return generated_count, prefilled_count


def count_tokens(cfg: dict, messages: list, custom_limit: int = -1):
    filtered_messages = []
    if cfg["dataset"]["conversation_style"] == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=cfg["dataset"]["train_on_input"],
            column_map={"conversations": cfg["dataset"]["conversation_column"]},
        )
    elif cfg["dataset"]["conversation_style"] == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=cfg["dataset"]["train_on_input"],
            column_map={"messages": cfg["dataset"]["conversation_column"]},
        )
    tokenizer = config.instantiate(cfg["tokenizer"])
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
