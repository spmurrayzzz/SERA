import argparse
import json
import os
import random
import re
import time
import yaml

from tqdm import tqdm
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', nargs="+")
    parser.add_argument('-i', '--instance-fp', nargs="+")
    parser.add_argument('-f', '--folder', nargs="+")
    parser.add_argument('-fm', '--filter-mode')
    parser.add_argument('-th', '--threshold', type=int)
    return parser.parse_args()

args = get_args()
# assert args.dataset_type in ["initial", "e2e"]

def get_edited_function_ids(diff_content: str) -> set[str]:
    """
    Return edited/added/removed Python-style function identifiers from a unified diff.

    Identifier format: "<funcname>@L<line>"
      - If the def line is present in the diff hunk body, <line> is the def line number.
      - If only the hunk header provides function context, <line> is the hunk's +start line (proxy).
    """
    edited: set[str] = set()

    # @@ -old_start,old_len +new_start,new_len @@ ...
    hunk_re = re.compile(r'^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@')

    # Context after second @@ can be: "def foo(...):" (or nothing / class / etc.)
    hunk_def_re = re.compile(r'^@@.*@@\s+(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(')

    # def line inside hunk content (context / added / removed)
    def_re = re.compile(r'^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(')

    in_hunk = False
    old_ln: Optional[int] = None
    new_ln: Optional[int] = None

    current_name: Optional[str] = None
    current_idline: Optional[int] = None  # def line if known; else proxy

    for raw in diff_content.splitlines():
        if raw.startswith('diff --git'):
            in_hunk = False
            current_name = None
            current_idline = None
            continue

        m_h = hunk_re.match(raw)
        if m_h:
            in_hunk = True
            old_ln = int(m_h.group(1))
            new_ln = int(m_h.group(2))
            current_name = None
            current_idline = None

            # If header includes "def name(", seed scope with a proxy line number
            m_hd = hunk_def_re.match(raw)
            if m_hd:
                current_name = m_hd.group(1)
                current_idline = new_ln  # proxy: hunk start line in new file
            continue

        if not in_hunk:
            continue

        # Skip file markers
        if raw.startswith(('+++', '---')):
            continue

        if not raw or raw[0] not in (' ', '+', '-'):
            continue

        prefix = raw[0]
        line = raw[1:]

        # Determine the line number for THIS line (old vs new side)
        if prefix == '+':
            this_ln = new_ln
        elif prefix == '-':
            this_ln = old_ln
        else:
            this_ln = new_ln  # context lines: prefer new-side numbering

        # If we see a def line (added/removed/context), update current scope
        m_def = def_re.match(line)
        if m_def:
            current_name = m_def.group(1)
            current_idline = this_ln  # best: actual def line number (old/new as appropriate)

            # If def line itself is added/removed, count it
            if prefix in ('+', '-') and this_ln is not None:
                edited.add(f"{current_name}@L{this_ln}")

        # Any +/- change line counts toward the most recent def we’ve seen
        if prefix in ('+', '-') and current_name and current_idline is not None:
            edited.add(f"{current_name}@L{current_idline}")

        # Advance hunk counters
        if prefix == ' ':
            old_ln += 1
            new_ln += 1
        elif prefix == '+':
            new_ln += 1
        else:  # '-'
            old_ln += 1

    return list(edited)

def analyze_diff(patch_text: str):
    added = 0
    deleted = 0
    new_files = 0

    current_file_is_new = False

    for line in patch_text.splitlines():

        # Detect entering a new file diff block
        if line.startswith("diff --git"):
            # Reset for new file
            current_file_is_new = False

        # Detect new files (git diff marks them like this)
        if line.startswith("new file mode"):
            current_file_is_new = True
            new_files += 1

        # Count added / removed lines
        # Skip diff metadata lines
        if line.startswith('+++') or line.startswith('---') or line.startswith('diff --git') or line.startswith('@@'):
            continue

        # Added lines (but not "+++" metadata)
        if line.startswith('+') and not line.startswith('+++'):
            added += 1

        # Removed lines (but not "---" metadata)
        if line.startswith('-') and not line.startswith('---'):
            deleted += 1

    return {
        "added_lines": added,
        "deleted_lines": deleted,
        "new_files": new_files
    }


def get_hunk_count(text):
    hunk_count = 0
    if text is None:
        return 0
    for line in text.splitlines():
        # Mark start of hunk (reset line count)
        if line.startswith('@@'):
            # Searches for +{start_line}, {number of lines in hunk}
            match = re.search(r'\-(\d+)(?:,(\d+))?', line)
            if match:
                hunk_count += 1
    return hunk_count

remove_ids = set()

print("Loading current datasets...")
loaded_dataset = []
for data_fp in args.dataset:
    with open(data_fp, "r") as f:
        loaded_dataset += [json.loads(line) for line in f.readlines()]
dataset_ids = set([d["instance_id"] for d in loaded_dataset])
print(f"Current dataset length: {len(loaded_dataset)}")
front, tail = os.path.splitext(args.dataset[0])

print(args.filter_mode, "=====")
if args.filter_mode == "insufficient_edit":
    print("Requires instance file path")
    for inst_fp in args.instance_fp:
        with open(inst_fp, "r") as f:
            loaded_instance_yaml = yaml.safe_load(f)
        print(inst_fp)
        for inst in tqdm(loaded_instance_yaml):
            edited_fn_names = get_edited_function_ids(inst["extra_fields"]["pred_patch"])
            hunk_count = get_hunk_count(inst["extra_fields"]["pred_patch"])
            # print(inst["extra_fields"]["pred_patch"])
            # print(edited_fn_names)
            # time.sleep(10)
            if (len(edited_fn_names) == 1 and 
                edited_fn_names[0].split("@")[0] in inst["extra_fields"]["start_fn"] and
                hunk_count == 1):
                if random.random() < 0.9:
                    remove_ids.add(inst["id"])
            #     print(f"REMOVING becuase MATCH {inst['extra_fields']['start_fn']}")
            # print(f"NOT REMOVING becuase NO MATCH {inst['extra_fields']['start_fn']}")
            # print("=====================")
elif args.filter_mode == "long_edit" or args.filter_mode == "soft_edit":
    print("Requires data directory")
    print(f"Set non default threshold: {args.threshold}")
    for folder in args.folder:
        subdirs = os.listdir(folder)
        print(folder)
        for inst_id in tqdm(subdirs):
            if inst_id not in dataset_ids:
                continue
            patch = None
            pred_path = os.path.join(folder, inst_id, f"{inst_id}.pred")
            if not os.path.exists(pred_path):
                remove_ids.add(inst_id)
            else:
                try:
                    with open(pred_path, "r") as f:
                        patch = json.load(f)["model_patch"]
                except Exception as e:
                    remove_ids.add(inst_id)
            if patch:
                diff_stats = analyze_diff(patch)
                if args.filter_mode == "long_edit":
                    threshold = args.threshold or 40
                    if diff_stats["added_lines"] + diff_stats["deleted_lines"] > threshold:
                        remove_ids.add(inst_id)
                        # print(patch)
                        # time.sleep(10)
                else:
                    if diff_stats["added_lines"] + diff_stats["deleted_lines"] < 5:
                        remove_ids.add(inst_id)
elif args.filter_mode == "user_length":
    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    print("loaded tokenizer")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    TARGET_PREFIX = "OBSERVATION:\nThank you for your work on this issue."
    threshold = args.threshold or 600
    response_lengths = []
    for data in tqdm(loaded_dataset):
        cur_lengths = 0
        for msg in data["messages"]:
            if msg["role"] == "user":
                if TARGET_PREFIX in msg["content"]:
                    break
                cur_lengths += count_tokens(msg["content"])
            # elif msg["role"] == "assistant":
            #     cur_resp_lengths += count_tokens(msg["content"])
        if cur_lengths / (len(data["messages"]) // 2) > threshold:
            remove_ids.add(data["instance_id"])
else:
    raise RuntimeError()

print(f"Want to remove {len(remove_ids)} instances")

if args.filter_mode == "long_edit" or args.filter_mode == "soft_edit" or args.filter_mode == "user_length":
    new_fp = f"{front}_filter_{args.filter_mode}_{threshold}{tail}"
else:
    new_fp = f"{front}_filter_{args.filter_mode}{tail}"
kept_data = 0
assert not os.path.exists(new_fp)
with open(new_fp, "w") as f:
    for d in loaded_dataset:
        if d["instance_id"] not in remove_ids:
            f.write(json.dumps(d) + "\n")
            kept_data += 1
print(f"New dataset length: {kept_data}")
print(new_fp)