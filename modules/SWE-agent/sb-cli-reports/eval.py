import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('-r', '--repo', default='django')
    return parser.parse_args()

args = get_args()
with open(args.file, "r") as f:
    results_json = json.load(f)

repo_name = args.repo
correct = [name for name in results_json["resolved_ids"] if repo_name in name]
missed = [name for name in results_json["unresolved_ids"] if repo_name in name]

print(f"{len(results_json["resolved_ids"]) / 500}")
print(f"{len(correct)} / {len(correct + missed)} = {len(correct) / len(correct + missed)}")



