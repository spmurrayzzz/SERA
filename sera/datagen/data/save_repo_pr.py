import datasets
import json

target_repo = ""
dataset = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
repo_to_pr = {}
prs = {}
for instance in dataset:
    # repo_name = instance["instance_id"].split("__")[0]
    # if target_repo not in repo_name:
    #     continue
    prs[instance["instance_id"]] = instance["problem_statement"]
print(len(prs))
with open(f"swebench_all_prs.json", "w") as f:
    json.dump(prs, f, indent=4)