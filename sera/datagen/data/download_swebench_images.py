import datasets
import json

swebench = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

image_dict = {}

for data in swebench:
    inst_id = data["instance_id"]
    base_commit = data["base_commit"]
    id_1, id_2 = inst_id.split("__")
    image_name = f"swebench/sweb.eval.x86_64.{id_1}_1776_{id_2}"
    image_dict[inst_id] = {"base_commit": base_commit, "image_name": image_name}

with open("metadata/swebench_docker_images.json", "w") as f:
    json.dump(image_dict, f, indent=4)