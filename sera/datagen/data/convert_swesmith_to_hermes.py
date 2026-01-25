import datasets
import json

from sera.utils import SYSTEM_SIMPLE, convert_xml_hermes

swesmith = datasets.load_dataset("SWE-bench/SWE-smith-trajectories")

converted_instances = []
for instance in swesmith:
    output_instance = {}
    output_instance["instance_id"] = instance["instance_id"]
    output_instance["messages"] = []
    for message in instance["messages"]:
        hermes_message = convert_xml_hermes(message["content"], message["role"], think=True)
        output_instance["messages"].append({"role": message["role"], "content": hermes_message})
    converted_instances.append(output_instance)
output_file = "swesmith_hermes.jsonl"
with open(output_file, "w") as f:
    for instance in converted_instances:
        f.write(json.dumps(instance) + "\n")