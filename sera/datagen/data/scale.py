import argparse
import json
import os
import random
import yaml

from sera.utils import filter_messages


def create_start_fn_dataset(dataset):
    start_fn_dataset = {}
    for data in dataset:
        start_prompt = data["messages"][1]["content"]
        if start_prompt not in start_fn_dataset:
            start_fn_dataset[start_prompt] = []
        start_fn_dataset[start_prompt].append(data)
    return start_fn_dataset

def split_start_fn_dataset(start_fn_dataset, n_start_fn):
    assert isinstance(start_fn_dataset, dict)
    start_fn_dict = {}
    for start_fn, start_fn_inst in start_fn_dataset.items():
        # Get how many inst with this start fns
        cur_n_start_fn = min(len(start_fn_inst), n_start_fn)
        # Create entry if needed
        if cur_n_start_fn not in start_fn_dict:
            start_fn_dict[cur_n_start_fn] = {}
        # Add the instances to appropriate count dictionary
        start_fn_dict[cur_n_start_fn][start_fn] = random.sample(start_fn_inst, k=cur_n_start_fn)
    return start_fn_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', nargs="+")
    parser.add_argument('-t', '--type')
    parser.add_argument('-n', '--number', type=float)
    parser.add_argument('-nsf', '--n-start-fns', type=int)
    parser.add_argument('-i', '--instance_fp')
    parser.add_argument('-nf', '--no-filter', action='store_true')
    parser.add_argument('-th', '--threshold', type=float)
    parser.add_argument('-o', '--output-file')
    return parser.parse_args()

def scale_repos(dataset, number):
    new_dataset = []
    repo_to_data = {}
    for data in dataset:
        repo_name = data["instance_id"].split("_")[0]
        if repo_name not in repo_to_data:
            repo_to_data[repo_name] = []
        repo_to_data[repo_name].append(data)
    for repo_name in repo_to_data:
        print(f"{repo_name}: {len(repo_to_data[repo_name])}")
    repo_order = list(repo_to_data.keys())
    random.shuffle(repo_order)
    for repo_name in repo_order:
        print(f"Adding {len(repo_to_data[repo_name])} data from {repo_name}...")
        if len(new_dataset) < number:
            new_dataset += repo_to_data[repo_name]
        else:
            break
    return new_dataset

def scale_start_fns(dataset, number, n_start_fns):
    # How this works is I first create a dataset mapping start fns to all
    # instances beginning with that start fn.
    # Next, I filter this into dictionaries of total start fn count, up to n_start_fns.
    # So to create a dataset, we want the average # of inst/start fn to be as close to the
    # desired as possible.
    # Then, we randomly fill up the dataset.
    new_dataset = []
    start_fn_dataset = create_start_fn_dataset(dataset)
    start_fn_dict = split_start_fn_dataset(start_fn_dataset, n_start_fn=n_start_fns)
    for start_fn_ct in range(n_start_fns, -1, -1):
        print(f"Frequency {start_fn_ct}: {start_fn_ct*len(start_fn_dict[start_fn_ct])}")
        dataset_to_add = start_fn_dict.get(start_fn_ct, {})
        unique_starts = list(dataset_to_add.keys())
        random.shuffle(unique_starts)
        for start_fn in unique_starts:
            data_to_add = dataset_to_add[start_fn]
            new_dataset += data_to_add
            if len(new_dataset) >= number:
                return new_dataset
    raise RuntimeError(f"Unable to fill up dataset fully, reached {len(new_dataset)}/{number}")

def scale_bug_type(dataset, instance_fp, number, num_per_bug=4000):
    new_dataset = []
    instance_dict = {}
    bug_type_dict = {}
    with open(instance_fp, "r") as f:
        instances = yaml.safe_load(f)
        for inst in instances:
            instance_dict[inst["id"]] = inst
    for data in dataset:
        if data["instance_id"] not in instance_dict:
            continue
        cur_bug_type = instance_dict[data["instance_id"]]["extra_fields"]["bug_type"]
        if cur_bug_type not in bug_type_dict:
            bug_type_dict[cur_bug_type] = []
        bug_type_dict[cur_bug_type].append(data)
    bug_types = list(bug_type_dict.keys())
    random.shuffle(bug_types)
    for bug_type in bug_types:
        bug_type_inst = bug_type_dict[bug_type]
        print(f"Bug Type {bug_type} Count {len(bug_type_inst)}")
        # if "Logic Bug" in bug_type: # or "Interface Bugs" == bug_type:
        #     new_dataset += random.sample(bug_type_inst, k=7000)
        # print(f"Bug Type {bug_type} Count {len(bug_type_inst)}")
        if len(bug_type_inst) >= num_per_bug:
            new_dataset += random.sample(bug_type_inst, num_per_bug)
            print(f"\tAdding {num_per_bug} from {bug_type}")
        if len(new_dataset) >= number:
            return new_dataset
    return new_dataset

def scale_tokens(dataset, number, threshold=None):
    print("Threshold:", threshold)
    repo_to_data = {}
    _, token_to_data_tuples = filter_messages(dataset, truncate=True, return_token_to_data_tuples=True) # truncate=True must be set
    one_count = 0
    random.shuffle(token_to_data_tuples) # randomize because stable sorting
    for tup in token_to_data_tuples:
        if tup[0] == 1:
            one_count += 1
    number = min(number, len(token_to_data_tuples))
    print(f"Fully Included Data: {one_count}")
    sorted_token_to_tuples = sorted(token_to_data_tuples, key=lambda x: x[0], reverse=True)
    print(f"Proportion of Steps: {sorted_token_to_tuples[0][0]} to {sorted_token_to_tuples[number-1][0]}")
    if not threshold:
        new_dataset = [seq for _, seq in sorted_token_to_tuples][:number]
    else:
        print("Applying threshold to token scaling such that only data over the threshold is included")
        new_dataset = []
        for ratio, seq in sorted_token_to_tuples:
            if ratio >= threshold:
                new_dataset.append(seq)
        new_dataset = new_dataset[:number]
    return new_dataset

def main():
    args = get_args()
    total_ds = []
    for dataset_fp in args.dataset:
        with open(dataset_fp, "r") as f:
            ds = [json.loads(line) for line in f.readlines()]
            if not args.no_filter and not args.type == "tokens": 
                # If no filter, or if tokens (so filter later) then we skip this step
                ds = filter_messages(ds)
            total_ds += ds
    if args.number > len(total_ds):
        args.number = len(total_ds)
    elif args.number > 0 and args.number < 1:
        args.number = args.number * len(total_ds)
        print("setting number based on proportion")
    number = int(args.number)
    print(f"selecting {number} instances")
    scaled_ds = None
    random.shuffle(total_ds)
    if args.type == "repo" or args.type == "synth_pr_repo":
        scaled_ds = scale_repos(dataset=total_ds, number=number)
        number = len(scaled_ds)
        fp = f"{args.type}_{number}.jsonl"
    elif args.type == "start_fn":
        scaled_ds = scale_start_fns(dataset=total_ds, number=number, n_start_fns=args.n_start_fns)
        number = len(scaled_ds)
        fp = f"{args.type}_{number}_nsf{args.n_start_fns}.jsonl"
    elif args.type == "bug":
        scaled_ds = scale_bug_type(dataset=total_ds, instance_fp=args.instance_fp, number=number)
        number = len(scaled_ds)
        fp = f"{args.type}_{number}.jsonl"
    elif args.type == "tokens":
        scaled_ds = scale_tokens(dataset=total_ds, number=number, threshold=args.threshold)
        number = len(scaled_ds)
        fp = f"{args.type}_{number}.jsonl"
    elif args.type == "random":
        scaled_ds = random.sample(total_ds, k=number)
    if args.output_file:
        fp = f"{args.output_file}.jsonl"
    if scaled_ds:
        print("Number of instances:", len(scaled_ds))
        # original_ds_name = os.path.splitext(os.path.basename(args.dataset))[0]
        assert not os.path.exists(fp)
        with open(fp, "w") as f:
            for data in scaled_ds:
                f.write(json.dumps(data) + "\n")
        print(f'saved to {fp}')

main()