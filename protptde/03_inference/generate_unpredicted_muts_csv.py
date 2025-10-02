import json
import click
import random
import operator
import functools
import itertools
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


# estimate nosorted_result_list length
def estimate_total_combinations(tmp_dict, mut_counts):
    """
    Estimates the total number of mutation combinations for a given mutation count.

    This function calculates the total number of possible mutation combinations
    by selecting position combinations and multiplying the number of possible
    mutations at each selected position.

    Args:
        tmp_dict (dict): Dictionary mapping positions to lists of possible mutations
        mut_counts (int): Number of mutation positions to combine

    Returns:
        int: Estimated total number of mutation combinations
    """
    keys = list(tmp_dict.keys())
    est_total = 0
    for pos_comb in itertools.combinations(keys, mut_counts):
        counts = [len(tmp_dict[pos]) for pos in pos_comb]
        est_total += functools.reduce(operator.mul, counts, 1)
    return est_total


@click.command()
@click.option("--mut_counts", required=True, type=int)
def main(mut_counts):
    """
    Main function to generate mutation combinations and create sequence data for inference.

    This function generates all possible or randomly sampled mutation combinations
    based on the specified mutation count. It creates wild-type and mutant sequences
    for each combination and saves the results to a CSV file for further analysis.

    Args:
        mut_counts (int): Number of mutations to include in each combination
    """
    basic_data_name = config["basic_data_name"]
    max_mutations = config["inference"]["max_mutations"]
    wt_seq = str(list(SeqIO.parse("../features/wt/result.fasta", "fasta"))[0].seq)
    data = pd.read_csv(f"../data/{basic_data_name}.csv", index_col=0)
    tmp = []
    for i in data.index.str.split(","):
        tmp += i

    # get dict (key: mut_pos, value: [wt_res + mut_pos + mut_res])
    train_tmp_dict = {}
    for value in list(set(tmp)):
        mut_pos = int(value[1:-1])
        train_tmp_dict.setdefault(mut_pos, []).append(value)

    tmp_dict = train_tmp_dict

    total_comb_count = estimate_total_combinations(tmp_dict, mut_counts)
    print(f"Estimated total mutation combinations: {total_comb_count}")

    if total_comb_count <= max_mutations:
        print("Generating all possible combinations...")
        nosorted_result_list = []
        for multi_mut_pos in itertools.combinations(tmp_dict, mut_counts):
            tmp_list = [tmp_dict[mut_pos] for mut_pos in multi_mut_pos]
            nosorted_result_list += list(itertools.product(*tmp_list))
    else:
        print(f"Too many combinations ({total_comb_count}), using random sampling.")
        unique_combinations = set()
        keys = list(tmp_dict.keys())

        with tqdm(total=max_mutations, desc="Sampling") as pbar:
            while len(unique_combinations) < max_mutations:
                sampled_positions = random.sample(keys, mut_counts)
                sampled_positions.sort()
                sampled_muts = tuple(random.choice(tmp_dict[pos]) for pos in sampled_positions)
                if sampled_muts not in unique_combinations:
                    unique_combinations.add(sampled_muts)
                    pbar.update(1)

        nosorted_result_list = list(unique_combinations)

    # sort mutations
    sorted_result_list = []
    for i in nosorted_result_list:
        tmp = sorted(i, key=lambda x: int(x[1:-1]))
        sorted_result_list.append(",".join(tmp))

    # generate data csv
    data = pd.DataFrame(columns=["wt_seq", "mut_seq", "pred"], index=sorted_result_list)
    data["wt_seq"] = wt_seq
    for multi_mut_info in tqdm(sorted_result_list):
        mut_seq = list(wt_seq)
        for single_mut_info in multi_mut_info.split(","):
            mut_seq[int(single_mut_info[1:-1])] = single_mut_info[-1]
        data.loc[multi_mut_info, "mut_seq"] = "".join(mut_seq)

    data.index.name = "mut_name"
    data.to_csv(f"sorted_mut_counts_{mut_counts}.csv")


if __name__ == "__main__":
    main()
