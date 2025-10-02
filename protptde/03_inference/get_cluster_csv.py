import json
import click
import pandas as pd


with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ensemble_size = config["ensemble_size"]


@click.command()
@click.option("--mut_counts", required=True, type=int)
def main(mut_counts):
    """
    Merges ensemble prediction CSVs, computes statistics,
    keeps the highest mean per mutation position set,
    sorts by Standard deviation, and saves to CSV.

    Args:
        mut_counts (int): Number of mutations per variant.
    """
    dfs = []
    for i in range(1, ensemble_size + 1):
        df = pd.read_csv(f"predicted_sorted_mut_counts_{mut_counts}_ensemble_{i}.csv")
        dfs.append(df.set_index("mut_name"))

    merged = pd.concat(dfs, axis=1)
    merged.columns = [f"pred_{i+1}" for i in range(ensemble_size)]
    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    merged["mean"] = merged[pred_cols].mean(axis=1)
    merged["std"] = merged[pred_cols].std(axis=1)

    merged["positions"] = merged.index.map(lambda mut_name: tuple(sorted([int(x[1:-1]) for x in mut_name.split(",")])))
    merged = merged.reset_index()
    merged = merged.loc[merged.groupby("positions")["mean"].idxmax()]

    merged = merged.sort_values("std", ascending=True)
    merged = merged.drop(columns=["positions"])
    merged.to_csv(f"predicted_sorted_mut_counts_{mut_counts}_ensemble.csv", index=False)


if __name__ == "__main__":
    main()
