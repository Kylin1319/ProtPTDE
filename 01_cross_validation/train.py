import os
import json
import click
import torch
import optuna
import itertools
import numpy as np
import pandas as pd
from optuna.samplers import GridSampler
from model import BatchData, ModelUnion, to_gpu, spearman_loss, spearman_corr
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def stratified_sampling_for_mutation_data(mut_info_list):
    """
    Creates stratified sampling data for mutation analysis by extracting positions and generating binary vectors.

    This function processes mutation information to extract mutation positions,creates
    an index mapping, and generates binary vectors indicating which positions are
    mutated for each mutation combination.

    Args:
        mut_info_list (list): List of mutation info strings, where each string contains
                            single or multiple mutations (e.g., "A123B" or "A123B,C456D")

    Returns:
        tuple: A 3-tuple containing:
            - sorted_mut_positions (list): Sorted list of mutation positions
            - index_map (dict): Mapping from mutation position to vector index
            - vectors_dict (dict): Mapping from mutation info to binary vector
    """
    # extract unique mutation positions
    positions = set()
    for multiple_mut_info in mut_info_list:
        for single_mut_info in multiple_mut_info.split(","):
            positions.add(int(single_mut_info[1:-1]))
    sorted_mut_positions = sorted(positions)
    index_map = {mut_pos: index for index, mut_pos in enumerate(sorted_mut_positions)}

    # generate mutation vectors
    vectors_dict = {}
    for multiple_mut_info in mut_info_list:
        vec = [0] * len(index_map)
        for single_mut_info in multiple_mut_info.split(","):
            vec[index_map[int(single_mut_info[1:-1])]] = 1
        vectors_dict[multiple_mut_info] = vec

    return sorted_mut_positions, index_map, vectors_dict


def objective(trial, random_seed):
    """
    Optuna objective function for hyperparameter optimization with cross-validation.

    This function performs hyperparameter search using Optuna trials, including model
    combination selection, layer count, and maximum learning rate optimization. It uses
    multilabel stratified cross-validation to evaluate model performance and returns
    a score based on mean test correlation minus standard deviation for robust optimization.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter suggestion
        random_seed (int): Random seed for reproducible results

    Returns:
        float: Optimization score (mean test correlation - std deviation)
    """
    basic_data_name = config["basic_data_name"]
    model_number = config["cross_validation"]["model_number"]
    training_parameter = config["cross_validation"]["training_parameter"]
    hyperparameter_search = config["cross_validation"]["hyperparameter_search"]

    all_models = sorted(list(config["all_model"].keys()))
    model_combinations = [",".join(combo) for combo in itertools.combinations(all_models, model_number)]

    selected_models = trial.suggest_categorical("model_combination", model_combinations).split(",")
    num_layer = trial.suggest_int("num_layer", hyperparameter_search["num_layer"]["min"], hyperparameter_search["num_layer"]["max"])
    max_lr = trial.suggest_categorical("max_lr", hyperparameter_search["max_lr"]["choices"])

    device = training_parameter["device"]
    min_lr = training_parameter["min_lr"]
    initial_lr = training_parameter["initial_lr"]
    total_epochs = training_parameter["total_epochs"]
    warmup_epochs = int(training_parameter["warmup_epochs_ratio"] * total_epochs)
    batch_size = training_parameter["batch_size"]
    test_size = training_parameter["test_size"]
    k_fold = training_parameter["k_fold"]
    cv_shuffle = training_parameter["shuffle"]

    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True

    all_csv = pd.read_csv(f"../data/{basic_data_name}.csv", index_col=0)

    # generate position mapping and mutation vectors
    mut_info_list = all_csv.index.tolist()
    sorted_mut_positions, index_map, vectors = stratified_sampling_for_mutation_data(mut_info_list)
    # print(f"Sorted positions: {sorted_mut_positions}")
    # print(f"Index map: {index_map}")
    y_mut_pos = np.array([vectors[multiple_mut_info] for multiple_mut_info in mut_info_list])

    # test data
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    for train_validation_index, test_index in msss.split(y_mut_pos, y_mut_pos):
        train_validation_csv = all_csv.iloc[train_validation_index].copy()
        test_csv = all_csv.iloc[test_index].copy()

    models_name = ",".join(selected_models)
    file = f"results/{models_name}/num_layer_{num_layer}_max_lr_{max_lr}_random_seed_{random_seed}"
    os.makedirs(file, exist_ok=True)
    k_fold_test_corr = []

    # k-fold cross-validation
    mskf = MultilabelStratifiedKFold(n_splits=k_fold, shuffle=cv_shuffle, random_state=random_seed)
    for k_fold_index, (train_index, validation_index) in enumerate(mskf.split(y_mut_pos[train_validation_index], y_mut_pos[train_validation_index])):
        train_csv = train_validation_csv.iloc[train_index].copy()
        validation_csv = train_validation_csv.iloc[validation_index].copy()

        train_dataset = BatchData(train_csv, selected_models)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        validation_dataset = BatchData(validation_csv, selected_models)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = BatchData(test_csv, selected_models)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = ModelUnion(num_layer, selected_models).to(device)

        # fixed parameters
        for name, param in model.named_parameters():
            if "finetune_coef" in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=initial_lr)

        # scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs) * (max_lr / initial_lr))
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, min_lr=min_lr)

        best_loss = float("inf")
        best_corr = float("-inf")
        loss = pd.DataFrame()

        for epoch in range(total_epochs):

            # train
            model.train()
            epoch_loss = 0
            for wt_data, mut_data, label in train_loader:
                wt_data, mut_data, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(label, device)
                optimizer.zero_grad()

                pred = model(wt_data, mut_data)
                train_loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, "kl")
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
                optimizer.step()

                epoch_loss += train_loss.item()

            train_loss = epoch_loss / len(train_loader)

            # validation
            model.eval()
            preds = []
            trues = []
            with torch.no_grad():
                for wt_data, mut_data, label in validation_loader:
                    wt_data, mut_data, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(label, device)
                    pred = model(wt_data, mut_data)
                    preds += pred.detach().cpu().tolist()
                    trues += label.detach().cpu().tolist()

            validation_loss = -spearman_corr(torch.tensor(preds), torch.tensor(trues)).item()

            # test
            preds = []
            trues = []
            with torch.no_grad():
                for wt_data, mut_data, label in test_loader:
                    wt_data, mut_data, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(label, device)
                    pred = model(wt_data, mut_data)
                    preds += pred.detach().cpu().tolist()
                    trues += label.detach().cpu().tolist()

            test_corr = spearman_corr(torch.tensor(preds), torch.tensor(trues)).item()

            # save results
            loss.loc[f"{epoch}", "train_loss"] = train_loss
            loss.loc[f"{epoch}", "validation_loss"] = validation_loss
            loss.loc[f"{epoch}", "test_corr"] = test_corr
            loss.loc[f"{epoch}", "learning_rate"] = optimizer.param_groups[0]["lr"]
            loss.to_csv(f"{file}/k_fold_index-{k_fold_index}_loss.csv")

            # scheduler
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(validation_loss)

            # check if validation loss improved
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_corr = test_corr
            elif optimizer.param_groups[0]["lr"] <= min_lr and epoch > warmup_epochs:
                print(f"Stopping at epoch {epoch} due to no improvement in validation loss.")
                break
        k_fold_test_corr.append(best_corr)

    return float(pd.Series(k_fold_test_corr).mean()) - float(pd.Series(k_fold_test_corr).std())


@click.command()
@click.option("--random_seed", type=int, required=True)
def main(random_seed):
    """
    Main function to run hyperparameter optimization using Optuna with grid search.

    This function sets up the search space for model combinations, layer counts, and
    maximum learning rates, then performs grid search optimization using Optuna.
    Results are saved to CSV files for analysis.

    Args:
        random_seed (int): Random seed for reproducible optimization results
    """
    model_number = config["cross_validation"]["model_number"]
    hyperparameter_search = config["cross_validation"]["hyperparameter_search"]

    all_models = sorted(list(config["all_model"].keys()))
    model_combinations = [",".join(combo) for combo in itertools.combinations(all_models, model_number)]
    search_space = {"model_combination": model_combinations, "num_layer": list(range(hyperparameter_search["num_layer"]["min"], hyperparameter_search["num_layer"]["max"] + 1)), "max_lr": hyperparameter_search["max_lr"]["choices"]}

    study = optuna.create_study(direction="maximize", sampler=GridSampler(search_space))
    study.optimize(lambda trial: objective(trial, random_seed), n_jobs=1)
    os.makedirs("best_score_results", exist_ok=True)
    study.trials_dataframe().to_csv(f"best_score_results/random_seed_{random_seed}.csv", index=False)


if __name__ == "__main__":
    main()
