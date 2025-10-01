import json
import torch
import soft_rank_pytorch
from Bio import SeqIO

with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


#######################################################################
# soft spearman loss
#######################################################################


def spearman_loss(pred, true, regularization_strength, regularization):
    """
    Computes negative Spearman correlation as loss for ranking tasks.

    Args:
        pred (torch.Tensor): Predicted values, shape (1, N)
        true (torch.Tensor): Ground truth values, shape (1, N)
        regularization_strength (float): Strength parameter for soft ranking
        regularization (str): Regularization type for soft ranking

    Returns:
        torch.Tensor: Negative Spearman correlation coefficient
    """
    assert pred.device == true.device
    assert pred.shape == true.shape
    assert pred.shape[0] == 1
    assert pred.ndim == 2

    device = pred.device

    soft_pred = soft_rank_pytorch.soft_rank(pred.cpu(), regularization_strength=regularization_strength, regularization=regularization).to(device)
    soft_true = _rank_data(true.squeeze(0)).to(device)
    preds_diff = soft_pred - soft_pred.mean()
    target_diff = soft_true - soft_true.mean()

    cov = (preds_diff * target_diff).mean()
    preds_std = torch.sqrt((preds_diff * preds_diff).mean())
    target_std = torch.sqrt((target_diff * target_diff).mean())

    spearman_corr = cov / (preds_std * target_std + 1e-6)
    return -spearman_corr


#######################################################################
# spearman corr
#######################################################################


def _find_repeats(data):
    """
    Finds elements that appear more than once in the tensor.

    Args:
        data (torch.Tensor): Input tensor to analyze

    Returns:
        torch.Tensor: Unique values that appear at least twice
    """
    temp = data.detach().clone()
    temp = temp.sort()[0]

    change = torch.cat([torch.tensor([True], device=temp.device), temp[1:] != temp[:-1]])
    unique = temp[change]
    change_idx = torch.cat([torch.nonzero(change), torch.tensor([[temp.numel()]], device=temp.device)]).flatten()
    freq = change_idx[1:] - change_idx[:-1]
    atleast2 = freq > 1
    return unique[atleast2]


def _rank_data(data):
    """
    Computes ranks of data elements with tied values handled by averaging.

    Args:
        data (torch.Tensor): Input tensor to rank

    Returns:
        torch.Tensor: Ranked data with same shape as input
    """
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)

    repeats = _find_repeats(data)
    for r in repeats:
        condition = data == r
        rank[condition] = rank[condition].mean()
    return rank


def spearman_corr(pred, true):
    """
    Computes Spearman correlation coefficient between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted values, 1D or 2D tensor
        true (torch.Tensor): Ground truth values, same shape as pred

    Returns:
        torch.Tensor: Spearman correlation coefficient(s), clamped to [-1, 1]
    """
    assert pred.dtype == true.dtype
    assert pred.ndim <= 2 and true.ndim <= 2

    if pred.ndim == 1:
        pred = _rank_data(pred)
        true = _rank_data(true)
    else:
        pred = torch.stack([_rank_data(p) for p in pred.T]).T
        true = torch.stack([_rank_data(t) for t in true.T]).T

    preds_diff = pred - pred.mean(0)
    target_diff = true - true.mean(0)

    cov = (preds_diff * target_diff).mean(0)
    preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
    target_std = torch.sqrt((target_diff * target_diff).mean(0))

    spearman_corr = cov / (preds_std * target_std + 1e-6)
    return torch.clamp(spearman_corr, -1.0, 1.0)


#######################################################################
# get mut pos
#######################################################################


def get_mutation_positions_from_sequences(wt_seq, mut_seq):
    """
    Identifies mutation positions by comparing wild-type and mutant sequences.

    Args:
        wt_seq (list or str): Wild-type sequence(s)
        mut_seq (list or str): Mutant sequence(s)

    Returns:
        torch.Tensor: Binary tensor marking mutation positions (1=mutated, 0=unchanged)
    """
    if not isinstance(wt_seq, list):
        wt_seq = [wt_seq]
        mut_seq = [mut_seq]
    mut_pos_list = []
    for wt, mut in zip(wt_seq, mut_seq):
        assert len(wt) == len(mut), f"Sequence length mismatch: WT={len(wt)}, MUT={len(mut)}"
        mut_pos = torch.zeros(len(wt), dtype=torch.int)
        for i, (wt_aa, mut_aa) in enumerate(zip(wt, mut)):
            if wt_aa != mut_aa:
                mut_pos[i] = 1
        mut_pos_list.append(mut_pos)
    return torch.stack(mut_pos_list)


#######################################################################
# data
#######################################################################


class BatchData(torch.utils.data.Dataset):
    """
    A PyTorch dataset for loading protein mutation data with embeddings.

    This dataset loads wild-type and mutant protein sequences along with their
    pre-computed embeddings from multiple models.

    Args:
        csv (pd.DataFrame): DataFrame containing mutation information and labels
        selected_models (list): List of pre-trained model names to use

    Attributes:
        csv (pd.DataFrame): DataFrame containing mutation information and labels
        selected_models (list): List of pre-trained model names to use
        wt_seq (str): Wild-type protein sequence
    """

    def __init__(self, csv, selected_models):
        """
        Initializes the dataset with mutation data and model selection.

        Args:
            csv (pd.DataFrame): DataFrame containing mutation information and labels
            selected_models (list): List of pre-trained model names to use
        """
        self.csv = csv
        self.selected_models = selected_models
        self.wt_seq = str(list(SeqIO.parse("../features/wt/result.fasta", "fasta"))[0].seq)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.csv)

    def __getitem__(self, index):
        """
        Retrieves wild-type and mutant data with embeddings for specified index.

        Args:
            index (int): Dataset index

        Returns:
            tuple: (wt_data, mut_data, label) where data dicts contain sequences and model embeddings
        """
        mut_info = self.csv.iloc[index].name
        mut_seq = str(list(SeqIO.parse(f"../features/{mut_info}/result.fasta", "fasta"))[0].seq)

        wt_data = {"seq": self.wt_seq}
        mut_data = {"seq": mut_seq}
        for model_name in self.selected_models:
            wt_data[f"{model_name}_embedding"] = torch.load(f"../features/wt/{model_name}_embedding.pt")
            mut_data[f"{model_name}_embedding"] = torch.load(f"../features/{mut_info}/{model_name}_embedding.pt")

        return wt_data, mut_data, torch.tensor(self.csv.loc[mut_info, "label"]).to(torch.float32)


def to_gpu(obj, device):
    """
    Recursively moves tensors in nested data structures to specified device.

    Args:
        obj: Input object (tensor, list, tuple, dict, or other)
        device: Target device for tensor placement

    Returns:
        Same type as input with tensors moved to device
    """
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [to_gpu(i, device=device) for i in obj]
    elif isinstance(obj, tuple):
        return (to_gpu(i, device=device) for i in obj)
    elif isinstance(obj, dict):
        return {i: to_gpu(j, device=device) for i, j in obj.items()}
    else:
        return obj


#######################################################################
# model
#######################################################################


class DownStreamModel(torch.nn.Module):
    """
    A downstream neural network model for protein function prediction.

    This model processes embeddings from multiple pre-trained protein models,
    transforms them to a common dimension, concatenates them, and passes through
    fully connected layers for final prediction.

    Args:
        num_layer (int): Number of hidden layers in the readout network
        selected_models (list): List of pre-trained model names to use

    Attributes:
        selected_models (list): List of pre-trained model names to use
        model_transforms (torch.nn.ModuleDict): Transform layers for each model
        read_out (torch.nn.Sequential): Final prediction layers
    """

    def __init__(self, num_layer, selected_models):
        """
        Initializes the downstream model with embedding transforms and readout layers.

        Args:
            num_layer (int): Number of hidden layers in readout network
            selected_models (list): List of pre-trained model names to use
        """
        super().__init__()

        self.config = config
        self.selected_models = selected_models
        self.model_transforms = torch.nn.ModuleDict()

        embedding_output_dim = self.config["single_model_embedding_output_dim"]
        for model_name in self.selected_models:
            input_dim = self.config["all_model"][model_name]["shape"]

            self.model_transforms[model_name] = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, embedding_output_dim), torch.nn.LeakyReLU())

        total_output_dim = len(self.selected_models) * embedding_output_dim

        layers = []
        input_dim = total_output_dim
        for _ in range(num_layer):
            layers.append(torch.nn.Linear(input_dim, 64))
            layers.append(torch.nn.LeakyReLU())
            input_dim = 64

        layers.append(torch.nn.Linear(64, 1))
        self.read_out = torch.nn.Sequential(*layers)

    def forward(self, embeddings_dict):
        """
        Forward pass through the model.

        Args:
            embeddings_dict (dict): Dictionary containing model embeddings

        Returns:
            torch.Tensor: Model predictions with shape (batch_size, 1)
        """
        transformed_embeddings = []
        for model_name in self.selected_models:
            embedding = embeddings_dict[f"{model_name}_embedding"]
            transformed = self.model_transforms[model_name](embedding)
            transformed_embeddings.append(transformed)

        x = torch.cat(transformed_embeddings, dim=-1)
        x = self.read_out(x)
        return x


class ModelUnion(torch.nn.Module):
    """
    A union model that computes mutation effects by comparing wild-type and mutant predictions.

    This model uses a downstream prediction model to compute values for both wild-type
    and mutant sequences, then calculates the delta (difference) weighted by mutation
    positions to predict the functional impact of mutations.

    Args:
        num_layer (int): Number of hidden layers for the downstream model
        selected_models (list): List of pre-trained model names to use

    Attributes:
        down_stream_model (DownStreamModel): Model for computing sequence predictions
        finetune_coef (torch.nn.Parameter): Scaling coefficient for delta values
    """

    def __init__(self, num_layer, selected_models):
        """
        Initializes the model union with downstream model and fine-tuning coefficient.

        Args:
            num_layer (int): Number of hidden layers in downstream model
            selected_models (list): List of pre-trained model names to use
        """
        super().__init__()
        self.down_stream_model = DownStreamModel(num_layer, selected_models)
        self.finetune_coef = torch.nn.Parameter(torch.tensor([1.0], requires_grad=False))

    def forward(self, wt_data, mut_data):
        """
        Forward pass calculates the delta (difference) weighted by mutation positions to predict the functional impact of mutations

        Args:
            wt_data (dict): Wild-type data containing sequence and embeddings
            mut_data (dict): Mutant data containing sequence and embeddings

        Returns:
            torch.Tensor: Predicted mutation effects with shape (batch_size,)
        """
        # downstream model and calculate delta value
        wt_embeddings = {key: emb for key, emb in wt_data.items() if key.endswith("_embedding")}
        mut_embeddings = {key: emb for key, emb in mut_data.items() if key.endswith("_embedding")}

        wt_value = self.down_stream_model(wt_embeddings)
        mut_value = self.down_stream_model(mut_embeddings)
        mut_pos = get_mutation_positions_from_sequences(wt_data["seq"], mut_data["seq"])
        device = mut_value.device
        mut_pos = mut_pos.to(device)

        delta_value = (mut_value - wt_value).squeeze(-1) * mut_pos
        delta_value = delta_value.sum(1)
        return self.finetune_coef * delta_value
