import json
import torch

with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


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
