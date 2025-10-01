import os
import json
import torch
import importlib
from Bio import SeqIO
from tqdm import tqdm


def process_single_model(model_name, model_dir, saved_folder, config):
    """
    Processes a single protein language model to generate embeddings for all mutations.

    This function dynamically imports the specified model's generation function,
    processes all mutation sequences in the saved folder to generate embeddings,
    and saves the embeddings as PyTorch tensors. The last dim of embedding shape is recorded
    in the config for the first processed mutation.

    Args:
        model_name (str): Name identifier for the protein language model
        model_dir (str): Directory path containing the model's generation function
        saved_folder (str): Path to folder containing mutation sequence files
        config (dict): Configuration dictionary to store model metadata
    """
    module = importlib.import_module(f"{model_dir}.function")
    func_name = f"generate_{model_dir}"
    model_func = getattr(module, func_name)

    for i, mut_info in enumerate(tqdm(os.listdir(saved_folder), desc=model_name)):
        seq = str(list(SeqIO.parse(f"{saved_folder}/{mut_info}/result.fasta", "fasta"))[0].seq)
        embedding = model_func(seq)
        if i == 0:
            config["all_model"][model_name] = {"shape": list(embedding.shape)[-1]}
        torch.save(embedding, f"{saved_folder}/{mut_info}/{model_name}_embedding.pt")
        del embedding

    del model_func, module


def main():
    """
    Main function to process all protein language models and generate embeddings.

    This function discovers all available protein language model directories,
    processes each model to generate embeddings for all mutation sequences,
    and updates the configuration file with model metadata.
    """
    saved_folder = "../features"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with open("../config/config.json", "r") as f:
        config = json.load(f)

    if "all_model" not in config:
        config["all_model"] = {}

    config["all_model"].clear()

    model_dirs = [(item.replace("_embedding", ""), item) for item in os.listdir(".") if os.path.isdir(item) and item.endswith("_embedding")]
    print(f"Found models: {[name for name, _ in model_dirs]}")

    for model_name, model_dir in model_dirs:
        process_single_model(model_name, model_dir, saved_folder, config)

    with open("../config/config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
