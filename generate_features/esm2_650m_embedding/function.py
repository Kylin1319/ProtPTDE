import torch

_esm2_model_cache = None


def get_esm2_model():
    """
    Loads and caches the ESM2 protein language model for sequence embedding generation.

    This function implements a singleton pattern to load the ESM2 650M parameter model
    only once and cache it globally. Subsequent calls return the cached model and
    batch converter to avoid redundant loading and improve performance.

    Returns:
        tuple: A tuple containing (esm2_model, batch_converter) where:
            - esm2_model: The loaded ESM2 model in evaluation mode on GPU
            - batch_converter: Alphabet batch converter for sequence preprocessing
    """
    global _esm2_model_cache

    if _esm2_model_cache is None:

        esm2_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        batch_converter = alphabet.get_batch_converter()
        esm2_model = esm2_model.eval().cuda()

        _esm2_model_cache = (esm2_model, batch_converter)

    return _esm2_model_cache


def generate_esm2_650m_embedding(seq):
    """
    Generates protein sequence embeddings using the ESM2 650M parameter model.

    This function takes a protein sequence and generates its embedding representation
    using the pre-trained ESM2 model. The embedding is extracted from the final layer
    (layer 33) and excludes special tokens (start/end tokens) to return only the
    sequence-specific representations.

    Args:
        seq (str): Protein sequence string to generate embeddings for

    Returns:
        torch.Tensor: Protein sequence embedding tensor of shape [seq_len, hidden_dim]
                     where seq_len is the length of the input sequence and hidden_dim
                     is the ESM2 650M embedding dimension
    """
    with torch.no_grad():
        esm2_model, batch_converter = get_esm2_model()

        _, _, batch_tokens = batch_converter([("", seq)])
        embedding = esm2_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)["representations"][33][0, 1:-1, :].detach().cpu().clone()

        return embedding
