import torch
from transformers import AutoModelForMaskedLM

_esmc_300m_model_cache = None


def get_esmc_300m_model():
    """
    Loads and caches the ESMC 300M protein language model for sequence embedding generation.

    This function implements a singleton pattern to load the ESMC 300M parameter model
    only once and cache it globally. The model is loaded from Hugging Face with float32
    precision and moved to GPU for inference. Subsequent calls return the cached model
    and tokenizer to avoid redundant loading.

    Returns:
        tuple: A tuple containing (esmc_model, tokenizer) where:
            - esmc_model: The loaded ESMC model in evaluation mode on GPU
            - tokenizer: The model's tokenizer for sequence preprocessing
    """
    global _esmc_300m_model_cache

    if _esmc_300m_model_cache is None:

        esmc_model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_small", torch_dtype=torch.float32, trust_remote_code=True).eval().cuda()
        tokenizer = esmc_model.tokenizer

        _esmc_300m_model_cache = (esmc_model, tokenizer)

    return _esmc_300m_model_cache


def generate_esmc_300m_embedding(seq):
    """
    Generates protein sequence embeddings using the ESMC 300M parameter model.

    This function takes a protein sequence and generates its embedding representation
    using the pre-trained ESMC model. The sequence is tokenized, processed through
    the model, and the last hidden state is extracted. Special tokens (start/end)
    are excluded to return only sequence-specific representations.

    Args:
        seq (str): Protein sequence string to generate embeddings for

    Returns:
        torch.Tensor: Protein sequence embedding tensor of shape [seq_len, hidden_dim]
                     where seq_len is the length of the input sequence and hidden_dim
                     is the ESMC 300M embedding dimension
    """
    with torch.no_grad():

        esmc_model, tokenizer = get_esmc_300m_model()

        tokenized = tokenizer([seq], padding=True, return_tensors="pt")
        tokenized = {key: value.cuda() for key, value in tokenized.items()}
        embedding = esmc_model(**tokenized).last_hidden_state[0, 1:-1, :].detach().cpu().clone()

        return embedding
