"""
The code is copied from my answers from ARENA: https://github.com/callummcdougall/ARENA_2.0
All credits to the original authors.
"""

import torch as t
from torch import Tensor
from typing import List, Tuple
import numpy as np

from transformers import GPT2Tokenizer

from gpt2_ai.train.model import GPT2



class TransformerSampler:

    def __init__(self, model: GPT2, tokenizer: GPT2Tokenizer):
        self.model = model
        self.cfg = model.config
        # self.config_trainer = model.config_trainer
        self.tokenizer = tokenizer
        # self.device = self.config_trainer.device
        self.device = "cuda" if t.cuda.is_available() else "cpu"

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> List:
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        '''
        # SOLUTION
        self.model.eval()
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.device)[0]

        for i in range(max_tokens_generated):
            # Get new logits (make sure we don't pass in more tokens than the model's context length)
            # TODO add message that input_ids were truncated
            logits = self.model(input_ids[None, -self.cfg.n_ctx:])
            logits = logits.logits
            # We only take logits for the last token, because this is what we're sampling
            logits = logits[0, -1]
            # Get next token (as a tensor of size (1, 1) so we can concat it to input_ids)
            next_token = self.sample_next_token(input_ids, logits, **kwargs)
            next_token = t.tensor([next_token], device=self.device)
            # Create new input ids string, with shape (1, old_seq_len + 1)
            input_ids = t.cat([input_ids, next_token], dim=-1)
            # Print out results, if required
            if verbose:
                print(self.tokenizer.decode(input_ids), end="\r")
            # If our new token was the end-of-text token, stop
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break

        return self.tokenizer.decode(input_ids)

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        verbose=False
    ) -> List[Tuple[float, t.Tensor]]:
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        '''
        pass


    def sample_next_token(
        self,
        input_ids: Tensor,
        logits: Tensor,
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None
    ):
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return self.greedy_search(logits)
        elif temperature != 1.0:
            logits = self.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = self.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if top_k > 0:
            return self.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return self.sample_top_p(logits, top_p)
        return self.sample_basic(logits)


    @staticmethod
    def greedy_search(logits: Tensor) -> int:
        '''
        Returns the most likely token (as an int).
        '''
        out = logits.argmax().item()
        return out


    @staticmethod
    def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
        '''
        Applies temperature scaling to the logits.
        '''
        # SOLUTION
        return logits / temperature



    @staticmethod
    def apply_frequency_penalty(
        input_ids: Tensor, logits: Tensor, freq_penalty: float) -> Tensor:
        '''
        Applies a frequency penalty to the logits.
        '''
        # SOLUTION
        d_vocab = logits.size(0)
        id_freqs = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * id_freqs


    def sample_basic(self, logits: Tensor) -> int:
        '''
        Samples from the distribution defined by the logits.
        '''
        # SOLUTION
        sampled_token = t.distributions.categorical.Categorical(logits=logits).sample()
        return sampled_token.item()


    def sample_top_k(self, logits: Tensor, k: int) -> int:
        '''
        Samples from the top k most likely tokens.
        '''
        # SOLUTION
        top_k_logits, top_k_token_ids = logits.topk(k)
        # Get sampled token (which is an index corresponding to the list of top-k tokens)
        sampled_token_idx = t.distributions.categorical.Categorical(logits=top_k_logits).sample()
        # Get the actual token id, as an int
        return top_k_token_ids[sampled_token_idx].item()



    # @staticmethod
    # def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
    #     '''
    #     Samples from the most likely tokens which make up at least p cumulative probability.
    #     '''
    #     # SOLUTION
    #     # Sort logits, and get cumulative probabilities
    #     logits_sorted, indices = logits.sort(descending=True, stable=True)
    #     cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
    #     # Choose which tokens to keep, in the set we sample from
    #     n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
    #     n_keep = max(n_keep, min_tokens_to_keep)
    #     keep_idx = indices[:n_keep]
    #     keep_logits = logits[keep_idx]
    #     # Perform the sampling
    #     sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
    #     return keep_idx[sample].item()