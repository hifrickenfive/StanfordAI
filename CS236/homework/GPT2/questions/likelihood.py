import torch
import torch.nn as nn


def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar.
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        ##      Hint: Checkout Pytorch softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        ##                     Pytorch negative log-likelihood: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        ##                     Pytorch Cross-Entropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        ## Hint: Implementation should only takes 3~7 lines of code.

        logits, __ = model(text, past=None)
        probabilities = nn.functional.softmax(logits, dim=-1)
        log_likelihood = 0

        num_tokens = text.shape[1]
        for i in range(num_tokens - 1):
            next_token_ID = text[0, i + 1]  # what is the true next token?
            probability_next_token = probabilities[0, i, next_token_ID]  # get probability of the true next token, given the logits for the current token
            log_probability_next_token = torch.log(probability_next_token)
            log_likelihood += log_probability_next_token

        return log_likelihood
