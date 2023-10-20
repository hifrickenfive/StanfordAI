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

        # Why are we doing this? We're measuring how good the model is at predicting a ground truth input string.
        # Given a truth string, the model is trying to predict that same truth string in an autoregressive NLP manner.
        # That's why the logits.shape is [num_tokens, vocab] and text.shape is [num_tokens]

        # Get logits
        logits, __ = model(text, past=None)

        # Flatten the logits tensor from [1, text_length, vocab_size] to [text_length, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))

        # Flatten the input_text tensor from [1, text_length] to [text_length]
        text_flat = text.view(-1)

        # Cross_entropy performs softmax then negative log likelihood
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss = nn.functional.cross_entropy(logits_flat[1:, :], text_flat[1:], reduction='sum') # ignore first token in text and sum per Ed post Q6 Part 4 #134
        negative_log_likelihood = loss.item() # use loss.item() to return a scalar
        log_likelihood = - negative_log_likelihood

        return log_likelihood
