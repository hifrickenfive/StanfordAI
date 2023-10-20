import torch
import torch.nn.functional as F
from tqdm import trange


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(
        logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits
    )


def temperature_scale(
    logits, model, new_past, config, temperature, temperature_horizon
):
    if temperature is None:
        return logits

    if temperature_horizon == 1:
        ##TODO:
        ## Return logits scaled by the temperature parameter
        return logits / temperature
    elif temperature_horizon == 2:
        ## Compute the logits for all length-2 generations, and scale them by the temperature parameter
        ## Return the logits for the first generated token (by marginalizing out the second token)

        # joint_prob[i,j] will store the joint probability of the first generated token being first_tokens[i] and the second generated token being j
        first_tokens = []
        joint_probs = []
        return_logits = torch.ones((1, config.vocab_size)) * -1e10

        # TODO compute probability of first token. Missing TODO per Q6 Part 7 #80
        first_probs = F.softmax(logits, dim=-1)

        for t in range(config.vocab_size):
            if logits[0, t] <= -1e10:
                # to speed up computation, ignore first tokens that were filtered out by top-k
                continue
            first_prob = first_probs[0, t]
            first_tokens.append(t)
            new_current_text = torch.tensor([[t]])

            # TODO: compute the 1-D tensor joint_prob_t, where joint_prob_t[j] stores the joint probability of the first generated token being t and the second generated token being j
            # Don't forget to also do top-k filtering when computing probabilities for the second token
          
            # Compute logits for the second token given the first token
            second_logits, _ = model(new_current_text, past=new_past)
            second_logits = second_logits[:, -1, :]
            second_logits = top_k_logits(second_logits, k=config.top_k) # Apply masking
            second_probs = F.softmax(second_logits, dim=-1)

            # Compute the joint probability
            joint_prob_t = first_prob * second_probs[0]
            
            joint_probs.append(joint_prob_t)

        # convert to logits
        joint_probs = torch.cat(joint_probs, dim=0) 
        joint_logits = torch.log(joint_probs + 1e-10)

        # TODO: scale joint_logits by temperature, and compute first_logits by marginalizing out the second token dimension
        # Scale joint_logits by temperature
        joint_logits = joint_logits / temperature
        # Compute first_logits by marginalizing out the second token dimension
        joint_logits = joint_logits.view(-1, config.vocab_size) # reshape to 2D so I can use logsumexp
        # Marginalise second token dimension, which are columns dim=1
        # https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch-logsumexp
        first_logits = torch.logsumexp(joint_logits, dim=1) 

        return_logits[0, first_tokens] = first_logits
        return return_logits


def sample(model, start_text, config, length, temperature=None, temperature_horizon=1):
    current_text = start_text
    past = None
    output = [start_text]
    with torch.no_grad():
        for _ in trange(length):
            logits, new_past = model(current_text, past=past)
            # Input parameters:
            #     current_text: the encoded text token at t-1
            #     past: the calculated hidden state of previous text or None if no previous text given
            # Return:
            #     logits: a tensor of shape (batch_size, sequence_length, size_of_vocabulary)
            #     past: the calculated hidden state of previous + current text

            current_logits = logits[:, -1, :]
            logits = top_k_logits(current_logits, k=config.top_k)
            logits = temperature_scale(
                logits, model, new_past, config, temperature, temperature_horizon
            )

            ##TODO:
            ## 1) sample using the given `logits` tensor;
            ## 2) append the sample to the list `output`;
            ## 3) update `current_text` so that sampling can continue.
            ##    Hint: Checkout Pytorch softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
            ##                   Pytorch multinomial sampling: https://pytorch.org/docs/stable/generated/torch.multinomial.html
            ## Hint: Implementation should only takes 3~5 lines of code.
            ##       The text generated should look like a technical paper.
            probabilities = F.softmax(logits, dim=-1)
            tokenID = torch.multinomial(probabilities, 1)

            # We only sample input the next token given the previous hidden states
            # At first I thought we should concat current_text to start_text
            # But past is the hidden state matrix that represents the original string + new tokens i.e. t=0...t-1
            # This saves recomputing the hidden states
            current_text = tokenID
            output.append(tokenID)
            past = new_past

        output = torch.cat(output, dim=1)
        return output
