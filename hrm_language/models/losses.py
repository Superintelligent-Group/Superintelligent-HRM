import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

# A sentinel value for labels that should be ignored during loss calculation.
# This is a standard practice in language model training.
IGNORE_LABEL_ID = -100

class SoftmaxCrossEntropyLoss(nn.Module):
    """
    A simplified loss head for our language model. It wraps the core model,
    computes the cross-entropy loss, and calculates accuracy.
    """
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__()
        self.model = model

    def forward(self, carry: Any, batch: Dict[str, torch.Tensor]) -> Tuple[Any, torch.Tensor, Dict[str, Any]]:
        """
        The forward pass for training. It receives the recurrent state (carry)
        and the data batch, passes them to the core model, and computes the loss.
        """
        # Pass the carry and batch to the underlying HierarchicalReasoningModel
        new_carry, outputs = self.model(carry, batch)

        # The model outputs a dictionary. The logits are under the key "logits".
        logits = outputs["logits"]
        labels = batch["labels"]

        # Reshape for cross_entropy
        # The logits are (batch, seq_len, vocab_size) and labels are (batch, seq_len)
        # We need to flatten them to (batch * seq_len, vocab_size) and (batch * seq_len)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        
        # Compute loss, ignoring the padding token
        loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_LABEL_ID)
        
        # Compute metrics for logging
        with torch.no_grad():
            # Calculate accuracy only on non-ignored tokens
            valid_mask = flat_labels != IGNORE_LABEL_ID
            predicted_tokens = flat_logits.argmax(-1)
            accuracy = (predicted_tokens[valid_mask] == flat_labels[valid_mask]).float().mean()
            
            metrics = {
                "loss": loss.detach(),
                "accuracy": accuracy,
                "count": valid_mask.sum(), # The number of valid tokens used for the loss
            }

        return new_carry, loss, metrics
