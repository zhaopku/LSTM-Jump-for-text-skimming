"""Main model architecture for LSTM-Jump text skimming.

This implements a sequence classification model with learned skipping behavior.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import Tuple, Dict, Any, Optional
from jax_models.acl_cell import ACLSkipLSTMCell, ACLLSTMState


class SkimTextClassifier(nn.Module):
    """Text classification model with learned skimming.

    Attributes:
        vocab_size: Size of vocabulary
        embedding_size: Dimension of word embeddings
        hidden_size: LSTM hidden state dimension
        num_classes: Number of output classes
        max_steps: Maximum sequence length
        min_read: Minimum tokens to read before skipping
        max_skip: Maximum tokens to skip per jump
        max_jump: Maximum number of jumps allowed
        eps: Epsilon for epsilon-greedy exploration
        dropout_rate: Dropout probability
        use_pretrained_embeddings: Whether to use pretrained embeddings
    """
    vocab_size: int
    embedding_size: int = 300
    hidden_size: int = 200
    num_classes: int = 2
    max_steps: int = 50
    min_read: int = 8
    max_skip: int = 5
    max_jump: int = -1
    eps: float = 0.1
    dropout_rate: float = 0.0
    use_pretrained_embeddings: bool = False

    def setup(self):
        # Embedding layer
        if self.use_pretrained_embeddings:
            # Will be set via params during initialization
            self.embed = None
        else:
            self.embed = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.embedding_size,
                embedding_init=nn.initializers.xavier_uniform()
            )

        # ACL Skip LSTM Cell
        self.lstm_cell = ACLSkipLSTMCell(
            hidden_size=self.hidden_size,
            min_read=self.min_read,
            max_skip=self.max_skip,
            max_jump=self.max_jump,
            eps=self.eps
        )

        # Classification head
        self.classifier = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )

    def embed_inputs(self, input_ids: jnp.ndarray, embeddings: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Embed input token IDs.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            embeddings: Optional pretrained embeddings [vocab_size, embed_size]

        Returns:
            Embedded inputs [batch_size, seq_len, embed_size]
        """
        if self.use_pretrained_embeddings and embeddings is not None:
            # Use provided pretrained embeddings
            embedded = embeddings[input_ids]
        else:
            embedded = self.embed(input_ids)

        return embedded

    def process_sequence(
        self,
        embedded: jnp.ndarray,
        lengths: jnp.ndarray,
        is_training: bool = True,
        use_random: bool = False,
        rng: jax.random.PRNGKey = None
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Process sequence through ACL Skip LSTM.

        Args:
            embedded: Embedded inputs [batch_size, max_steps, embed_size]
            lengths: Actual sequence lengths [batch_size]
            is_training: Whether in training mode
            use_random: Whether to use random actions
            rng: Random key for sampling

        Returns:
            final_hidden: Final hidden state [batch_size, hidden_size]
            info: Dictionary with skip information
        """
        batch_size = embedded.shape[0]
        seq_len = embedded.shape[1]

        # Initialize state
        state = self.lstm_cell.initialize_state(batch_size)

        # Storage for skip information
        all_skips = []
        all_probs = []
        all_valid = []

        # Process sequence
        def scan_fn(carry, inputs):
            state, rng_key = carry
            x, step_idx = inputs

            # Split RNG for this step
            if rng_key is not None:
                rng_key, subkey = jax.random.split(rng_key)
            else:
                subkey = None

            # Process one step
            new_state, h_out = self.lstm_cell(
                carry=state,
                x=x,
                is_training=is_training,
                use_random=use_random,
                rng=subkey
            )

            # Collect skip information
            skip_info = {
                'skip_flag': (state.s > 0).astype(jnp.float32),
                'probs': new_state.probs,
                'valid': new_state.valid,
                'n_skip': new_state.n,
            }

            return (new_state, rng_key), (h_out, skip_info)

        # Prepare inputs for scan
        inputs_transposed = jnp.transpose(embedded, (1, 0, 2))  # [seq_len, batch_size, embed_size]
        step_indices = jnp.arange(seq_len)

        # Run scan
        (final_state, _), (all_h, all_skip_info) = jax.lax.scan(
            scan_fn,
            (state, rng),
            (inputs_transposed, step_indices)
        )

        # Extract final hidden state
        final_hidden = final_state.h

        # Process skip information
        skip_flags = all_skip_info['skip_flag']  # [seq_len, batch_size]
        skip_flags = jnp.transpose(skip_flags, (1, 0))  # [batch_size, seq_len]

        probs = all_skip_info['probs']  # [seq_len, batch_size]
        probs = jnp.transpose(probs, (1, 0))  # [batch_size, seq_len]

        valid = all_skip_info['valid']  # [seq_len, batch_size]
        valid = jnp.transpose(valid, (1, 0))  # [batch_size, seq_len]

        n_skips = all_skip_info['n_skip']  # [seq_len, batch_size]
        n_skips = jnp.transpose(n_skips, (1, 0))  # [batch_size, seq_len]

        # Mask out steps beyond sequence length
        mask = jnp.arange(seq_len)[None, :] < lengths[:, None]
        skip_flags = skip_flags * mask.astype(jnp.float32)
        probs = probs * mask.astype(jnp.float32)
        valid = valid * mask
        n_skips = n_skips * mask.astype(jnp.int32)

        # Compute skip statistics
        total_skips = jnp.sum(skip_flags, axis=1)  # [batch_size]
        skip_rate = total_skips / lengths.astype(jnp.float32)

        n_valid = jnp.sum(valid.astype(jnp.float32), axis=1)  # [batch_size]

        info = {
            'skip_flags': skip_flags,
            'probs': probs,
            'valid': valid,
            'n_skips': n_skips,
            'skip_rate': skip_rate,
            'n_valid': n_valid,
            'predicted_skips': n_skips
        }

        return final_hidden, info

    def __call__(
        self,
        input_ids: jnp.ndarray,
        lengths: jnp.ndarray,
        is_training: bool = True,
        use_random: bool = False,
        embeddings: Optional[jnp.ndarray] = None,
        rng: jax.random.PRNGKey = None
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch_size, max_steps]
            lengths: Actual sequence lengths [batch_size]
            is_training: Whether in training mode
            use_random: Whether to use random actions
            embeddings: Optional pretrained embeddings
            rng: Random key

        Returns:
            logits: Classification logits [batch_size, num_classes]
            info: Dictionary with skip information
        """
        # Embed inputs
        embedded = self.embed_inputs(input_ids, embeddings)

        # Apply dropout to embeddings
        if self.dropout_rate > 0:
            embedded = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(
                embedded
            )

        # Process sequence
        final_hidden, info = self.process_sequence(
            embedded, lengths, is_training, use_random, rng
        )

        # Classify
        logits = self.classifier(final_hidden)

        return logits, info


def compute_loss_and_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    skip_info: Dict[str, jnp.ndarray],
    sparse_coeff: float = 10.0
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Compute cross-entropy loss and metrics.

    Args:
        logits: Classification logits [batch_size, num_classes]
        labels: True labels [batch_size]
        skip_info: Dictionary with skip information
        sparse_coeff: Coefficient for sparsity reward

    Returns:
        loss: Scalar loss value
        metrics: Dictionary of metrics
    """
    # Cross-entropy loss
    ce_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )

    # Accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)

    # Skip statistics
    skip_rate = jnp.mean(skip_info['skip_rate'])

    metrics = {
        'ce_loss': ce_loss,
        'accuracy': accuracy,
        'skip_rate': skip_rate,
        'n_valid': jnp.mean(skip_info['n_valid']),
    }

    return ce_loss, metrics


def compute_rl_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    skip_info: Dict[str, jnp.ndarray],
    sparse_coeff: float = 10.0
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Compute policy gradient loss for RL training.

    This implements the REINFORCE algorithm with reward normalization.

    Args:
        logits: Classification logits [batch_size * n_samples, num_classes]
        labels: True labels [batch_size * n_samples]
        skip_info: Dictionary with skip information
        sparse_coeff: Coefficient for sparsity reward

    Returns:
        total_loss: Combined CE + PG loss
        metrics: Dictionary of metrics
    """
    # Cross-entropy loss (task loss)
    ce_loss_per_sample = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    # Compute rewards
    # Reward = -ce_loss + sparse_coeff * num_skips
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == labels).astype(jnp.float32)

    # Base reward: 1 for correct, -1 for incorrect
    base_reward = 2.0 * correct - 1.0

    # Sparse reward: encourage skipping
    total_predicted_skips = jnp.sum(skip_info['predicted_skips'], axis=1)
    sparse_reward = sparse_coeff * total_predicted_skips.astype(jnp.float32)

    # Total reward
    rewards = base_reward + sparse_reward

    # Normalize rewards (within batch)
    rewards_mean = jnp.mean(rewards)
    rewards_std = jnp.std(rewards) + 1e-8
    rewards_norm = (rewards - rewards_mean) / rewards_std

    # Compute policy gradient loss
    # PG loss = -sum(valid_steps) log(prob) * reward
    probs = skip_info['probs'] + 1e-8  # [batch_size, seq_len]
    valid = skip_info['valid'].astype(jnp.float32)  # [batch_size, seq_len]

    # Expand rewards to match sequence dimension
    rewards_expanded = rewards_norm[:, None]  # [batch_size, 1]
    rewards_tiled = jnp.tile(rewards_expanded, (1, probs.shape[1]))  # [batch_size, seq_len]

    # Mask by valid decisions
    rewards_masked = rewards_tiled * valid

    # Stop gradient on rewards
    rewards_masked = jax.lax.stop_gradient(rewards_masked)

    # Policy gradient loss
    pg_loss_per_step = rewards_masked * jnp.log(probs)

    # Average over valid steps for each sample
    n_valid = jnp.maximum(skip_info['n_valid'], 1.0)  # Avoid division by zero
    pg_loss_per_sample = jnp.sum(pg_loss_per_step, axis=1) / n_valid

    # Average over batch
    pg_loss = -jnp.mean(pg_loss_per_sample)

    # Total loss
    ce_loss = jnp.mean(ce_loss_per_sample)
    total_loss = ce_loss + pg_loss

    # Metrics
    accuracy = jnp.mean(correct)
    skip_rate = jnp.mean(skip_info['skip_rate'])

    metrics = {
        'total_loss': total_loss,
        'ce_loss': ce_loss,
        'pg_loss': pg_loss,
        'accuracy': accuracy,
        'skip_rate': skip_rate,
        'mean_reward': jnp.mean(rewards),
        'n_valid': jnp.mean(skip_info['n_valid']),
    }

    return total_loss, metrics
