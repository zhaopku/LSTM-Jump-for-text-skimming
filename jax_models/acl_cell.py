"""ACL Skip LSTM Cell implementation in Flax.

This implements an LSTM cell with learned skipping behavior for text skimming.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, NamedTuple
from functools import partial


class ACLLSTMState(NamedTuple):
    """State tuple for ACL Skip LSTM.

    Attributes:
        c: Cell state [batch_size, hidden_size]
        h: Hidden state [batch_size, hidden_size]
        r: Number of tokens read before current step [batch_size]
        s: Number of skips remaining before current step [batch_size]
        n: Predicted number of skips [batch_size]
        probs: Probability of the predicted skip action [batch_size]
        valid: Whether skip decision is valid at this step [batch_size]
        jump: Number of jumps taken so far [batch_size]
        e: End flag indicating if processing is complete [batch_size]
    """
    c: jnp.ndarray  # [batch_size, hidden_size]
    h: jnp.ndarray  # [batch_size, hidden_size]
    r: jnp.ndarray  # [batch_size] int
    s: jnp.ndarray  # [batch_size] int
    n: jnp.ndarray  # [batch_size] int
    probs: jnp.ndarray  # [batch_size] float
    valid: jnp.ndarray  # [batch_size] bool
    jump: jnp.ndarray  # [batch_size] int
    e: jnp.ndarray  # [batch_size] bool


class ACLSkipLSTMCell(nn.Module):
    """LSTM cell with learned skipping behavior.

    This cell can learn to skip input tokens during processing, enabling
    efficient text skimming while maintaining good performance on the task.

    Attributes:
        hidden_size: Number of hidden units
        min_read: Minimum tokens to read before allowing a skip
        max_skip: Maximum number of tokens to skip in one jump
        max_jump: Maximum total number of jumps allowed (-1 for unlimited)
        eps: Epsilon for epsilon-greedy exploration during training
    """
    hidden_size: int
    min_read: int = 8
    max_skip: int = 5
    max_jump: int = -1
    eps: float = 0.1

    def setup(self):
        # Standard LSTM parameters
        self.lstm_cell = nn.LSTMCell(
            features=self.hidden_size,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )

        # Skip prediction parameters
        # Output dimension is max_skip + 2:
        # [0, 1, 2, ..., max_skip, END]
        # where END (max_skip+1) indicates choosing to end processing
        self.skip_kernel = self.param(
            'skip_kernel',
            nn.initializers.xavier_uniform(),
            (self.hidden_size, self.max_skip + 2)
        )
        self.skip_bias = self.param(
            'skip_bias',
            nn.initializers.zeros,
            (self.max_skip + 2,)
        )

    def predict_skip(
        self,
        h: jnp.ndarray,
        r: jnp.ndarray,
        skip_flag: jnp.ndarray,
        is_training: bool,
        use_random: bool,
        rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predict number of steps to skip.

        Args:
            h: Hidden state [batch_size, hidden_size]
            r: Number of steps read [batch_size]
            skip_flag: Whether currently skipping [batch_size]
            is_training: Whether in training mode
            use_random: Whether to use random actions
            rng: Random key for sampling

        Returns:
            n_skip: Number of steps to skip [batch_size]
            probs: Probability of the chosen action [batch_size]
            valid: Whether this decision is valid [batch_size]
        """
        # Compute logits for skip prediction
        logits = jnp.dot(h, self.skip_kernel) + self.skip_bias  # [batch_size, max_skip+2]

        # During inference, use greedy selection
        predictions_greedy = jnp.argmax(logits, axis=-1)  # [batch_size]

        if is_training or use_random:
            rng1, rng2, rng3 = jax.random.split(rng, 3)

            # Epsilon-greedy: sample from softmax or random
            eps_values = jax.random.uniform(rng1, shape=(logits.shape[0],))

            # Sample from softmax distribution
            predictions_sampled = jax.random.categorical(rng2, logits)

            # Random uniform between [0, max_skip+2)
            predictions_random = jax.random.randint(
                rng3,
                shape=(logits.shape[0],),
                minval=0,
                maxval=self.max_skip + 2
            )

            if is_training:
                # Epsilon-greedy during training
                use_sampled = (eps_values > self.eps).astype(jnp.int32)
                predictions = (
                    use_sampled * predictions_sampled +
                    (1 - use_sampled) * predictions_random
                )
            else:
                predictions = predictions_greedy

            if use_random:
                predictions = predictions_random
        else:
            predictions = predictions_greedy

        # Compute probabilities
        probs = jax.nn.softmax(logits, axis=-1)  # [batch_size, max_skip+2]

        # Get probability of chosen action
        probs_chosen = probs[jnp.arange(predictions.shape[0]), predictions]

        # Only allow skipping if:
        # 1. Read more than min_read tokens (r > min_read - 1)
        # 2. Not currently in a skip (skip_flag == 0)
        mask_r = (r > (self.min_read - 1)).astype(jnp.int32)
        mask_s = (1 - skip_flag.astype(jnp.int32))

        # Apply masks to predictions
        n_skip = predictions * mask_r * mask_s

        # Valid decisions are those where we're eligible to skip
        valid = (mask_r * mask_s) > 0

        return n_skip, probs_chosen, valid

    @staticmethod
    def update_s(s: jnp.ndarray, skip_flag: jnp.ndarray) -> jnp.ndarray:
        """Update remaining skips counter.

        If currently skipping (skip_flag=1), decrement s by 1.
        Otherwise, keep s unchanged.
        """
        skip_flag = skip_flag.astype(jnp.int32)
        s_subtracted = s - 1
        new_s = skip_flag * s_subtracted + (1 - skip_flag) * s
        return new_s

    @staticmethod
    def update_r(r: jnp.ndarray, skip_flag: jnp.ndarray) -> jnp.ndarray:
        """Update read counter.

        If not skipping (skip_flag=0), increment r by 1.
        """
        skip_flag = skip_flag.astype(jnp.int32)
        read_flag = 1 - skip_flag
        new_r = read_flag + r
        return new_r

    def __call__(
        self,
        carry: ACLLSTMState,
        x: jnp.ndarray,
        is_training: bool = True,
        use_random: bool = False,
        rng: jax.random.PRNGKey = None
    ) -> Tuple[ACLLSTMState, jnp.ndarray]:
        """Process one timestep.

        Args:
            carry: Previous state (ACLLSTMState)
            x: Input at current timestep [batch_size, input_size]
            is_training: Whether in training mode
            use_random: Whether to use random actions
            rng: Random key for sampling

        Returns:
            new_state: Updated ACLLSTMState
            output: Hidden state output [batch_size, hidden_size]
        """
        c, h, r, s, n_old, probs_old, valid_old, jump, e = carry

        # Always compute LSTM update (even if skipping)
        (c_updated, h_updated), _ = self.lstm_cell((c, h), x)

        # Determine if we skip this step based on OLD s value
        # If s > 0, we skip current step
        skip_flag = (s > 0).astype(jnp.float32)

        # Update c and h: use updated values if not skipping, else keep old
        new_c = (1.0 - skip_flag[:, None]) * c_updated + skip_flag[:, None] * c
        new_h = (1.0 - skip_flag[:, None]) * h_updated + skip_flag[:, None] * h

        # Update r and s based on skip_flag
        r_updated = self.update_r(r, skip_flag)
        s_updated = self.update_s(s, skip_flag)

        # Predict skip using UPDATED hidden state
        if rng is None:
            rng = jax.random.PRNGKey(0)

        n_skip, probs, valid = self.predict_skip(
            h_updated, r_updated, skip_flag, is_training, use_random, rng
        )

        # Add predicted skips to s
        new_s = n_skip + s_updated

        # Reset read counter when making a valid skip decision
        reset_flag = valid.astype(jnp.int32)
        new_r = r_updated * (1 - reset_flag)

        # Update jump counter when making valid decision
        new_jump = jump + valid.astype(jnp.int32)

        # Check for end conditions:
        # 1. Model chooses to end (predicts max_skip+1)
        # 2. Hit maximum number of jumps (if max_jump > 0)
        choose_to_end = (n_skip == self.max_skip + 1)
        if self.max_jump > 0:
            hit_max_jump = (new_jump >= self.max_jump)
        else:
            hit_max_jump = jnp.zeros_like(choose_to_end, dtype=bool)

        new_e = choose_to_end | hit_max_jump

        # If already ended, freeze the state
        new_c = jnp.where(e[:, None], c, new_c)
        new_h = jnp.where(e[:, None], h, new_h)
        new_r = jnp.where(e, r, new_r)
        new_s = jnp.where(e, s, new_s)
        n_skip = jnp.where(e, n_old, n_skip)
        probs = jnp.where(e, probs_old, probs)
        valid = jnp.where(e, valid_old, valid)
        new_jump = jnp.where(e, jump, new_jump)
        new_e = jnp.where(e, e, new_e)

        new_state = ACLLSTMState(
            c=new_c,
            h=new_h,
            r=new_r,
            s=new_s,
            n=n_skip,
            probs=probs,
            valid=valid,
            jump=new_jump,
            e=new_e
        )

        return new_state, new_h

    def initialize_state(self, batch_size: int) -> ACLLSTMState:
        """Initialize the cell state.

        Args:
            batch_size: Batch size

        Returns:
            Initial ACLLSTMState with zeros
        """
        return ACLLSTMState(
            c=jnp.zeros((batch_size, self.hidden_size)),
            h=jnp.zeros((batch_size, self.hidden_size)),
            r=jnp.zeros((batch_size,), dtype=jnp.int32),
            s=jnp.zeros((batch_size,), dtype=jnp.int32),
            n=jnp.zeros((batch_size,), dtype=jnp.int32),
            probs=jnp.zeros((batch_size,)),
            valid=jnp.zeros((batch_size,), dtype=bool),
            jump=jnp.zeros((batch_size,), dtype=jnp.int32),
            e=jnp.zeros((batch_size,), dtype=bool)
        )
