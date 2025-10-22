"""Simple test script to verify JAX/Flax implementation works."""
import jax
import jax.numpy as jnp
from jax_models.acl_cell import ACLSkipLSTMCell
from jax_models.model import SkimTextClassifier


def test_acl_cell():
    """Test ACLSkipLSTMCell basic functionality."""
    print("Testing ACLSkipLSTMCell...")

    batch_size = 4
    input_size = 300
    hidden_size = 200

    # Create cell
    cell = ACLSkipLSTMCell(
        hidden_size=hidden_size,
        min_read=3,
        max_skip=2,
        max_jump=5,
        eps=0.1
    )

    # Initialize
    rng = jax.random.PRNGKey(0)
    state = cell.initialize_state(batch_size)

    # Create dummy input
    x = jax.random.normal(rng, (batch_size, input_size))

    # Initialize parameters
    variables = cell.init(rng, state, x, is_training=False, use_random=False, rng=rng)

    # Forward pass
    rng, step_rng = jax.random.split(rng)
    new_state, h_out = cell.apply(
        variables,
        state,
        x,
        is_training=True,
        use_random=False,
        rng=step_rng
    )

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {h_out.shape}")
    print(f"  State h shape: {new_state.h.shape}")
    print(f"  State r: {new_state.r}")
    print(f"  State s: {new_state.s}")
    print("  ✓ ACLSkipLSTMCell test passed!")
    return True


def test_model():
    """Test SkimTextClassifier basic functionality."""
    print("\nTesting SkimTextClassifier...")

    batch_size = 4
    vocab_size = 1000
    seq_len = 20

    # Create model
    model = SkimTextClassifier(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=64,
        num_classes=2,
        max_steps=seq_len,
        min_read=3,
        max_skip=2,
        eps=0.1
    )

    # Create dummy data
    rng = jax.random.PRNGKey(42)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    lengths = jnp.array([15, 20, 12, 18])

    # Initialize model
    rng, init_rng = jax.random.split(rng)
    variables = model.init(
        init_rng,
        input_ids=input_ids,
        lengths=lengths,
        is_training=False,
        use_random=False,
        embeddings=None,
        rng=init_rng
    )

    # Forward pass
    rng, step_rng = jax.random.split(rng)
    logits, skip_info = model.apply(
        variables,
        input_ids=input_ids,
        lengths=lengths,
        is_training=True,
        use_random=False,
        embeddings=None,
        rng=step_rng
    )

    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Skip flags shape: {skip_info['skip_flags'].shape}")
    print(f"  Average skip rate: {jnp.mean(skip_info['skip_rate']):.3f}")
    print(f"  Average n_valid: {jnp.mean(skip_info['n_valid']):.3f}")
    print("  ✓ SkimTextClassifier test passed!")
    return True


def test_jit_compilation():
    """Test that JIT compilation works."""
    print("\nTesting JIT compilation...")

    batch_size = 4
    vocab_size = 1000
    seq_len = 20

    # Create model
    model = SkimTextClassifier(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=64,
        num_classes=2,
        max_steps=seq_len
    )

    # Create dummy data
    rng = jax.random.PRNGKey(42)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    lengths = jnp.array([15, 20, 12, 18])

    # Initialize
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, input_ids, lengths, False, False, None, init_rng)

    # Define JIT-compiled function
    @jax.jit
    def forward(params, input_ids, lengths):
        logits, skip_info = model.apply(
            {'params': params},
            input_ids=input_ids,
            lengths=lengths,
            is_training=False,
            use_random=False,
            embeddings=None,
            rng=None
        )
        return logits

    # Run JIT-compiled function
    logits = forward(variables['params'], input_ids, lengths)

    print(f"  JIT-compiled output shape: {logits.shape}")
    print("  ✓ JIT compilation test passed!")
    return True


def test_gradient_computation():
    """Test that gradients can be computed."""
    print("\nTesting gradient computation...")

    import optax

    batch_size = 4
    vocab_size = 1000
    seq_len = 20

    # Create model
    model = SkimTextClassifier(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=64,
        num_classes=2,
        max_steps=seq_len
    )

    # Create dummy data
    rng = jax.random.PRNGKey(42)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    lengths = jnp.array([15, 20, 12, 18])
    labels = jax.random.randint(rng, (batch_size,), 0, 2)

    # Initialize
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, input_ids, lengths, False, False, None, init_rng)

    # Define loss function
    def loss_fn(params):
        logits, skip_info = model.apply(
            {'params': params},
            input_ids=input_ids,
            lengths=lengths,
            is_training=True,
            use_random=False,
            embeddings=None,
            rng=None
        )
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        return loss

    # Compute gradients
    loss_value, grads = jax.value_and_grad(loss_fn)(variables['params'])

    print(f"  Loss value: {loss_value:.4f}")
    print(f"  Number of gradient trees: {len(jax.tree_util.tree_leaves(grads))}")
    print("  ✓ Gradient computation test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("JAX/Flax LSTM-Jump Implementation Tests")
    print("=" * 60)

    tests = [
        test_acl_cell,
        test_model,
        test_jit_compilation,
        test_gradient_computation
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            results.append(False)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    exit(main())
