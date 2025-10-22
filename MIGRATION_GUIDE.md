# Migration Guide: TensorFlow to JAX/Flax

This guide helps you transition from the old TensorFlow 1.8 implementation to the new JAX/Flax version.

## Quick Comparison

| Aspect | TensorFlow 1.8 (Old) | JAX/Flax (New) |
|--------|---------------------|----------------|
| Entry point | `main.py` | `main_jax.py` |
| Dependencies | `tensorflow==1.8` | `jax>=0.4.20, flax>=0.7.5` |
| Model location | `models/model_rl2.py` | `jax_models/model.py` |
| Cell implementation | `models/acl_cell.py` | `jax_models/acl_cell.py` |
| Data loading | `models/textData.py` | `jax_models/data_utils.py` |
| Training script | `models/train.py` | `jax_models/train.py` |

## Installation

### Old Version (TensorFlow)
```bash
pip install tensorflow==1.8 tqdm nltk
python main.py --skim
```

### New Version (JAX)
```bash
pip install -r requirements-jax.txt
python main_jax.py --skim
```

## Command-Line Arguments

Most arguments remain the same! The key differences:

### Renamed Arguments
- `--dropOut` → `--dropout` (lowercase)
- `--preEmbedding` → `--pre_embedding` (snake_case)

### New Arguments
- `--seed`: Set random seed (default: 42)

### Removed Arguments
- `--device`: JAX automatically handles device placement
- `--testModel`: Use separate evaluation script instead
- `--printgate`: Outputs are logged during training

## Code Migration

### Loading a Model

**Old (TensorFlow):**
```python
from models.model_rl2 import Model
from models.textData import TextData

textData = TextData(args)
model = Model(args, textData)
```

**New (JAX):**
```python
from jax_models import SkimTextClassifier, TextDataset

dataset = TextDataset(...)
model = SkimTextClassifier(vocab_size=dataset.get_vocab_size(), ...)
```

### Running Inference

**Old (TensorFlow):**
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions, feed_dict={...})
```

**New (JAX):**
```python
variables = model.init(rng, input_ids, lengths, ...)
logits, skip_info = model.apply(variables, input_ids, lengths, ...)
predictions = jnp.argmax(logits, axis=-1)
```

### Custom Training Loop

**Old (TensorFlow):**
```python
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    for epoch in range(epochs):
        _, loss_val = sess.run([train_op, loss], feed_dict={...})
```

**New (JAX):**
```python
import optax

tx = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits, skip_info = state.apply_fn({'params': params}, ...)
        return compute_loss(logits, batch['labels'], skip_info)

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

for epoch in range(epochs):
    for batch in batches:
        state = train_step(state, batch)
```

## Performance Improvements

The JAX implementation offers several advantages:

1. **Faster Training**: JIT compilation provides 2-5x speedup
2. **Better Memory**: More efficient memory usage
3. **Modern Hardware**: Native TPU support
4. **Easier Debugging**: Pure functions, no session management
5. **Reproducibility**: Explicit RNG handling

## Checkpoint Compatibility

⚠️ **Important**: TensorFlow and JAX checkpoints are **NOT** compatible.

If you have a trained TensorFlow model, you'll need to:

1. Extract weights from TensorFlow checkpoint
2. Convert to JAX format (write custom conversion script)
3. Load into JAX model

Or simply retrain using the JAX version (recommended).

## Testing Your Migration

Run the test suite to verify everything works:

```bash
python test_jax_model.py
```

Expected output:
```
Testing ACLSkipLSTMCell...
  ✓ ACLSkipLSTMCell test passed!
Testing SkimTextClassifier...
  ✓ SkimTextClassifier test passed!
...
Results: 4/4 tests passed
✓ All tests passed!
```

## Getting Help

- **JAX Documentation**: https://jax.readthedocs.io
- **Flax Documentation**: https://flax.readthedocs.io
- **Issues**: Open a GitHub issue with `[JAX]` prefix

## Why Migrate?

1. **TensorFlow 1.8 is deprecated** - No security updates
2. **Better performance** - JAX is faster
3. **Modern ecosystem** - Better tooling and libraries
4. **Research-friendly** - Easier to experiment and modify
5. **Future-proof** - Active development and support

## Gradual Migration

You can use both versions side-by-side:

```bash
# Old version
python main.py --skim

# New version
python main_jax.py --skim
```

Compare results to ensure consistency before fully switching.
