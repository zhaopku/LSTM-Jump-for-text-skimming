# LSTM-Jump for Text Skimming - JAX/Flax Implementation

Modern reimplementation of LSTM-Jump using JAX and Flax. This version provides significant improvements over the original TensorFlow 1.8 implementation:

- **Modern Framework**: Built with JAX and Flax for better performance and composability
- **Clean Architecture**: Modular, well-documented code following best practices
- **JIT Compilation**: Automatic optimization through JAX's JIT compilation
- **Type Hints**: Full type annotations for better code clarity
- **Hardware Acceleration**: Native support for GPU/TPU acceleration

## What is LSTM-Jump?

LSTM-Jump is a neural network architecture that learns to efficiently read text by selectively skipping less important words, similar to how humans skim text. The model uses reinforcement learning to learn an optimal skipping policy that balances accuracy and speed.

### Key Features

- **Adaptive Computation**: Dynamically allocates processing based on input importance
- **Reinforcement Learning**: Learns skipping behavior through policy gradients
- **Configurable Constraints**: Control minimum read length, maximum skip distance, etc.
- **Transfer Learning**: Can be pre-trained with supervised skip patterns

## Requirements

```bash
pip install -r requirements-jax.txt
```

Core dependencies:
- jax >= 0.4.20
- flax >= 0.7.5
- optax >= 0.1.7
- numpy >= 1.24.0
- tqdm >= 4.66.0

## Quick Start

### Basic Training (Supervised)

Train a standard LSTM classifier without skipping:

```bash
python main_jax.py \
  --dataset rotten \
  --batch_size 32 \
  --epochs 50 \
  --learning_rate 0.001
```

### RL-based Skimming

Enable learned skipping behavior with reinforcement learning:

```bash
python main_jax.py \
  --dataset rotten \
  --skim \
  --min_read 8 \
  --max_skip 5 \
  --sparse 10.0 \
  --n_samples 3 \
  --epochs 300
```

### With Pretrained Embeddings

Use GloVe or other pretrained word embeddings:

```bash
python main_jax.py \
  --dataset rotten \
  --skim \
  --pre_embedding \
  --embedding_file glove.840B.300d.txt \
  --embedding_size 300
```

## Command-Line Arguments

### Data Arguments
- `--data_dir`: Directory containing datasets (default: `data`)
- `--dataset`: Dataset name (default: `rotten`)
- `--train_file`: Training data filename (default: `train.txt`)
- `--val_file`: Validation data filename (default: `val.txt`)
- `--test_file`: Test data filename (default: `test.txt`)
- `--vocab_size`: Maximum vocabulary size, -1 for unlimited (default: `-1`)

### Model Arguments
- `--embedding_size`: Word embedding dimension (default: `300`)
- `--hidden_size`: LSTM hidden state dimension (default: `200`)
- `--max_steps`: Maximum sequence length (default: `50`)
- `--num_classes`: Number of output classes (default: `2`)
- `--pre_embedding`: Use pretrained word embeddings

### Skip/RL Arguments
- `--skim`: Enable RL-based skimming behavior
- `--min_read`: Minimum tokens to read before allowing a skip (default: `8`)
- `--max_skip`: Maximum tokens to skip in one jump (default: `5`)
- `--max_jump`: Maximum total jumps allowed, -1 for unlimited (default: `-1`)
- `--eps`: Epsilon for epsilon-greedy exploration (default: `0.1`)
- `--sparse`: Coefficient for sparsity reward (default: `10.0`)
- `--n_samples`: Number of RL samples per training example (default: `3`)

### Training Arguments
- `--batch_size`: Batch size (default: `32`)
- `--learning_rate`: Learning rate (default: `0.001`)
- `--dropout`: Dropout keep probability (default: `1.0`)
- `--epochs`: Number of training epochs (default: `300`)
- `--seed`: Random seed (default: `42`)

### Checkpoint Arguments
- `--model_path`: Directory for saving/loading checkpoints (default: `saved_jax`)
- `--load_model`: Load existing checkpoint before training

## Architecture Overview

### ACLSkipLSTMCell

The core component is the `ACLSkipLSTMCell`, which extends a standard LSTM with skipping behavior:

```python
from jax_models.acl_cell import ACLSkipLSTMCell

cell = ACLSkipLSTMCell(
    hidden_size=200,
    min_read=8,      # Must read 8 tokens before skipping
    max_skip=5,      # Can skip up to 5 tokens
    max_jump=-1,     # Unlimited total jumps
    eps=0.1          # 10% random exploration
)
```

The cell maintains a rich state tuple:
- `c, h`: Standard LSTM cell/hidden states
- `r`: Number of tokens read
- `s`: Remaining skips counter
- `n`: Predicted skip length
- `probs`: Action probabilities
- `valid`: Whether skip decision is valid
- `jump`: Total jumps taken
- `e`: End flag

### SkimTextClassifier

The full model combines embeddings, ACL LSTM, and classification:

```python
from jax_models.model import SkimTextClassifier

model = SkimTextClassifier(
    vocab_size=10000,
    embedding_size=300,
    hidden_size=200,
    num_classes=2,
    min_read=8,
    max_skip=5
)
```

### Training with RL

The model uses REINFORCE (policy gradient) for RL training:

1. **Sample Actions**: Generate multiple trajectories per example
2. **Compute Rewards**: Combine task reward (accuracy) + sparsity reward (skips)
3. **Normalize Rewards**: Baseline subtraction within batch
4. **Update Policy**: Gradient ascent on log-probability weighted by reward

## Performance Tips

### GPU Acceleration

JAX automatically uses GPU if available. To force CPU:

```bash
export JAX_PLATFORMS=cpu
python main_jax.py ...
```

### Memory Usage

For large models or batches:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python main_jax.py --batch_size 16
```

### JIT Compilation

Training/eval steps are JIT-compiled automatically. First iteration will be slow, then very fast.

## Code Structure

```
jax_models/
├── __init__.py           # Package exports
├── acl_cell.py          # ACLSkipLSTMCell implementation
├── model.py             # SkimTextClassifier and loss functions
├── data_utils.py        # Dataset loading and batching
└── train.py             # Training loop and CLI

main_jax.py              # Entry point
requirements-jax.txt     # Dependencies
```

## Differences from TensorFlow Version

### Improvements

1. **No Sessions/Graphs**: JAX uses functional programming, no session management
2. **Automatic Differentiation**: Native autograd, cleaner than TF1.x
3. **JIT Compilation**: Transparent performance optimization
4. **Pure Functions**: Easier to reason about and test
5. **Better Hardware Support**: Unified CPU/GPU/TPU code

### API Changes

- Use `main_jax.py` instead of `main.py`
- Checkpoints saved in Flax format (not TF checkpoints)
- Different random number handling (explicit PRNG keys)

## Examples

### Train with Different Skip Policies

Aggressive skipping (more speed, less accuracy):
```bash
python main_jax.py --skim --min_read 4 --max_skip 8 --sparse 20.0
```

Conservative skipping (more accuracy, less speed):
```bash
python main_jax.py --skim --min_read 12 --max_skip 3 --sparse 5.0
```

### Analyze Skip Patterns

The model outputs skip rates during training:

```
Epoch 50/300
Train - Acc: 0.8234, Skip Rate: 0.4521
Val   - Acc: 0.7956, Skip Rate: 0.4312
Test  - Acc: 0.7889, Skip Rate: 0.4398
```

This indicates ~45% of tokens are skipped while maintaining ~79% accuracy.

## Citation

If you use this implementation, please cite the original work:

```bibtex
@article{lstm-jump,
  title={LSTM-Jump: Learning to Skim with Learned Jumps},
  author={Original Authors},
  year={2018}
}
```

## License

Same as original implementation.

## Contact

For JAX implementation questions: See GitHub issues
For original implementation: zhmeng@student.ethz.ch
