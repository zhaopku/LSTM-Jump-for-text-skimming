"""Training script for LSTM-Jump text skimming model."""
import os
import argparse
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
from flax import jax_utils
import optax
from tqdm import tqdm
import numpy as np
from typing import Dict, Any

from jax_models.model import SkimTextClassifier, compute_loss_and_metrics, compute_rl_loss
from jax_models.data_utils import TextDataset


class TrainState(train_state.TrainState):
    """Extended train state with additional fields."""
    batch_stats: Any = None
    dropout_rng: jax.random.PRNGKey = None


def create_train_state(
    rng: jax.random.PRNGKey,
    model: SkimTextClassifier,
    learning_rate: float,
    vocab_size: int,
    pretrained_embeddings: np.ndarray = None
) -> TrainState:
    """Create initial training state.

    Args:
        rng: Random key
        model: Model instance
        learning_rate: Learning rate for optimizer
        vocab_size: Vocabulary size
        pretrained_embeddings: Optional pretrained embeddings

    Returns:
        Initial TrainState
    """
    # Initialize model
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # Dummy inputs for initialization
    dummy_input_ids = jnp.ones((1, model.max_steps), dtype=jnp.int32)
    dummy_lengths = jnp.array([model.max_steps], dtype=jnp.int32)

    # Initialize parameters
    variables = model.init(
        init_rng,
        input_ids=dummy_input_ids,
        lengths=dummy_lengths,
        is_training=False,
        use_random=False,
        embeddings=pretrained_embeddings,
        rng=init_rng
    )

    params = variables['params']

    # If using pretrained embeddings, set them in params
    if pretrained_embeddings is not None and model.use_pretrained_embeddings:
        # Need to manually set embeddings in params
        # This is a workaround for Flax's embedding layer
        pass  # Embeddings are passed at runtime

    # Create optimizer
    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_rng=dropout_rng
    )


@jax.jit
def train_step_supervised(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    embeddings: jnp.ndarray = None
) -> tuple:
    """Single training step with supervised learning (no RL).

    Args:
        state: Current training state
        batch: Batch of data
        embeddings: Optional pretrained embeddings

    Returns:
        Updated state and metrics
    """
    def loss_fn(params):
        logits, skip_info = state.apply_fn(
            {'params': params},
            input_ids=batch['input_ids'],
            lengths=batch['lengths'],
            is_training=True,
            use_random=False,
            embeddings=embeddings,
            rng=state.dropout_rng
        )

        loss, metrics = compute_loss_and_metrics(
            logits=logits,
            labels=batch['labels'],
            skip_info=skip_info,
            sparse_coeff=0.0  # No sparsity penalty in supervised mode
        )

        return loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, metrics


@jax.jit
def train_step_rl(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    sparse_coeff: float,
    n_samples: int,
    embeddings: jnp.ndarray = None
) -> tuple:
    """Single training step with RL (policy gradient).

    Args:
        state: Current training state
        batch: Batch of data
        sparse_coeff: Coefficient for sparsity reward
        n_samples: Number of samples per training example
        embeddings: Optional pretrained embeddings

    Returns:
        Updated state and metrics
    """
    # Replicate batch for n_samples
    input_ids_repeated = jnp.repeat(batch['input_ids'], n_samples, axis=0)
    lengths_repeated = jnp.repeat(batch['lengths'], n_samples, axis=0)
    labels_repeated = jnp.repeat(batch['labels'], n_samples, axis=0)

    def loss_fn(params):
        logits, skip_info = state.apply_fn(
            {'params': params},
            input_ids=input_ids_repeated,
            lengths=lengths_repeated,
            is_training=True,
            use_random=False,
            embeddings=embeddings,
            rng=state.dropout_rng
        )

        loss, metrics = compute_rl_loss(
            logits=logits,
            labels=labels_repeated,
            skip_info=skip_info,
            sparse_coeff=sparse_coeff
        )

        return loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, metrics


@jax.jit
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    embeddings: jnp.ndarray = None,
    use_random: bool = False
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step.

    Args:
        state: Training state
        batch: Batch of data
        embeddings: Optional pretrained embeddings
        use_random: Whether to use random actions

    Returns:
        Metrics dictionary
    """
    logits, skip_info = state.apply_fn(
        {'params': state.params},
        input_ids=batch['input_ids'],
        lengths=batch['lengths'],
        is_training=False,
        use_random=use_random,
        embeddings=embeddings,
        rng=None
    )

    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['labels'])
    skip_rate = jnp.mean(skip_info['skip_rate'])

    metrics = {
        'accuracy': accuracy,
        'skip_rate': skip_rate,
        'n_valid': jnp.mean(skip_info['n_valid'])
    }

    return metrics


def evaluate(
    state: TrainState,
    batches: list,
    embeddings: np.ndarray = None,
    use_random: bool = False
) -> Dict[str, float]:
    """Evaluate model on dataset.

    Args:
        state: Training state
        batches: List of batches
        embeddings: Optional pretrained embeddings
        use_random: Whether to use random actions

    Returns:
        Average metrics
    """
    all_metrics = []

    for batch in batches:
        # Convert to JAX arrays
        jax_batch = {k: jnp.array(v) for k, v in batch.items()}

        # Evaluate
        metrics = eval_step(state, jax_batch, embeddings, use_random)

        # Convert to numpy
        metrics = {k: float(v) for k, v in metrics.items()}
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics


def train(args):
    """Main training function.

    Args:
        args: Parsed command-line arguments
    """
    # Set random seed
    rng = jax.random.PRNGKey(args.seed)

    # Load data
    print("Loading dataset...")
    dataset = TextDataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        max_steps=args.max_steps,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file
    )

    # Load pretrained embeddings if specified
    pretrained_embeddings = None
    if args.pre_embedding:
        pretrained_embeddings = dataset.load_pretrained_embeddings(
            args.embedding_file,
            args.embedding_size
        )
        pretrained_embeddings = jnp.array(pretrained_embeddings)

    # Create model
    print("Creating model...")
    model = SkimTextClassifier(
        vocab_size=dataset.get_vocab_size(),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
        max_steps=args.max_steps,
        min_read=args.min_read,
        max_skip=args.max_skip,
        max_jump=args.max_jump,
        eps=args.eps,
        dropout_rate=1.0 - args.dropout,
        use_pretrained_embeddings=args.pre_embedding
    )

    # Create training state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=args.learning_rate,
        vocab_size=dataset.get_vocab_size(),
        pretrained_embeddings=pretrained_embeddings
    )

    # Load checkpoint if specified
    if args.load_model and os.path.exists(args.model_path):
        print(f"Loading checkpoint from {args.model_path}...")
        state = checkpoints.restore_checkpoint(args.model_path, state)

    # Create batches
    print("Creating batches...")
    train_batches = dataset.create_batches(dataset.train_samples, shuffle=True)
    val_batches = dataset.create_batches(dataset.val_samples, shuffle=False)
    test_batches = dataset.create_batches(dataset.test_samples, shuffle=False)

    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")
    print(f"Test batches: {len(test_batches)}")

    # Training loop
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Training
        train_metrics_list = []
        pbar = tqdm(train_batches, desc="Training")

        for batch in pbar:
            # Convert to JAX arrays
            jax_batch = {k: jnp.array(v) for k, v in batch.items()}

            # Update RNG
            rng, step_rng = jax.random.split(rng)
            state = state.replace(dropout_rng=step_rng)

            # Training step
            if args.skim:
                # RL training
                state, metrics = train_step_rl(
                    state=state,
                    batch=jax_batch,
                    sparse_coeff=args.sparse,
                    n_samples=args.n_samples,
                    embeddings=pretrained_embeddings
                )
            else:
                # Supervised training
                state, metrics = train_step_supervised(
                    state=state,
                    batch=jax_batch,
                    embeddings=pretrained_embeddings
                )

            # Convert metrics to numpy
            metrics = {k: float(v) for k, v in metrics.items()}
            train_metrics_list.append(metrics)

            # Update progress bar
            avg_acc = np.mean([m['accuracy'] for m in train_metrics_list[-10:]])
            avg_skip = np.mean([m.get('skip_rate', 0.0) for m in train_metrics_list[-10:]])
            pbar.set_postfix({'acc': f'{avg_acc:.3f}', 'skip': f'{avg_skip:.3f}'})

        # Average training metrics
        avg_train_metrics = {}
        for key in train_metrics_list[0].keys():
            avg_train_metrics[key] = np.mean([m[key] for m in train_metrics_list])

        print(f"Train - Acc: {avg_train_metrics['accuracy']:.4f}, "
              f"Skip Rate: {avg_train_metrics.get('skip_rate', 0.0):.4f}")

        # Validation
        val_metrics = evaluate(state, val_batches, pretrained_embeddings, use_random=False)
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}, "
              f"Skip Rate: {val_metrics['skip_rate']:.4f}")

        # Test
        test_metrics = evaluate(state, test_batches, pretrained_embeddings, use_random=False)
        print(f"Test  - Acc: {test_metrics['accuracy']:.4f}, "
              f"Skip Rate: {test_metrics['skip_rate']:.4f}")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"New best validation accuracy: {best_val_acc:.4f}")

            if args.model_path:
                os.makedirs(args.model_path, exist_ok=True)
                checkpoints.save_checkpoint(
                    ckpt_dir=args.model_path,
                    target=state,
                    step=epoch,
                    overwrite=True,
                    keep=3
                )
                print(f"Saved checkpoint to {args.model_path}")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description='Train LSTM-Jump model with JAX/Flax')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--dataset', type=str, default='rotten', help='Dataset name')
    parser.add_argument('--train_file', type=str, default='train.txt')
    parser.add_argument('--val_file', type=str, default='val.txt')
    parser.add_argument('--test_file', type=str, default='test.txt')
    parser.add_argument('--embedding_file', type=str, default='glove.840B.300d.txt')
    parser.add_argument('--vocab_size', type=int, default=-1, help='Max vocabulary size')

    # Model arguments
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--pre_embedding', action='store_true', help='Use pretrained embeddings')

    # Skip/RL arguments
    parser.add_argument('--skim', action='store_true', help='Enable RL-based skimming')
    parser.add_argument('--min_read', type=int, default=8, help='Min tokens before skip')
    parser.add_argument('--max_skip', type=int, default=5, help='Max tokens per skip')
    parser.add_argument('--max_jump', type=int, default=-1, help='Max total jumps (-1=unlimited)')
    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon for exploration')
    parser.add_argument('--sparse', type=float, default=10.0, help='Sparsity reward coefficient')
    parser.add_argument('--n_samples', type=int, default=3, help='RL samples per example')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)

    # Checkpoint arguments
    parser.add_argument('--model_path', type=str, default='saved_jax')
    parser.add_argument('--load_model', action='store_true', help='Load existing checkpoint')

    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    train(args)


if __name__ == '__main__':
    main()
