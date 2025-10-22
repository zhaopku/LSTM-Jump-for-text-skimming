"""Data loading and preprocessing utilities."""
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import csv


@dataclass
class Sample:
    """A single training/test sample.

    Attributes:
        sentence: List of words
        label: Class label (0 or 1 for binary classification)
        length: Actual length of sentence
        token_ids: List of token IDs
        gates: Optional gate values for transfer learning
    """
    sentence: List[str]
    label: int
    length: int
    token_ids: Optional[List[int]] = None
    gates: Optional[List[float]] = None

    def init_token_ids(self, word2id: Dict[str, int], unk_word: str = 'unk'):
        """Convert words to token IDs using vocabulary."""
        unk_id = word2id.get(unk_word, 0)
        self.token_ids = [word2id.get(word, unk_id) for word in self.sentence]

    def init_gates(self, gates: List[float]):
        """Initialize gate values for transfer learning."""
        self.gates = gates


class TextDataset:
    """Dataset for text classification with skimming.

    Handles loading data, building vocabulary, and creating batches.

    Attributes:
        data_dir: Directory containing data files
        dataset_name: Name of dataset (e.g., 'rotten')
        max_steps: Maximum sequence length
        vocab_size: Maximum vocabulary size (-1 for unlimited)
        batch_size: Batch size for training
    """

    UNK_WORD = 'unk'
    PAD_WORD = '<pad>'

    def __init__(
        self,
        data_dir: str = 'data',
        dataset_name: str = 'rotten',
        max_steps: int = 50,
        vocab_size: int = -1,
        batch_size: int = 32,
        train_file: str = 'train.txt',
        val_file: str = 'val.txt',
        test_file: str = 'test.txt'
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.max_steps = max_steps
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.word2id = {}
        self.id2word = {}

        # Build vocabulary and load data
        dataset_path = os.path.join(data_dir, dataset_name)

        self.train_samples = self._load_samples(os.path.join(dataset_path, train_file))
        self.val_samples = self._load_samples(os.path.join(dataset_path, val_file))
        self.test_samples = self._load_samples(os.path.join(dataset_path, test_file))

        # Build vocabulary from training data
        self._build_vocabulary(self.train_samples)

        # Convert words to IDs
        for sample in self.train_samples:
            sample.init_token_ids(self.word2id, self.UNK_WORD)
        for sample in self.val_samples:
            sample.init_token_ids(self.word2id, self.UNK_WORD)
        for sample in self.test_samples:
            sample.init_token_ids(self.word2id, self.UNK_WORD)

        print(f"Loaded {len(self.train_samples)} train, "
              f"{len(self.val_samples)} val, "
              f"{len(self.test_samples)} test samples")
        print(f"Vocabulary size: {len(self.word2id)}")

    def _load_samples(self, file_path: str) -> List[Sample]:
        """Load samples from file.

        Expected format: label<tab>sentence
        """
        samples = []

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            return samples

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 2:
                    continue

                label = int(parts[0])
                words = parts[1].split()

                # Truncate to max_steps
                if len(words) > self.max_steps:
                    words = words[:self.max_steps]

                sample = Sample(
                    sentence=words,
                    label=label,
                    length=len(words)
                )
                samples.append(sample)

        return samples

    def _build_vocabulary(self, samples: List[Sample]):
        """Build vocabulary from samples."""
        # Count word frequencies
        word_counts = {}
        for sample in samples:
            for word in sample.sentence:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Add special tokens
        self.word2id[self.PAD_WORD] = 0
        self.word2id[self.UNK_WORD] = 1
        self.id2word[0] = self.PAD_WORD
        self.id2word[1] = self.UNK_WORD

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Add words to vocabulary
        if self.vocab_size > 0:
            sorted_words = sorted_words[:self.vocab_size - 2]

        for idx, (word, _) in enumerate(sorted_words):
            word_id = idx + 2
            self.word2id[word] = word_id
            self.id2word[word_id] = word

    def load_gates(
        self,
        gates_file: str,
        samples: List[Sample],
        tag: str = 'train'
    ):
        """Load gate values for transfer learning.

        Args:
            gates_file: Path to gates CSV file
            samples: List of samples to attach gates to
            tag: Dataset tag ('train', 'val', or 'test')
        """
        gates_path = os.path.join(self.data_dir, self.dataset_name, gates_file)

        if not os.path.exists(gates_path):
            print(f"Warning: Gates file {gates_path} not found")
            return

        with open(gates_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            sample_cnt = 0

            for idx, row in enumerate(reader):
                if idx % 2 == 0:
                    # Sentence line (for verification)
                    sentence = ' '.join(row)[:-1].strip()
                    sample_sentence = ' '.join(samples[sample_cnt].sentence[:samples[sample_cnt].length])
                    assert sentence == sample_sentence, f"Sentence mismatch at {sample_cnt}"
                else:
                    # Gates line
                    gates = [float(s) for s in row]
                    samples[sample_cnt].init_gates(gates)
                    sample_cnt += 1

        print(f"Loaded gates for {sample_cnt} {tag} samples")

    def create_batches(
        self,
        samples: List[Sample],
        shuffle: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """Create batches from samples.

        Args:
            samples: List of samples
            shuffle: Whether to shuffle samples

        Returns:
            List of batch dictionaries
        """
        if shuffle:
            np.random.shuffle(samples)

        batches = []
        num_batches = (len(samples) + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            batch_samples = samples[i * self.batch_size:(i + 1) * self.batch_size]

            if len(batch_samples) == 0:
                continue

            batch = self._create_batch_arrays(batch_samples)
            batches.append(batch)

        return batches

    def _create_batch_arrays(self, samples: List[Sample]) -> Dict[str, np.ndarray]:
        """Convert list of samples to batch arrays.

        Args:
            samples: List of samples in this batch

        Returns:
            Dictionary with batch arrays:
                - input_ids: [batch_size, max_steps]
                - labels: [batch_size]
                - lengths: [batch_size]
                - gates: [batch_size, max_steps] (optional)
        """
        batch_size = len(samples)

        # Initialize arrays
        input_ids = np.zeros((batch_size, self.max_steps), dtype=np.int32)
        labels = np.zeros(batch_size, dtype=np.int32)
        lengths = np.zeros(batch_size, dtype=np.int32)

        has_gates = samples[0].gates is not None
        if has_gates:
            gates = np.zeros((batch_size, self.max_steps), dtype=np.float32)

        # Fill arrays
        for i, sample in enumerate(samples):
            seq_len = min(sample.length, self.max_steps)
            input_ids[i, :seq_len] = sample.token_ids[:seq_len]
            labels[i] = sample.label
            lengths[i] = seq_len

            if has_gates:
                gates[i, :seq_len] = sample.gates[:seq_len]

        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'lengths': lengths
        }

        if has_gates:
            batch['gates'] = gates

        return batch

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word2id)

    def load_pretrained_embeddings(
        self,
        embedding_file: str,
        embedding_size: int = 300
    ) -> np.ndarray:
        """Load pretrained word embeddings (e.g., GloVe).

        Args:
            embedding_file: Path to embeddings file
            embedding_size: Dimension of embeddings

        Returns:
            Embedding matrix [vocab_size, embedding_size]
        """
        embeddings = np.random.randn(len(self.word2id), embedding_size).astype(np.float32) * 0.01

        embedding_path = os.path.join(self.data_dir, self.dataset_name, embedding_file)

        if not os.path.exists(embedding_path):
            print(f"Warning: Embedding file {embedding_path} not found, using random embeddings")
            return embeddings

        print(f"Loading pretrained embeddings from {embedding_path}...")
        loaded_count = 0

        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != embedding_size + 1:
                    continue

                word = parts[0]
                if word in self.word2id:
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    embeddings[self.word2id[word]] = vector
                    loaded_count += 1

        print(f"Loaded {loaded_count}/{len(self.word2id)} word embeddings")
        return embeddings
