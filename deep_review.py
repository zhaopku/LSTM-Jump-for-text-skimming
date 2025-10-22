"""Deep code review for JAX/Flax implementation."""
import ast
import re
from pathlib import Path
from typing import List, Tuple


class CodeReviewer:
    """Analyze JAX/Flax code for potential issues."""

    def __init__(self):
        self.issues = []
        self.warnings = []

    def check_jax_patterns(self, file_path: Path) -> None:
        """Check for common JAX/Flax patterns and anti-patterns."""
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # Check for numpy usage instead of jax.numpy
        for i, line in enumerate(lines, 1):
            # Check for np.array when jnp should be used
            if re.search(r'\bnp\.(?:array|zeros|ones|arange)', line) and 'import numpy as np' in content:
                if 'jnp' in content and 'convert' not in line.lower():
                    self.warnings.append(
                        f"{file_path}:{i}: Consider using jnp instead of np for arrays"
                    )

        # Check for in-place operations
        for i, line in enumerate(lines, 1):
            if re.search(r'(\w+)\s*\+=\s*', line) or re.search(r'(\w+)\s*\-=\s*', line):
                self.warnings.append(
                    f"{file_path}:{i}: In-place operation detected - JAX requires functional updates"
                )

        # Check for proper random key usage
        if 'jax.random' in content:
            # Look for PRNGKey usage
            if 'jax.random.PRNGKey' not in content:
                self.warnings.append(
                    f"{file_path}: Uses jax.random but doesn't create PRNGKey"
                )

    def check_flax_patterns(self, file_path: Path) -> None:
        """Check for Flax-specific patterns."""
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # Check for nn.Module
        if 'class' in content and 'nn.Module' in content:
            # Check for setup() method
            if 'def setup(self)' not in content and 'def __call__(self' in content:
                self.warnings.append(
                    f"{file_path}: nn.Module subclass should have setup() method"
                )

        # Check for compact decorator usage
        if '@nn.compact' in content and 'def setup(' in content:
            self.warnings.append(
                f"{file_path}: Don't use both @nn.compact and setup() - choose one"
            )

    def check_type_hints(self, file_path: Path) -> None:
        """Check for proper type hints."""
        with open(file_path, 'r') as f:
            content = f.read()

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if public function has return type hint
                    if not node.name.startswith('_'):
                        if node.returns is None and node.name not in ['setup', '__init__', '__post_init__']:
                            self.warnings.append(
                                f"{file_path}:{node.lineno}: Function '{node.name}' missing return type hint"
                            )

        except Exception as e:
            pass  # Skip if can't parse

    def check_jit_compatibility(self, file_path: Path) -> None:
        """Check for JIT compilation compatibility issues."""
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # Check for print statements in JIT-compiled functions
        in_jit_function = False
        for i, line in enumerate(lines, 1):
            if '@jax.jit' in line:
                in_jit_function = True
                continue

            if in_jit_function:
                if 'def ' in line and not line.strip().startswith('#'):
                    # New function, check for print in next lines
                    pass
                if 'print(' in line and not line.strip().startswith('#'):
                    self.warnings.append(
                        f"{file_path}:{i}: print() in JIT-compiled function (use jax.debug.print instead)"
                    )
                if line.strip() and not line.strip().startswith('#') and line[0] not in ' \t':
                    in_jit_function = False

    def review_file(self, file_path: Path) -> None:
        """Review a single file."""
        print(f"Reviewing {file_path}...")

        self.check_jax_patterns(file_path)
        self.check_flax_patterns(file_path)
        self.check_type_hints(file_path)
        self.check_jit_compatibility(file_path)

    def print_summary(self) -> int:
        """Print review summary."""
        print("\n" + "=" * 60)
        print("Code Review Summary")
        print("=" * 60)

        if self.issues:
            print(f"\n❌ Issues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.issues and not self.warnings:
            print("\n✓ No issues or warnings found!")
            return 0
        elif self.issues:
            print(f"\n❌ Found {len(self.issues)} issues and {len(self.warnings)} warnings")
            return 1
        else:
            print(f"\n✓ No critical issues - only {len(self.warnings)} warnings")
            return 0


def main():
    """Run code review."""
    print("=" * 60)
    print("Deep Code Review - JAX/Flax Implementation")
    print("=" * 60)
    print()

    reviewer = CodeReviewer()

    # Files to review
    jax_models_dir = Path("jax_models")
    files_to_review = [
        jax_models_dir / "acl_cell.py",
        jax_models_dir / "model.py",
        jax_models_dir / "data_utils.py",
        jax_models_dir / "train.py",
    ]

    for file_path in files_to_review:
        if file_path.exists():
            reviewer.review_file(file_path)
        else:
            print(f"⚠️  File not found: {file_path}")

    return reviewer.print_summary()


if __name__ == '__main__':
    exit(main())
