"""Syntax validation script for JAX models (no JAX required)."""
import ast
import sys
from pathlib import Path


def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    print(f"Checking {file_path}...")
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✓ Syntax OK")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_imports(file_path):
    """Check if imports are correctly structured."""
    print(f"Checking imports in {file_path}...")
    with open(file_path, 'r') as f:
        code = f.read()

    try:
        tree = ast.parse(code)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # Check for common issues
        issues = []

        # Check for duplicate imports
        if len(imports) != len(set(imports)):
            duplicates = [imp for imp in imports if imports.count(imp) > 1]
            issues.append(f"Duplicate imports: {set(duplicates)}")

        if issues:
            for issue in issues:
                print(f"  ! Warning: {issue}")
        else:
            print(f"  ✓ Imports OK")

        return len(issues) == 0

    except Exception as e:
        print(f"  ✗ Error checking imports: {e}")
        return False


def check_function_signatures(file_path):
    """Check for common function signature issues."""
    print(f"Checking function signatures in {file_path}...")
    with open(file_path, 'r') as f:
        code = f.read()

    try:
        tree = ast.parse(code)
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for mutable default arguments
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(
                            f"Function '{node.name}' has mutable default argument at line {node.lineno}"
                        )

        if issues:
            for issue in issues:
                print(f"  ! Warning: {issue}")
        else:
            print(f"  ✓ Function signatures OK")

        return len(issues) == 0

    except Exception as e:
        print(f"  ✗ Error checking function signatures: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("JAX Models Validation (Syntax & Structure)")
    print("=" * 60)
    print()

    # Files to check
    jax_models_dir = Path("jax_models")
    files_to_check = [
        jax_models_dir / "__init__.py",
        jax_models_dir / "acl_cell.py",
        jax_models_dir / "model.py",
        jax_models_dir / "data_utils.py",
        jax_models_dir / "train.py",
        Path("main_jax.py"),
        Path("test_jax_model.py"),
    ]

    results = []
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            results.append(False)
            continue

        syntax_ok = check_syntax(file_path)
        imports_ok = check_imports(file_path)
        sigs_ok = check_function_signatures(file_path)

        results.append(syntax_ok and imports_ok and sigs_ok)
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} files passed validation")
    print("=" * 60)

    if all(results):
        print("✓ All files passed validation!")
        print("\nNote: This only checks syntax and structure.")
        print("To test functionality, install JAX and run: python test_jax_model.py")
        return 0
    else:
        print("✗ Some files have issues!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
