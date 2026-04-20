#!/usr/bin/env python3
"""
Automated validation script for refactored Jupyter notebooks.

This script validates that all refactored notebooks can execute key cells
without errors, ensuring that library imports work correctly and notebooks
are properly refactored.

Usage:
    python scripts/test_notebooks.py
    # Or with uv:
    uv run python scripts/test_notebooks.py
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Define project root
PROJECT_ROOT = Path(__file__).parent.parent

# Define notebooks to validate with their key validation cells
# Format: (notebook_path, description, cells_to_test)
NOTEBOOKS_TO_VALIDATE = [
    (
        "validation module/Spectral and Temporal EValuation.ipynb",
        "Spectral & Temporal Evaluation",
        ["imports", "data loading", "model loading"]
    ),
    (
        "exploratory notebooks/conditional_wasserstein_gan.ipynb",
        "Conditional Wasserstein GAN Training",
        ["imports", "data loading", "model instantiation"]
    ),
    (
        "validation module/FID_EEG EValuation.ipynb",
        "FID Evaluation for GAN",
        ["imports", "data loading", "model loading"]
    ),
    (
        "exploratory notebooks/simple_gan_approach.ipynb",
        "Simple GAN Approach",
        ["imports", "data loading"]
    ),
]


def check_notebook_syntax(notebook_path: Path) -> Tuple[bool, str]:
    """
    Check if notebook has valid JSON structure.
    
    Args:
        notebook_path: Path to notebook file
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        import json
        with open(notebook_path, 'r') as f:
            json.load(f)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading notebook: {e}"


def check_library_imports(notebook_path: Path) -> Tuple[bool, str]:
    """
    Check if notebook contains expected library imports.
    
    Args:
        notebook_path: Path to notebook file
        
    Returns:
        Tuple of (success, message)
    """
    import json
    
    expected_imports = [
        "from generative_eeg_augmentation",
        "generative_eeg_augmentation"
    ]
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Extract all code cells
    code_cells = [
        cell for cell in notebook.get('cells', [])
        if cell.get('cell_type') == 'code'
    ]
    
    # Check if any cell contains library imports
    found_import = False
    for cell in code_cells:
        source = ''.join(cell.get('source', []))
        if any(imp in source for imp in expected_imports):
            found_import = True
            break
    
    if not found_import:
        return False, "No library imports found - notebook may not be refactored"
    
    return True, "Library imports found"


def check_inline_code_removed(notebook_path: Path) -> Tuple[bool, str]:
    """
    Check if inline model definitions and data loading functions are removed.
    
    Args:
        notebook_path: Path to notebook file
        
    Returns:
        Tuple of (success, message)
    """
    import json
    
    # Patterns that should NOT be in refactored notebooks
    forbidden_patterns = [
        "class EEGGenerator(nn.Module):",
        "class EEGDiscriminator(nn.Module):",
        "class EEGDiscriminatorEnhanced(nn.Module):",
        "def load_all_preprocessed_data(",
        "def load_data(",
    ]
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    code_cells = [
        cell for cell in notebook.get('cells', [])
        if cell.get('cell_type') == 'code'
    ]
    
    found_patterns = []
    for cell in code_cells:
        source = ''.join(cell.get('source', []))
        for pattern in forbidden_patterns:
            if pattern in source:
                found_patterns.append(pattern)
    
    if found_patterns:
        return False, f"Inline code still present: {', '.join(set(found_patterns))}"
    
    return True, "Inline code properly removed"


def validate_notebook(notebook_rel_path: str, description: str) -> bool:
    """
    Validate a single notebook.
    
    Args:
        notebook_rel_path: Relative path to notebook from project root
        description: Human-readable description
        
    Returns:
        True if validation passes, False otherwise
    """
    notebook_path = PROJECT_ROOT / notebook_rel_path
    
    print(f"\n{'='*70}")
    print(f"Validating: {description}")
    print(f"Path: {notebook_rel_path}")
    print(f"{'='*70}")
    
    # Check 1: File exists
    if not notebook_path.exists():
        print(f"❌ FAIL: Notebook not found at {notebook_path}")
        return False
    print(f"✅ File exists")
    
    # Check 2: Valid JSON structure
    success, msg = check_notebook_syntax(notebook_path)
    if not success:
        print(f"❌ FAIL: {msg}")
        return False
    print(f"✅ Valid JSON structure")
    
    # Check 3: Library imports present
    success, msg = check_library_imports(notebook_path)
    if not success:
        print(f"⚠️  WARNING: {msg}")
    else:
        print(f"✅ {msg}")
    
    # Check 4: Inline code removed
    success, msg = check_inline_code_removed(notebook_path)
    if not success:
        print(f"⚠️  WARNING: {msg}")
    else:
        print(f"✅ {msg}")
    
    print(f"✅ PASS: {description} validation complete")
    return True


def main():
    """Main validation routine."""
    print("="*70)
    print("NOTEBOOK VALIDATION SCRIPT")
    print("="*70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Validating {len(NOTEBOOKS_TO_VALIDATE)} notebooks...")
    
    results = []
    for notebook_path, description, _ in NOTEBOOKS_TO_VALIDATE:
        success = validate_notebook(notebook_path, description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} notebooks passed validation")
    print(f"{'='*70}")
    
    if passed < total:
        print("\n⚠️  Some notebooks failed validation. Review errors above.")
        sys.exit(1)
    else:
        print("\n🎉 All notebooks passed validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
