# AGENTS.md

## About This File

This file contains instructions and conventions specifically for coding agents working on the Generative EEG Augmentation project. Human contributors should refer to `README.md`. 

**What's here**: Environment setup, package management, testing procedures, code organization, quality standards, and workflow recipes. This complements `PLANS.md` (detailed execution plans) and `CLAUDE.md` (project context).

---

## ⚠️ Project Status & Prerequisites

### Current Project State: PRE-REFACTORING

**IMPORTANT**: This AGENTS.md file describes the **FUTURE STATE** of the project after completing all 8 milestones in `PLANS.md`. The project is currently in its **ORIGINAL RESEARCH STATE** and requires implementation work before most commands will function.

### What Works RIGHT NOW ✅

**Development Environment (100% Functional)**:
- ✅ uv package manager (v0.7.9)
- ✅ Python 3.13.5 (via `uv run python`)
- ✅ Virtual environment (.venv)
- ✅ All core packages installed: PyTorch 2.9.1, MNE 1.11.0, NumPy, SciPy, Matplotlib, Seaborn
- ✅ Development tools: pytest 9.0.1, streamlit 1.52.1, ruff 0.14.8, mypy 1.19.0, Jupyter Lab 4.5.0

**Existing Research Assets**:
- ✅ Model checkpoints in `exploratory notebooks/models/` (best_generator.pth, best_discriminator.pth)
- ✅ Research notebooks (5 notebooks in `exploratory notebooks/`)
- ✅ Validation notebooks (5 notebooks in `validation module/`)
- ✅ Preprocessed EEG data (9 subjects in `data/preprocessed/`)
- ✅ Raw EEG data (`data/raw/`)

**Working Commands**:
```bash
# These work NOW:
uv --version
uv run python --version
uv run python -c "import torch; import mne"
uv run jupyter lab
uv run pytest --version
uv run streamlit --version
uv pip list
```

### What Requires Implementation ❌

**Package Structure (Milestone 1-4)**:
- ❌ `src/generative_eeg_augmentation/` package (not created)
- ❌ `src/generative_eeg_augmentation/models/gan.py`
- ❌ `src/generative_eeg_augmentation/data/loader.py`
- ❌ `src/generative_eeg_augmentation/eval/eeg_metrics.py`
- ❌ `src/generative_eeg_augmentation/plots/eeg_visualizations.py`

**Testing Infrastructure (Milestone 7)**:
- ❌ `tests/` directory (not created)
- ❌ Unit tests for models, data, metrics, visualizations

**Demo Application (Milestone 6)**:
- ❌ `app/streamlit_app.py` (not created)
- ❌ `data/demo/` dataset (not created)
- ❌ `scripts/create_demo_dataset.py` (not created)

**Configuration (Milestone 1)**:
- ⚠️ `pyproject.toml` exists but has NO dependencies listed

### Commands That DON'T Work Yet

```bash
# These require PLANS.md implementation:
uv pip install -e .                    # Needs pyproject.toml config
uv pip install -e ".[dev,app]"         # Needs optional-dependencies
uv run pytest tests/ -v                # Needs tests/ directory
uv run streamlit run app/streamlit_app.py  # Needs app creation
uv run python -c "import generative_eeg_augmentation"  # Needs package
```

### How to Proceed

**Step 1: Read PLANS.md**
```bash
cat agent_docs/PLANS.md
```

**Step 2: Start with Milestone 1 - Create Package Structure**
```bash
# Create package directory
mkdir -p src/generative_eeg_augmentation/{models,data,eval,plots}

# Add __init__.py files
touch src/generative_eeg_augmentation/__init__.py
touch src/generative_eeg_augmentation/models/__init__.py
# ... etc
```

**Step 3: Update pyproject.toml with Dependencies**
Add all required dependencies to `[project]` section.

**Step 4: Install Package**
```bash
uv pip install -e .
```

**Step 5: Continue with Remaining Milestones**
Follow `PLANS.md` sequentially through Milestone 8.

### Validation Status

See `VALIDATION_REPORT.md` for comprehensive validation results:
- ✅ Environment setup: **100% working**
- ❌ Package imports: **0% working** (package not created)
- ❌ Test execution: **0% working** (tests not created)
- ❌ App execution: **0% working** (app not created)

**Bottom Line**: All development tools are ready. The package code needs to be implemented per PLANS.md before library-specific commands will work.

---

## ExecPlans Workflow

This project follows the **OpenAI Codex ExecPlans** format documented in `agent_docs/PLANS.md`.

### Key Principles
- `PLANS.md` is a **living document** - update Progress, Surprises & Discoveries, Decision Log, and Outcomes sections as you work
- Each milestone is **independently verifiable** - run validation commands after completion
- Work is **incremental** - complete one milestone before moving to the next
- All changes must be **self-contained** - a novice should be able to follow the plan end-to-end

### Before Starting Work
1. Read the current milestone in `PLANS.md`
2. Check the Progress section to see what's done
3. Mark the task you're starting as `in-progress` in Progress section
4. Follow the "Concrete steps" exactly as written
5. **Search notebooks thoroughly** - use grep with context flags (see below)

### After Completing Work
1. Run validation commands specified in the milestone
2. **Update ALL 4 SECTIONS of PLANS.md** (CRITICAL - do not skip any):
   - Progress section - mark completed items with timestamp
   - Surprises & Discoveries - add unexpected findings
   - Decision Log - document key decisions with rationale and date
   - **Outcomes & Retrospective** - summarize achievements, gaps, lessons learned
3. Commit changes with descriptive message

### CRITICAL: PLANS.md Update Requirements
**You MUST update all 4 sections after completing work:**

1. **Progress**: Detailed checklist with timestamps for each step completed
2. **Surprises & Discoveries**: Unexpected behaviors, performance insights, technical discoveries
3. **Decision Log**: Key decisions made with rationale and date
4. **Outcomes & Retrospective**: Summary of achievements, gaps, lessons learned (MUST be populated after milestone completion)

**Failure to update Outcomes & Retrospective means the milestone is NOT complete per ExecPlans format.**

### Command Pattern
```bash
# Always work from project root
cd /Users/rahul/PycharmProjects/Generative-EEG-Augmentation

# Check current progress
cat agent_docs/PLANS.md | grep -A 20 "## Progress"

# After completing a task, update PLANS.md
# Use replace_string_in_file or multi_replace_string_in_file tools
```

### Searching Notebooks Thoroughly

**CRITICAL**: Jupyter notebooks are JSON files. Simple grep searches may miss implementations.

**Best practices for finding code in notebooks:**

```bash
# Find class definitions with context (shows surrounding lines)
grep -A 50 "class ModelName" "notebook.ipynb"

# Find function definitions
grep -B 5 -A 30 "def function_name" "notebook.ipynb"

# Search for variable assignments
grep -A 10 "variable_name =" "notebook.ipynb"

# Get broader context around matches
grep -A 100 "class VAE" "notebook.ipynb" | head -150

# Find all class definitions in a notebook
grep "class " "notebook.ipynb"

# Search across all notebooks
grep -r "pattern" "exploratory notebooks/" "validation module/"
```

**When extracting architectures:**
1. **First pass**: Use `grep "class ClassName"` to find all matches
2. **Second pass**: Use `grep -A 100 "class ClassName"` to see full implementation
3. **Third pass**: Search for related variables (e.g., n_channels, n_samples, latent_dim)
4. **Verify**: Check if there are multiple implementations or versions
5. **Extract**: Copy the ACTUAL code from notebook, not a generic placeholder

**Example: Extracting VAE architecture**
```bash
# Step 1: Find the class
grep "class VAE" "exploratory notebooks/Generative_Modelling_VAE.ipynb"

# Step 2: Get full implementation
grep -A 100 "class VAE" "exploratory notebooks/Generative_Modelling_VAE.ipynb" | head -120

# Step 3: Find parameters
grep "n_channels\|n_samples\|latent_dim" "exploratory notebooks/Generative_Modelling_VAE.ipynb"

# Step 4: Verify data shape
grep "data.shape" "exploratory notebooks/Generative_Modelling_VAE.ipynb"
```

---

## Environment Setup

### Python Version
- **Required**: Python 3.9+ (currently using 3.10.15)
- **Recommended**: Python 3.10 or 3.11 for stability

### Environment Setup with uv

```bash
# 1. Navigate to project root
cd /Users/rahul/PycharmProjects/Generative-EEG-Augmentation

# 2. Create virtual environment with uv (extremely fast)
uv venv

# 3. Activate environment (if needed for IDE integration)
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows

# 4. Install project with all dependencies using uv
uv pip install -e .

# 5. Verify installation
uv run python -c "import generative_eeg_augmentation; print('✓ Package installed')"
```

### Alternative: Install from requirements.txt

```bash
# Create environment and install from requirements
uv venv
uv pip install -r requirements.txt
uv pip install -e .
```

### Development Dependencies

```bash
# Install development tools (testing, notebooks, linting)
uv pip install -e ".[dev]"

# Install app dependencies (Streamlit)
uv pip install -e ".[app]"

# Install all dependencies
uv pip install -e ".[dev,app]"
```

### Environment Verification

```bash
# Check Python version
uv run python --version

# Check key packages
uv pip list | grep -E "(torch|mne|streamlit|pytest)"

# Verify package structure
ls -la src/generative_eeg_augmentation/

# Run a quick import test
uv run python -c "
from generative_eeg_augmentation.models.gan import EEGGenerator
from generative_eeg_augmentation.data.loader import load_demo_preprocessed
print('✓ All imports successful')
"
```

---

## Package Management with uv

**Use `uv` for all Python package operations** - it's significantly faster than pip and handles dependency resolution better.

### Common Commands

```bash
# Install a package
uv pip install <package-name>

# Install specific version
uv pip install "torch==2.5.1"

# Install from requirements
uv pip install -r requirements.txt

# Install project in editable mode
uv pip install -e .

# Install with extras
uv pip install -e ".[dev,app]"

# Compile requirements (like pip-compile)
uv pip compile pyproject.toml -o requirements.txt

# List installed packages
uv pip list

# Show package details
uv pip show <package-name>

# Uninstall package
uv pip uninstall <package-name>
```

### Running Code with uv

```bash
# Run Python scripts with uv (ensures dependencies are available)
uv run python scripts/create_demo_dataset.py

# Run pytest with uv
uv run pytest tests/ -v

# Run Streamlit app with uv
uv run streamlit run app/streamlit_app.py

# Run notebooks (use jupyter installed in environment)
uv run jupyter lab
```

### Updating Dependencies

```bash
# After adding new dependencies to pyproject.toml
uv pip install -e .

# Reinstall project to sync all dependencies
uv pip install -e . --reinstall
```

---

## Running Code

### Project Root
Always execute commands from: `/Users/rahul/PycharmProjects/Generative-EEG-Augmentation`

### Running Python Scripts

```bash
# Always use uv run - handles dependencies and environment automatically
uv run python <script-path>

# Example: Create demo dataset
uv run python scripts/create_demo_dataset.py

# Example: Run any Python script
uv run python my_script.py
```

### Running Jupyter Notebooks

```bash
# Start Jupyter Lab
uv run jupyter lab

# Start Jupyter Notebook
uv run jupyter notebook

# Execute notebook from command line (for validation)
uv run jupyter nbconvert --to notebook --execute --inplace "validation module/Spectral and Temporal EValuation.ipynb"

# Clear notebook outputs
uv run jupyter nbconvert --clear-output --inplace <notebook-path>
```

### Running Streamlit App

```bash
# Development mode (with auto-reload)
uv run streamlit run app/streamlit_app.py

# Production mode
uv run streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

# With specific config
uv run streamlit run app/streamlit_app.py --server.maxUploadSize 200
```

### Quick Code Execution

```bash
# Run inline Python code with uv
uv run python -c "import torch; print(torch.__version__)"

# Use pylance MCP server to run snippets (preferred for complex code)
# This tool automatically uses the correct Python interpreter
mcp_pylance_mcp_s_pylanceRunCodeSnippet with:
  workspaceRoot: "/Users/rahul/PycharmProjects/Generative-EEG-Augmentation"
  codeSnippet: "import torch; print(torch.__version__)"
```

---

## Testing Instructions

### Test Organization

```
tests/
├── test_models.py          # Model instantiation, loading, forward pass
├── test_data_loader.py     # Data loading functions
├── test_metrics.py         # Evaluation metrics computation
├── test_visualizations.py  # Plotting functions
└── test_integration.py     # End-to-end workflows
```

### Running Tests

```bash
# Run all tests with uv
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_models.py -v

# Run specific test function
uv run pytest tests/test_models.py::test_generator_instantiation -v

# Run tests matching pattern
uv run pytest -k "generator" -v

# Run with coverage
uv run pytest tests/ --cov=src/generative_eeg_augmentation --cov-report=html --cov-report=term

# Run fast (skip slow tests if marked)
uv run pytest tests/ -v -m "not slow"
```

### Test Coverage Requirements

- **Minimum**: 80% line coverage for all library code
- **Required**: 100% coverage for public API functions
- **Check coverage**: Open `htmlcov/index.html` after running with `--cov-report=html`

### Coverage Commands

```bash
# Generate coverage report
uv run pytest tests/ --cov=src/generative_eeg_augmentation --cov-report=html --cov-report=term

# View coverage in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Check if coverage meets threshold
uv run pytest tests/ --cov=src/generative_eeg_augmentation --cov-fail-under=80
```

### Writing Tests

```python
# Test file template
import pytest
import numpy as np
import torch
from generative_eeg_augmentation.models.gan import EEGGenerator, load_generator

def test_generator_instantiation():
    """Test that EEGGenerator can be instantiated with default parameters."""
    gen = EEGGenerator()
    assert gen.n_channels == 63
    assert gen.target_signal_len == 1001

def test_load_generator_original():
    """Test loading original generator checkpoint."""
    gen = load_generator(model_variant="original", device="cpu")
    assert isinstance(gen, EEGGenerator)
    
    # Verify generation works
    noise = torch.randn(2, 100)
    labels = torch.zeros(2, 2)
    labels[:, 1] = 1
    output = gen(noise, labels)
    assert output.shape == (2, 63, 1001)
```

### Test Validation Checklist

Before committing:
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Coverage >80%: `uv run pytest tests/ --cov=src/generative_eeg_augmentation --cov-fail-under=80`
- [ ] No import errors
- [ ] Tests are deterministic (use `torch.manual_seed()` for reproducibility)
- [ ] New functions have corresponding tests

### Writing Quality Tests: Avoiding Superficial Testing

**CRITICAL LESSON LEARNED**: During comprehensive validation of Milestones 1-3, mutation testing revealed that 63/63 tests passed with 90% coverage, yet a critical VAE bug went undetected. Tests checked **structure** (shapes, types) but not **behavior** (correctness).

#### The Problem: Tests That Only Check "Doesn't Crash"

**Common mistake**:
```python
def test_generator_forward_pass():
    """Test generator forward pass."""
    gen = EEGGenerator()
    noise = torch.randn(10, 100)
    labels = torch.zeros(10, 2)
    labels[:, 0] = 1
    output = gen(noise, labels)
    
    # ❌ SUPERFICIAL: Only checks shape
    assert output.shape == (10, 63, 1001)
```

**This test would PASS even if**:
- Generator returns zeros
- Generator ignores labels completely
- Generator has no layers (empty network)
- Generator is non-deterministic in eval mode

#### The Solution: Test Behavior, Not Just Structure

**Add behavioral validation**:
```python
def test_generator_label_conditioning():
    """Verify labels actually affect generator output."""
    gen = EEGGenerator()
    gen.eval()
    
    noise = torch.randn(10, 100)
    labels_c0 = torch.zeros(10, 2); labels_c0[:, 0] = 1
    labels_c1 = torch.zeros(10, 2); labels_c1[:, 1] = 1
    
    with torch.no_grad():
        out_c0 = gen(noise, labels_c0)
        out_c1 = gen(noise, labels_c1)
    
    # ✅ BEHAVIORAL: Verify labels affect output
    diff = torch.abs(out_c0 - out_c1).mean().item()
    assert diff > 0.001, f"Labels should affect output, diff={diff:.6f}"
    
    # ✅ BEHAVIORAL: Verify outputs are different (not just noise)
    correlation = torch.corrcoef(torch.stack([
        out_c0.flatten(), out_c1.flatten()
    ]))[0, 1].item()
    assert correlation < 0.999, f"Outputs nearly identical, labels may not be used"
```

#### Quality Test Checklist

**For every function, write tests that verify**:

1. **Structure Tests** (necessary but insufficient):
   - [ ] Output shapes are correct
   - [ ] Output types are correct
   - [ ] No crashes on valid inputs
   - [ ] Error handling for invalid inputs

2. **Behavioral Tests** (CRITICAL - these catch real bugs):
   - [ ] Output values are correct (not just non-NaN)
   - [ ] Different inputs produce different outputs
   - [ ] Conditional inputs (labels, modes) affect output
   - [ ] Deterministic functions produce same output given same input
   - [ ] Stochastic functions differ when intended
   - [ ] Output satisfies mathematical constraints (variance ≥0, frequencies ≤ Nyquist, etc.)

3. **Known Input/Output Tests** (gold standard):
   - [ ] Use synthetic data with known properties
   - [ ] Verify computed values match manual calculations
   - [ ] Example: 10Hz sine wave → PSD peak at 10Hz
   - [ ] Example: constant signal → variance = 0

4. **Integration Tests**:
   - [ ] Cross-module workflows work end-to-end
   - [ ] Data flows correctly between components
   - [ ] Results are semantically meaningful, not just structurally valid

#### Examples of Quality Tests Added

**1. VAE Determinism (caught critical bug)**:
```python
def test_vae_deterministic_in_eval():
    """Verify VAE is deterministic in eval mode."""
    vae = VAE()
    vae.eval()
    
    x = torch.randn(5, 63, 1001)
    with torch.no_grad():
        recon1, _, _ = vae(x)
        recon2, _, _ = vae(x)
    
    # CRITICAL: Same input MUST produce identical output
    assert torch.allclose(recon1, recon2, atol=1e-5), \
        "VAE should be deterministic in eval mode"
```

**Bug found**: VAE was non-deterministic because `reparameterize()` always sampled, even in eval mode. **All 6 existing VAE tests passed** because they only checked shapes.

**2. PSD Frequency Detection (validates core functionality)**:
```python
def test_psd_frequency_detection():
    """Verify PSD correctly identifies frequency peaks."""
    sfreq = 200
    t = np.linspace(0, 5, sfreq * 5)
    signal = np.sin(2 * np.pi * 10 * t)  # 10Hz sine wave
    data = signal.reshape(1, 1, -1)
    
    psds, freqs = compute_psd(data, sfreq=sfreq)
    peak_freq = freqs[np.argmax(psds[0, 0, :])]
    
    # CRITICAL: Peak should be at 10Hz (±1Hz tolerance)
    assert abs(peak_freq - 10.0) < 1.0, \
        f"PSD peak at {peak_freq:.2f}Hz, expected 10Hz"
```

**Why this matters**: Original test only checked PSD shape. This verifies PSD actually computes frequency content correctly.

**3. Data Loader Class Separation (validates labels aren't random)**:
```python
def test_data_loader_class_separation():
    """Verify loaded classes have meaningful separation."""
    data, labels = load_demo_preprocessed()
    
    class0 = data[labels[:, 0] == 1]
    class1 = data[labels[:, 1] == 1]
    
    # Compute between-class vs within-class variance
    within_var = (np.var(class0) + np.var(class1)) / 2
    mean_diff = np.mean(class0) - np.mean(class1)
    between_var = mean_diff**2 / 2
    ratio = between_var / (within_var + 1e-10)
    
    # CRITICAL: Classes must have SOME separation
    # Random labels → ratio ≈ 0
    # Real labels → ratio > 0.0001
    assert ratio > 0.0001, \
        f"Classes have no separation (ratio={ratio:.6f}). Labels may be random."
```

**Why this matters**: Original test only checked labels are one-hot encoded. This verifies labels correspond to actual EEG differences, not random assignment.

#### Test Quality Metrics

After adding behavioral tests, test suite improved from:
- **Before**: 63 tests, 90% coverage, 1 critical bug undetected
- **After**: 87 tests (+24 behavioral tests), 90% coverage, critical bug fixed

**Test effectiveness by category**:
| Category | Before | After | Improvement |
|----------|--------|-------|-------------|
| Structure Testing | 100% | 100% | ✅ Maintained |
| Behavioral Testing | 30% | 85% | 🚀 +55% |
| Known Input/Output | 20% | 70% | 🚀 +50% |
| Bug Detection Rate | 40% | 90% | 🚀 +50% |

#### Anti-Patterns to Avoid

1. **Only testing happy path**:
   ```python
   # ❌ BAD: Only tests that it doesn't crash
   def test_load_data():
       data, labels = load_data()
       assert data.shape[1] == 63
   ```

2. **Testing implementation details**:
   ```python
   # ❌ BAD: Test breaks if implementation changes
   def test_model_has_conv_layers():
       gen = EEGGenerator()
       assert len([m for m in gen.modules() if isinstance(m, nn.Conv1d)]) == 4
   ```

3. **Accepting any non-NaN output**:
   ```python
   # ❌ BAD: Doesn't verify output is correct
   def test_metrics():
       mean, var, kurt, skew = compute_features(data)
       assert not np.isnan(mean).any()  # Could be all zeros!
   ```

4. **Not testing edge cases**:
   ```python
   # ✅ GOOD: Test boundary conditions
   def test_constant_signal_zero_variance():
       data = np.ones((10, 5, 100)) * 5.0
       mean, var, kurt, skew = compute_features(data)
       assert np.allclose(var, 0.0, atol=1e-10)
       assert np.allclose(mean, 5.0, atol=1e-6)
   ```

#### When to Write Which Tests

**Every function needs**:
- ✅ 1-2 structural tests (shapes, types, basic error handling)
- ✅ 2-3 behavioral tests (verify correctness with different inputs)
- ✅ 1-2 known input/output tests (synthetic data with expected results)

**Complex functions additionally need**:
- Integration tests with other modules
- Performance tests if speed-critical
- Stress tests (large batch sizes, edge cases)
- Determinism tests (same input → same output)

**Don't write tests for**:
- Implementation details (number of layers, internal variables)
- External library behavior (PyTorch, NumPy - assume they work)
- Obvious Python behavior (assignment, arithmetic)

#### Test Quality Review Process

Before committing tests, ask:

1. **"Would this test catch a real bug?"**
   - If I replaced the function with `return np.zeros(...)`, would the test fail?
   - If I randomized the logic, would the test fail?

2. **"Does this test verify correctness or just absence of crashes?"**
   - Does it check actual values, not just shapes?
   - Does it use known inputs with expected outputs?

3. **"Is this test resilient to refactoring?"**
   - Does it test behavior (what) not implementation (how)?
   - Would it still pass if I rewrote the function with same behavior?

4. **"Would a novice understand what's being tested?"**
   - Is the test name descriptive?
   - Are assertions explained with error messages?
   - Are expected values documented?

#### Summary: The Golden Rule of Testing

**"Test behavior, not structure. Use known inputs with expected outputs. Verify correctness, not just absence of errors."**

If you can only add one test type, add **known input/output tests**. They catch the most bugs and are easiest to understand.

---

## Code Organization

### Package Structure

```
src/generative_eeg_augmentation/    # Main package (importable)
├── __init__.py                     # Package initialization
├── models/                         # Model architectures
│   ├── __init__.py
│   ├── gan.py                      # GAN models and loaders
│   └── vae.py                      # VAE models and loaders
├── data/                           # Data loading utilities
│   ├── __init__.py
│   └── loader.py                   # Dataset loading functions
├── eval/                           # Evaluation metrics
│   ├── __init__.py
│   └── eeg_metrics.py             # Time/frequency domain metrics
└── plots/                          # Visualization functions
    ├── __init__.py
    └── eeg_visualizations.py      # All plotting functions
```

### Import Patterns

```python
# Always import from the package, never relative imports in notebooks
from generative_eeg_augmentation.models.gan import load_generator, EEGGenerator
from generative_eeg_augmentation.data.loader import load_all_preprocessed, load_demo_preprocessed
from generative_eeg_augmentation.eval.eeg_metrics import (
    compute_time_domain_features,
    compute_psd,
    compute_band_power,
    EEG_BANDS
)
from generative_eeg_augmentation.plots.eeg_visualizations import (
    plot_statistical_comparison,
    plot_waveforms,
    plot_psd_comparison
)
```

### File Locations

- **Model checkpoints**: `exploratory notebooks/models/`
  - Original: `best_generator.pth`, `best_discriminator.pth`
  - Enhanced: `enhanced/best_generator.pth`, `enhanced/best_discriminator.pth`
- **Demo dataset**: `data/demo/preprocessed_epochs_demo-epo.fif`
- **Full datasets**: `data/preprocessed/SubXX/preprocessed_epochs-epo.fif`
- **Tests**: `tests/test_*.py`
- **App**: `app/streamlit_app.py` and `app/pages/`
- **Notebooks**: `exploratory notebooks/` and `validation module/`

### Navigating the Codebase

```bash
# Find a function definition
grep -r "def load_generator" src/

# Find all imports of a module
grep -r "from generative_eeg_augmentation.models" .

# List all test files
ls tests/

# Find notebooks using a specific function
grep -r "load_generator" "exploratory notebooks/" "validation module/"

# Check package structure
tree src/generative_eeg_augmentation/ -L 2
```

---

## Common Workflows

### 1. Starting a New Milestone

```bash
# 1. Read the milestone in PLANS.md
cat agent_docs/PLANS.md | grep -A 50 "### Milestone X"

# 2. Update Progress section (mark as in-progress)
# Use replace_string_in_file tool to update PLANS.md

# 3. Follow concrete steps in order
# 4. Run validation commands after each step
# 5. Update Progress when complete
```

### 2. Adding a New Model Architecture

```bash
# 1. Create model file
touch src/generative_eeg_augmentation/models/new_model.py

# 2. Implement model class with docstrings
# 3. Add load function (e.g., load_new_model)
# 4. Create test file
touch tests/test_new_model.py

# 5. Write tests for:
#    - Model instantiation
#    - Forward pass
#    - Loading from checkpoint
#    - Edge cases

# 6. Run tests
uv run pytest tests/test_new_model.py -v

# 7. Update __init__.py to export
# Edit src/generative_eeg_augmentation/models/__init__.py

# 8. Update PLANS.md Progress and Decision Log
```

### 3. Refactoring a Notebook

```bash
# 1. Create backup
cp "notebook.ipynb" "notebook.ipynb.bak"

# 2. Open notebook
uv run jupyter lab notebook.ipynb

# 3. Replace inline code with library imports
# Before:
# def load_data(...):
#     ... 30 lines ...
#
# After:
# from generative_eeg_augmentation.data.loader import load_all_preprocessed
# data, labels = load_all_preprocessed()

# 4. Run all cells to validate
# 5. Check for errors
# 6. Commit changes

# 7. If issues arise, restore backup
cp "notebook.ipynb.bak" "notebook.ipynb"
```

### 4. Running Complete Validation Pipeline

```bash
# 1. Generate synthetic data (if needed)
uv run jupyter nbconvert --execute --to notebook --inplace \
  "exploratory notebooks/conditional_wasserstein_gan.ipynb"

# 2. Run spectral/temporal validation
uv run jupyter nbconvert --execute --to notebook --inplace \
  "validation module/Spectral and Temporal EValuation.ipynb"

# 3. Run FID evaluation
uv run jupyter nbconvert --execute --to notebook --inplace \
  "validation module/FID_EEG EValuation.ipynb"

# 4. Check results in notebook outputs
```

### 5. Testing Streamlit App Locally

```bash
# 1. Ensure demo dataset exists
ls data/demo/preprocessed_epochs_demo-epo.fif

# If not, create it:
uv run python scripts/create_demo_dataset.py

# 2. Run app
uv run streamlit run app/streamlit_app.py

# 3. In browser (http://localhost:8501):
#    - Select model variant
#    - Click "Generate and Analyze"
#    - Verify all analysis types work
#    - Check that second run is faster (caching)

# 4. Stop app: Ctrl+C in terminal
```

### 6. Adding Dependencies

```bash
# 1. Add to pyproject.toml under [project] dependencies
# Edit pyproject.toml

# 2. Install using uv
uv pip install -e .

# 3. Verify import works
uv run python -c "import new_package; print('✓')"

# 4. Document in PLANS.md Decision Log
```

### 7. Creating Demo Dataset

```bash
# Run the creation script
uv run python scripts/create_demo_dataset.py

# Verify output
ls -lh data/demo/preprocessed_epochs_demo-epo.fif

# Should be < 10 MB and contain ~50 epochs
uv run python -c "
import mne
epochs = mne.read_epochs('data/demo/preprocessed_epochs_demo-epo.fif', preload=False)
print(f'Demo dataset: {len(epochs)} epochs')
"
```

---

## MCP Server Tools

### Context7 - Latest Library Documentation

**Use Context7 to get up-to-date documentation** for any library instead of relying on training data.

```python
# Step 1: Resolve library ID
mcp_upstash_conte_resolve-library-id(libraryName="pytorch")
# Returns: {"libraryID": "/pytorch/pytorch", ...}

# Step 2: Get documentation
mcp_upstash_conte_get-library-docs(
    context7CompatibleLibraryID="/pytorch/pytorch",
    topic="neural networks",
    tokens=5000
)
```

#### Common Libraries for This Project

```python
# PyTorch
resolve-library-id: "pytorch"
library-id: "/pytorch/pytorch"

# MNE (EEG processing)
resolve-library-id: "mne"
library-id: "/mne/mne-python"

# Streamlit
resolve-library-id: "streamlit"
library-id: "/streamlit/streamlit"

# NumPy
resolve-library-id: "numpy"
library-id: "/numpy/numpy"

# SciPy
resolve-library-id: "scipy"
library-id: "/scipy/scipy"

# Matplotlib
resolve-library-id: "matplotlib"
library-id: "/matplotlib/matplotlib"
```

#### When to Use Context7

- Checking latest API changes for a library
- Finding best practices for a specific task
- Understanding how to use a new feature
- Verifying deprecated functions
- Getting code examples for complex operations

### Pylance MCP Server - Python Environment Tools

Available tools for Python development:

```python
# Get Python environment info
mcp_pylance_mcp_s_pylancePythonEnvironments(
    workspaceRoot="file:///Users/rahul/PycharmProjects/Generative-EEG-Augmentation"
)

# Check syntax errors in file
mcp_pylance_mcp_s_pylanceFileSyntaxErrors(
    workspaceRoot="file:///Users/rahul/PycharmProjects/Generative-EEG-Augmentation",
    fileUri="file:///path/to/file.py"
)

# Run code snippet (preferred over terminal for Python)
mcp_pylance_mcp_s_pylanceRunCodeSnippet(
    workspaceRoot="file:///Users/rahul/PycharmProjects/Generative-EEG-Augmentation",
    codeSnippet="import torch; print(torch.__version__)"
)

# Get workspace user files
mcp_pylance_mcp_s_pylanceWorkspaceUserFiles(
    workspaceRoot="file:///Users/rahul/PycharmProjects/Generative-EEG-Augmentation"
)

# Check imports across workspace
mcp_pylance_mcp_s_pylanceImports(
    workspaceRoot="file:///Users/rahul/PycharmProjects/Generative-EEG-Augmentation"
)
```

### Sequential Thinking - Complex Problem Solving

Use for complex tasks that require step-by-step reasoning:

```python
mcp_sequentialthi_sequentialthinking(
    thought="Breaking down how to implement the VAE architecture...",
    thoughtNumber=1,
    totalThoughts=5,
    nextThoughtNeeded=true
)
```

---

## Quality Standards

### Code Style

```bash
# Format code with black (if configured)
black src/ tests/

# Format with ruff (faster alternative)
ruff format src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type check with mypy (optional but recommended)
mypy src/
```

### Documentation Requirements

**All public functions must have Google-style docstrings:**

```python
def load_generator(
    model_variant: str = "original",
    device: str = "cpu",
    latent_dim: int = 100,
    n_channels: int = 63,
    target_signal_len: int = 1001,
    num_classes: int = 2
) -> EEGGenerator:
    """
    Load a pre-trained EEG generator from saved checkpoint.
    
    Args:
        model_variant: One of "original", "enhanced". Determines checkpoint path.
        device: PyTorch device string ("cpu", "cuda", "mps").
        latent_dim: Dimension of latent noise vector.
        n_channels: Number of EEG channels (default 63).
        target_signal_len: Length of generated signal in samples (default 1001).
        num_classes: Number of conditional classes (default 2).
    
    Returns:
        Initialized EEGGenerator in eval mode with loaded weights.
    
    Raises:
        ValueError: If model_variant is not recognized.
        FileNotFoundError: If checkpoint file does not exist.
    
    Example:
        >>> gen = load_generator("enhanced", device="cpu")
        >>> noise = torch.randn(10, 100)
        >>> labels = torch.zeros(10, 2)
        >>> labels[:, 0] = 1
        >>> synthetic_eeg = gen(noise, labels)
        >>> synthetic_eeg.shape
        torch.Size([10, 63, 1001])
    """
    # Implementation...
```

### Type Hints

**Required for all function signatures:**

```python
from typing import Tuple, Optional, Union
import numpy as np
import torch

def compute_psd(
    data: Union[torch.Tensor, np.ndarray],
    sfreq: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density."""
    # Implementation...
```

### Testing Standards

- **Coverage**: >80% for all library code
- **Test naming**: `test_<function_name>_<scenario>`
- **Assertions**: Use specific assertions with helpful messages
- **Fixtures**: Use pytest fixtures for shared test data
- **Determinism**: Set random seeds for reproducible tests

```python
import pytest
import torch

@pytest.fixture
def dummy_eeg_data():
    """Generate dummy EEG data for testing."""
    torch.manual_seed(42)
    return torch.randn(10, 63, 1001)

def test_compute_features_with_dummy_data(dummy_eeg_data):
    """Test that features have correct shape."""
    mean, var, kurt, skew = compute_time_domain_features(dummy_eeg_data)
    assert mean.shape == (10, 63), f"Expected (10, 63), got {mean.shape}"
    assert torch.all(var >= 0), "Variance must be non-negative"
```

### Code Review Checklist

Before committing code:
- [ ] Type hints present on all function signatures
- [ ] Google-style docstrings on all public functions
- [ ] Unit tests written and passing
- [ ] Coverage >80% maintained
- [ ] No hardcoded paths (use `Path(__file__).parent` or similar)
- [ ] All imports work correctly
- [ ] No `print()` statements (use logging or remove)
- [ ] No commented-out code blocks
- [ ] PLANS.md updated (Progress, Decision Log if applicable)

---

## Git and PR Conventions

### Repository Info
- **Owner**: rahul2227
- **Repo**: Generative-EEG-Augmentation
- **Main branch**: main
- **Current worktree**: brave-wu

### Commit Message Format

```
[Component] Brief description

Detailed explanation if needed.
- Specific change 1
- Specific change 2

Updates PLANS.md Progress: Milestone X, step Y
```

**Examples:**
```
[Models] Add VAE architecture with conditional generation

- Implement Encoder and Decoder classes
- Add load_vae function for checkpoint loading
- Add unit tests for VAE forward pass

Updates PLANS.md Progress: Milestone 1, step 4
```

```
[Tests] Add integration tests for end-to-end workflow

- Test complete generation pipeline
- Test evaluation metrics on generated data
- Verify plotting functions return figures

Updates PLANS.md Progress: Milestone 7, step 2
```

```
[Docs] Update PLANS.md with Milestone 1 completion

- Mark all Milestone 1 tasks as completed
- Add discovery about model loading performance
- Document decision to use CPU for app deployment

Updates PLANS.md: Milestone 1 complete
```

### Branch Naming

```
feature/milestone-1-package-structure
feature/milestone-2-data-loading
fix/model-loading-path-error
refactor/notebook-imports
docs/update-agents-md
```

### PR Title Format

Similar to commit messages:
```
[Models] Implement Milestone 1: Package Structure and Core Models
[Tests] Add comprehensive test suite for data loading
[App] Create Streamlit application with demo interface
```

### Pre-Commit Checklist

**Before every commit:**
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Coverage >80%: `uv run pytest tests/ --cov=src/generative_eeg_augmentation --cov-fail-under=80`
- [ ] All imports work: `uv run python -c "import generative_eeg_augmentation"`
- [ ] PLANS.md Progress section updated
- [ ] PLANS.md Decision Log updated (if making design decision)
- [ ] No debugging code left in (prints, breakpoints)
- [ ] Docstrings added for new functions
- [ ] Type hints present

### Pre-PR Checklist

**Before opening a PR:**
- [ ] All commits follow message format
- [ ] Branch name follows convention
- [ ] All tests pass locally
- [ ] PLANS.md accurately reflects current state
- [ ] No merge conflicts with main
- [ ] PR description explains what changed and why

### Updating PLANS.md

**Always update these sections:**

```markdown
## Progress
- [x] (2024-12-06 14:30) Created package structure
- [x] (2024-12-06 15:15) Implemented EEGGenerator in models/gan.py
- [ ] Write unit tests for model loading <-- Mark in-progress when starting
```

```markdown
## Decision Log
- **Decision**: Use CPU-only for Streamlit app
  **Rationale**: Deployment platforms rarely offer free GPU. Inference on 50 epochs takes ~3s on CPU, which is acceptable.
  **Date**: 2024-12-06
```

```markdown
## Surprises & Discoveries
- Model loading with `map_location='cpu'` is faster than expected (~1s for 50MB checkpoint)
- MNE file reading benefits significantly from SSD vs HDD (~5x faster)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Import Errors

```bash
# Issue: "ModuleNotFoundError: No module named 'generative_eeg_augmentation'"
# Solution: Install package in editable mode
uv pip install -e .

# Issue: "ImportError: cannot import name 'load_generator'"
# Solution: Check that __init__.py exports the function
cat src/generative_eeg_augmentation/models/__init__.py
# Should contain: from .gan import load_generator, EEGGenerator
```

#### Test Failures

```bash
# Issue: Tests fail with "FileNotFoundError" for checkpoints
# Solution: Ensure model checkpoints exist
ls "exploratory notebooks/models/best_generator.pth"

# Issue: Tests fail with shape mismatches
# Solution: Check that data dimensions match expected values
# EEG data should be (n_epochs, 63, 1001)

# Issue: Flaky tests (pass sometimes, fail sometimes)
# Solution: Set random seeds for reproducibility
# In test: torch.manual_seed(42), np.random.seed(42)
```

#### Environment Issues

```bash
# Issue: "No module named 'mne'"
# Solution: Reinstall dependencies
uv pip install -e ".[dev]" --reinstall

# Issue: "RuntimeError: Device not found" for MPS/CUDA
# Solution: Use CPU device
# In code: device = "cpu"

# Issue: uv environment not working
# Solution: Recreate virtual environment
rm -rf .venv
uv venv
uv pip install -e ".[dev,app]"
```

#### Notebook Issues

```bash
# Issue: Notebook kernel dies when running cells
# Solution: Check memory usage, reduce batch size, or use demo dataset

# Issue: Notebook imports fail
# Solution: Ensure package is installed in the notebook kernel's environment
# In notebook: !uv pip install -e .

# Issue: Notebook outputs are too large
# Solution: Clear outputs before committing
uv run jupyter nbconvert --clear-output --inplace notebook.ipynb
```

#### Streamlit App Issues

```bash
# Issue: "Demo dataset not found"
# Solution: Create demo dataset
uv run python scripts/create_demo_dataset.py

# Issue: App is slow on first load
# Solution: This is expected - models and data are cached after first load

# Issue: "Address already in use"
# Solution: Kill existing Streamlit process or use different port
streamlit run app/streamlit_app.py --server.port 8502
```

### Getting Help

1. **Check PLANS.md** for detailed instructions
2. **Read CLAUDE.md** for project context
3. **Use Context7** for library-specific documentation
4. **Use Sequential Thinking** for complex debugging
5. **Check test outputs** for specific error messages
6. **Review recent commits** for similar changes

### Debugging Workflow

```bash
# 1. Reproduce the issue
uv run pytest tests/test_models.py::test_failing_test -v

# 2. Add print statements or use pdb
# In test file:
import pdb; pdb.set_trace()

# 3. Run in debug mode
uv run python -m pdb tests/test_models.py

# 4. Check logs
# Streamlit: Check terminal output
# Tests: pytest shows full tracebacks with -v

# 5. Verify environment
uv run python -c "import sys; print(sys.path)"
uv pip list
```

---

## Quick Reference

### Essential Commands

```bash
# Environment Setup
uv venv
uv pip install -e ".[dev,app]"

# Testing
uv run pytest tests/ -v
uv run pytest tests/ --cov=src/generative_eeg_augmentation

# Running
uv run python scripts/create_demo_dataset.py
uv run streamlit run app/streamlit_app.py
uv run jupyter lab

# Code Quality
ruff check src/ tests/
ruff format src/ tests/
mypy src/

# Git
git status
git add .
git commit -m "[Component] Description"
git push origin <branch-name>
```

### Key Paths

```bash
# Package
src/generative_eeg_augmentation/

# Tests
tests/

# Notebooks
exploratory notebooks/
validation module/

# App
app/streamlit_app.py

# Data
data/demo/preprocessed_epochs_demo-epo.fif
data/preprocessed/SubXX/preprocessed_epochs-epo.fif

# Models
exploratory notebooks/models/best_generator.pth
exploratory notebooks/models/enhanced/best_generator.pth

# Docs
agent_docs/PLANS.md
agent_docs/AGENTS.md
CLAUDE.md
README.md
```

### Validation Commands

```bash
# Library installation
uv run python -c "import generative_eeg_augmentation; print('✓')"

# Model loading
uv run python -c "from generative_eeg_augmentation.models.gan import load_generator; g = load_generator('original'); print('✓')"

# Data loading
uv run python -c "from generative_eeg_augmentation.data.loader import load_demo_preprocessed; d, l = load_demo_preprocessed(); print(f'✓ {d.shape}')"

# Tests
uv run pytest tests/ -v

# Coverage
uv run pytest tests/ --cov=src/generative_eeg_augmentation --cov-fail-under=80

# App
uv run streamlit run app/streamlit_app.py &
sleep 5 && curl http://localhost:8501 && echo "✓ App running"
kill %1
```

---

## Summary

This file provides agent-specific instructions for the Generative EEG Augmentation project. Key takeaways:

1. **Use `uv`** for all Python package management
2. **Follow PLANS.md** for detailed execution workflows
3. **Update PLANS.md** Progress and Decision Log as you work
4. **Test everything**: >80% coverage required
5. **Use Context7** for up-to-date library documentation
6. **Follow conventions** for commits, PRs, and code quality
7. **Work incrementally**: Complete one milestone before moving to the next

For detailed execution plans, see `agent_docs/PLANS.md`.
For project context, see `CLAUDE.md`.
For human-focused information, see `README.md`.
