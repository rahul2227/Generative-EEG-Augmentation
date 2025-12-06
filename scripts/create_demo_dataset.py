"""Create a small demo dataset for Streamlit app and quick testing."""

import mne
from pathlib import Path


def main():
    """Create demo dataset from first available subject."""
    project_root = Path(__file__).parent.parent
    preprocessed_path = project_root / "data" / "preprocessed"
    demo_output_dir = project_root / "data" / "demo"
    demo_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load first available subject
    subject_folders = sorted([d for d in preprocessed_path.iterdir() if d.is_dir()])
    if not subject_folders:
        raise RuntimeError("No preprocessed subjects found.")
    
    first_subject = subject_folders[0]
    fif_path = first_subject / "preprocessed_epochs-epo.fif"
    
    if not fif_path.exists():
        raise FileNotFoundError(
            f"No preprocessed epochs found for {first_subject.name}"
        )
    
    print(f"Loading epochs from {first_subject.name}...")
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    
    # Take first 50 epochs
    n_demo_epochs = min(50, len(epochs))
    demo_epochs = epochs[:n_demo_epochs]
    
    output_path = demo_output_dir / "preprocessed_epochs_demo-epo.fif"
    demo_epochs.save(output_path, overwrite=True)
    print(f"✓ Saved {n_demo_epochs} epochs to {output_path}")
    print(f"✓ Demo data shape: {demo_epochs.get_data().shape}")
    
    # Check file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
