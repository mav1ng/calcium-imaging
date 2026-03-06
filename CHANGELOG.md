# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] — 2024-XX-XX

### Added
- Professional README with scientific abstract, methodology, and usage examples
- Comprehensive docstrings (NumPy/Google style) across all modules
- Type hints for core network and training functions
- Module-level documentation explaining design rationale and architecture
- `src/` package structure with organized subpackages:
  - `src/models/` — Neural network architectures (UNet, MeanShift, losses)
  - `src/data/` — Dataset classes and preprocessing
  - `src/training/` — Training utilities and experiment scripts
  - `src/analysis/` — Evaluation, scoring, and clustering
  - `src/visualization/` — Plotting and embedding visualization
  - `src/utils/` — Configuration, helpers, and correlation computation
- `tests/` and `examples/` directories for future test suite and notebooks
- `requirements.txt` with pinned dependency versions
- `.gitignore` for Python/PyTorch projects
- MIT License
- This changelog

### Changed
- `network.py`: Added architecture diagrams, layer dimension annotations,
  design rationale comments, and removed debug print statements
- `training.py`: Documented training strategies, initialization rationale,
  and learning rate schedule design
- `config.py`: Added section headers and per-parameter documentation
- All modules: Added module-level docstrings explaining purpose and key APIs
- Replaced `print()` calls with `logging` in `network.py`

### Removed
- Commented-out debug print statements in `network.py`
- Dead code blocks (commented matplotlib debug plots) in network forward pass
- Unused trailing test code at bottom of `network.py`

## [0.1.0] — Original

### Added
- Initial implementation of U-Net + Mean-Shift neuron segmentation pipeline
- Custom contrastive embedding loss with inverse-frequency weighting
- Spatial correlation feature extraction with multiple mask geometries
- Agglomerative and k-means clustering for instance segmentation
- Neurofinder benchmark evaluation integration
- Data augmentation (random crop, rotation, flip)
- TensorBoard training monitoring
- Hyperparameter random search infrastructure
- Systematic ablation study scripts
