"""Modern DeepSORT — modular detector / ReID / identity layer.

Built on top of the original ``deep_sort`` core (kept intact at repo root).
Submodules are added per stage:
    detectors/   — Stage 3 (YOLOv8, NanoDet, MMDetection adapters)
    reid/        — Stage 4 (OSNet, ResNet50-IBN, transformer adapters)
    segmentation/— Stage 6
    identity/    — Stage 7 (standalone body-ReID identity database)
    eval/        — Stage 2 (TrackEval/HOTA wrapper, detector & ReID metrics)
"""

__version__ = "0.1.0"
