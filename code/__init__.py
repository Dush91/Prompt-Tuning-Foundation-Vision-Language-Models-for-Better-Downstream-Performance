"""
DSPy-MIPRO for Vision-Language Models

This package contains code for optimizing CLIP prompts using DSPy-MIPRO
and integrating with MaPLe for few-shot learning.

Modules:
    descriptor_generator: DSPy signatures for descriptor generation
    mipro_optimizer: MIPRO optimization wrapper
    cupl_baseline: CuPL baseline implementation
    evaluate_clip_accuracy: Zero-shot CLIP evaluation
    maple_mipro_init: MaPLe with MIPRO initialization
    compare_mipro_init: Initialization comparison utilities

Version: 0.1.0
"""

__version__ = "0.1.0"

__all__ = [
    "DescriptorGenerator",
    "optimize_descriptors",
    "CLIPTextMetric",
    "generate_cupl_descriptors",
    "evaluate_clip_accuracy",
    "MaPLeMIPRO",
    "MultiModalPromptLearnerMIPRO",
    "compare_initialization",
]
