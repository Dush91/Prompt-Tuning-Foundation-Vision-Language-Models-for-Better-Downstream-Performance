"""
MIPRO Optimizer for CLIP Descriptors
Uses DSPy's MIPROv2 to optimize meta-prompts for class descriptor generation.
"""

import os
import json
import torch
from typing import List, Dict, Callable

import dspy
from dspy.teleprompt import MIPROv2

import clip


class TextBasedMetric:
    """
    Simple text-based metric for evaluating descriptor quality.
    Checks if descriptor mentions key visual/satellite terms.
    """

    def __init__(self):
        self.satellite_keywords = [
            'satellite', 'aerial', 'overhead', 'view from above',
            'texture', 'pattern', 'color', 'geometric', 'arrangement'
        ]
        self.visual_keywords = [
            'green', 'blue', 'grey', 'brown', 'texture', 'pattern',
            'shape', 'area', 'region', 'patch', 'cluster'
        ]

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace=None,
    ) -> float:
        """
        Compute metric for a generated descriptor.
        Returns score between 0 and 1.
        """
        descriptor = prediction.descriptor.lower()
        class_name = example.class_name.lower()

        score = 0.0

        # Check if descriptor mentions class name or synonyms
        if class_name.split()[0] in descriptor:
            score += 0.3

        # Check for satellite/aerial terms
        sat_matches = sum(1 for kw in self.satellite_keywords if kw in descriptor)
        score += 0.4 * min(sat_matches / 3, 1.0)  # Normalize to max 0.4

        # Check for visual descriptive terms
        vis_matches = sum(1 for kw in self.visual_keywords if kw in descriptor)
        score += 0.3 * min(vis_matches / 3, 1.0)  # Normalize to max 0.3

        return min(score, 1.0)


class MIPROOptimizer:
    """
    MIPRO optimizer for descriptor generation.
    Uses text-based metric as a proxy for CLIP quality.
    """

    def __init__(
        self,
        auto: str = "light",
        lm: dspy.LM = None,
    ):
        """
        Args:
            auto: MIPRO auto mode ("light", "medium", "heavy")
            lm: DSPy language model (defaults to GPT-3.5-turbo)
        """
        self.auto = auto

        # Configure LM
        if lm is None:
            lm = dspy.LM("openai/gpt-3.5-turbo")
        dspy.configure(lm=lm)

        # Initialize text-based metric
        self.metric = TextBasedMetric()

        # Initialize MIPRO teleprompter
        self.teleprompter = MIPROv2(
            metric=self.metric,
            auto=auto,
        )

    def optimize(
        self,
        student_module: dspy.Module,
        trainset: List[dspy.Example],
        output_path: str = None,
    ) -> dspy.Module:
        """
        Run MIPRO optimization.

        Args:
            student_module: DSPy module to optimize
            trainset: Training examples
            output_path: Save optimized module here

        Returns:
            Optimized DSPy module
        """
        print(f"Starting MIPRO optimization (auto={self.auto})...")

        optimized = self.teleprompter.compile(
            student_module,
            trainset=trainset,
        )

        if output_path:
            # Save optimization results
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Note: DSPy doesn't have a direct save method, so we save the config
            config = {
                "auto_mode": self.auto,
                "trainset_size": len(trainset),
            }
            with open(output_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved optimization config to {output_path}")

        return optimized


def create_eurosat_trainset() -> List[dspy.Example]:
    """Create training set from EuroSAT classes."""
    eurosat_classes = [
        "Annual Crop Land",
        "Forest",
        "Herbaceous Vegetation Land",
        "Highway or Road",
        "Industrial Buildings",
        "Pasture Land",
        "Permanent Crop Land",
        "Residential Buildings",
        "River",
        "Sea or Lake",
    ]

    trainset = []
    for class_name in eurosat_classes:
        example = dspy.Example(
            class_name=class_name,
            domain="satellite imagery",
        ).with_inputs("class_name", "domain")
        trainset.append(example)

    return trainset


def main():
    """Run MIPRO optimization for descriptor generation."""
    from descriptor_generator import OptimizedDescriptorGenerator

    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize generator
    print("Initializing descriptor generator...")
    generator = OptimizedDescriptorGenerator()

    # Create trainset
    trainset = create_eurosat_trainset()
    print(f"Created trainset with {len(trainset)} examples")

    # Initialize optimizer
    print("Initializing MIPRO optimizer...")
    optimizer = MIPROOptimizer(auto="light")

    # Run optimization
    optimized_generator = optimizer.optimize(
        generator,
        trainset=trainset,
        output_path="output/dspy_mipro/config.json",
    )

    # Test optimized generator
    print("\nTesting optimized generator:")
    test_class = "Industrial Buildings"
    result = optimized_generator(test_class, "satellite imagery")
    print(f"{test_class}: {result.descriptor}")


if __name__ == "__main__":
    main()
