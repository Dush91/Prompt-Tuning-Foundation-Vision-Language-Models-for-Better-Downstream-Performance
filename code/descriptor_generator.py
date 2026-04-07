"""
DSPy Descriptor Generator for CLIP
Defines DSPy signatures and modules for generating optimized class descriptors.
"""

import dspy
from typing import List


class ClassDescriptor(dspy.Signature):
    """
    Generate a visual description for a class that helps distinguish it from other classes.
    Focus on distinctive visual features, textures, colors, shapes, and spatial patterns.
    """
    class_name: str = dspy.InputField(desc="The name of the class to describe")
    domain: str = dspy.InputField(desc="The domain (e.g., 'satellite imagery')")
    descriptor: str = dspy.OutputField(
        desc="A concise visual description suitable for image classification"
    )


class MultiDescriptorGenerator(dspy.Signature):
    """
    Generate multiple diverse visual descriptions for a class.
    Each description should capture different visual aspects or discriminative features.
    """
    class_name: str = dspy.InputField()
    domain: str = dspy.InputField()
    num_descriptors: int = dspy.InputField()
    descriptors: List[str] = dspy.OutputField(
        desc="List of diverse visual descriptions"
    )


class DomainAwareDescriptor(dspy.Signature):
    """
    Generate a domain-specific visual description.
    Consider what makes this class visually distinctive in the given domain.
    """
    class_name: str = dspy.InputField()
    domain: str = dspy.InputField()
    distinguishing_features: str = dspy.InputField(
        desc="Key features that distinguish this class from similar classes"
    )
    descriptor: str = dspy.OutputField()


class DSPYDescriptorGenerator:
    """
    Wrapper for DSPy descriptor generation.
    Supports both basic and optimized (post-MIPRO) generation.
    """

    def __init__(self, lm: dspy.LM = None):
        """
        Args:
            lm: DSPy language model (defaults to OpenAI if not provided)
        """
        if lm is None:
            # Default to GPT-3.5-turbo
            lm = dspy.LM("openai/gpt-3.5-turbo")
        dspy.configure(lm=lm)
        self.predictor = dspy.Predict(ClassDescriptor)

    def generate_descriptor(
        self,
        class_name: str,
        domain: str = "satellite imagery",
    ) -> str:
        """Generate a single descriptor for a class."""
        result = self.predictor(class_name=class_name, domain=domain)
        return result.descriptor

    def generate_multiple(
        self,
        class_name: str,
        domain: str = "satellite imagery",
        n: int = 10,
    ) -> List[str]:
        """Generate multiple diverse descriptors for a class."""
        # Use ChainOfThought for more structured output
        multi_gen = dspy.ChainOfThought(MultiDescriptorGenerator)
        result = multi_gen(
            class_name=class_name,
            domain=domain,
            num_descriptors=n,
        )
        return result.descriptors[:n] if len(result.descriptors) > n else result.descriptors


class OptimizedDescriptorGenerator(dspy.Module):
    """
    Descriptor generator that can be optimized by MIPRO.
    The instructions in the signature will be optimized.
    """

    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(ClassDescriptor)

    def forward(self, class_name: str, domain: str = "satellite imagery") -> dspy.Prediction:
        """Forward pass for the generator."""
        return self.generator(class_name=class_name, domain=domain)


# Pre-defined meta-prompts for satellite imagery (starting point)
SATELLITE_SPECIFIC_INSTRUCTIONS = """
Generate visual descriptions optimized for satellite imagery classification.
Consider:
- Spectral characteristics (colors, reflectance)
- Textures and patterns visible from aerial view
- Spatial arrangement and geometry
- Distinctive features visible at satellite resolution
- Differences from ground-level photography
"""


def create_trainset_for_mipro(
    example_classes: List[str],
    example_images: dict = None,
) -> List[dspy.Example]:
    """
    Create a training set for MIPRO optimization.

    Args:
        example_classes: List of class names with known good descriptors
        example_images: Optional dict mapping class names to image features

    Returns:
        List of DSPy Examples for training
    """
    # For minimal scope, we'll use EuroSAT classes themselves
    # In a full implementation, we'd use human-verified descriptions

    trainset = []
    for class_name in example_classes:
        # Create example with class name as input
        # The "label" (what we optimize for) will be the CLIP accuracy
        example = dspy.Example(
            class_name=class_name,
            domain="satellite imagery",
        )
        trainset.append(example)

    return trainset


def main():
    """Test basic descriptor generation."""
    import os

    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable")
        return

    generator = DSPYDescriptorGenerator()

    test_classes = [
        "Annual Crop Land",
        "Forest",
        "Industrial Buildings",
    ]

    print("Testing DSPy Descriptor Generator:")
    print("=" * 60)

    for class_name in test_classes:
        print(f"\n{class_name}:")
        descriptor = generator.generate_descriptor(class_name)
        print(f"  {descriptor}")


if __name__ == "__main__":
    main()
