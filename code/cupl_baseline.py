"""
CuPL Baseline: Customized Prompts via Language models
Reproduces the CuPL approach from Pratt et al., ICCV 2023
"""

import os
import json
from typing import List, Dict
from openai import OpenAI


# CuPL (Full) LLM-prompts from the paper
CUPL_PROMPT_TEMPLATES = [
    "Describe what a(n) {} looks like:",
    "How can you identify a(n) {}?",
    "What does a(n) {} look like?",
    "A caption of an image of a(n) {}",
    "Describe an image from the internet of a(n) {}",
]

# CuPL (Base) - minimal 3 templates
CUPL_BASE_TEMPLATES = [
    "Describe what a/the {} looks like:",
    "Describe a/the {}:",
    "What are the identifying characteristics of a/the {}?",
]


class CuPLGenerator:
    """
    Generates class descriptors using CuPL approach.
    Queries GPT with fixed meta-prompts and aggregates responses.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.99,
        max_tokens: int = 50,
        n_generations: int = 10,
        use_base_only: bool = False,
    ):
        """
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use
            temperature: Sampling temperature (high for diversity, CuPL uses 0.99)
            max_tokens: Max tokens per generation
            n_generations: Number of descriptions to generate per prompt
            use_base_only: If True, use only 3 base templates instead of 5 full
        """
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_generations = n_generations
        self.templates = CUPL_BASE_TEMPLATES if use_base_only else CUPL_PROMPT_TEMPLATES

    def generate_descriptions(self, class_name: str) -> List[str]:
        """
        Generate descriptions for a class using CuPL meta-prompts.

        Args:
            class_name: Name of the class (e.g., "Annual Crop Land")

        Returns:
            List of description strings
        """
        all_descriptions = []

        for template in self.templates:
            prompt = template.format(class_name)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates concise visual descriptions for image classification. Provide short, descriptive sentences about what things look like visually."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.n_generations,
                )

                descriptions = [
                    choice.message.content.strip()
                    for choice in response.choices
                ]
                all_descriptions.extend(descriptions)

            except Exception as e:
                print(f"Error generating for prompt '{prompt}': {e}")
                continue

        return all_descriptions

    def generate_for_dataset(
        self,
        class_names: List[str],
        output_path: str = None,
    ) -> Dict[str, List[str]]:
        """
        Generate descriptions for all classes in a dataset.

        Args:
            class_names: List of class names
            output_path: If provided, save to this JSON file

        Returns:
            Dictionary mapping class names to lists of descriptions
        """
        results = {}

        for i, class_name in enumerate(class_names):
            print(f"Generating descriptions for {class_name} ({i+1}/{len(class_names)})...")
            descriptions = self.generate_descriptions(class_name)
            results[class_name] = descriptions
            print(f"  Generated {len(descriptions)} descriptions")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved descriptions to {output_path}")

        return results


def load_cupl_descriptors(path: str) -> Dict[str, List[str]]:
    """Load pre-generated CuPL descriptors from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    """Generate CuPL descriptors for EuroSAT dataset."""
    # EuroSAT class names (already human-readable from NEW_CNAMES)
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

    generator = CuPLGenerator(
        model="gpt-3.5-turbo",
        temperature=0.99,
        n_generations=10,
        use_base_only=False,  # Use full CuPL with 5 templates
    )

    output_file = "output/cupl_descriptors/eurosat_descriptors.json"
    descriptors = generator.generate_for_dataset(eurosat_classes, output_path=output_file)

    # Print sample
    print("\n" + "="*60)
    print("Sample descriptors for first class:")
    print("="*60)
    for desc in descriptors[eurosat_classes[0]][:5]:
        print(f"  - {desc}")


if __name__ == "__main__":
    main()
