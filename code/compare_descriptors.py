"""
Compare descriptors from different methods qualitatively
"""

import json


def load_descriptors(path):
    with open(path) as f:
        return json.load(f)


def compare_methods():
    # Load all descriptor sets
    cupl = load_descriptors('output/cupl_descriptors/eurosat_descriptors.json')
    dspy = load_descriptors('output/dspy_baseline/eurosat_descriptors.json')
    mipro = load_descriptors('output/dspy_mipro/eurosat_descriptors.json')

    print("=" * 70)
    print("DESCRIPTOR COMPARISON: CuPL vs DSPy Baseline")
    print("=" * 70)

    classes = list(cupl.keys())

    for cls in classes:
        print(f"\n{'='*70}")
        print(f"CLASS: {cls}")
        print("=" * 70)

        print("\n📌 CuPL (5 meta-prompts x 10 generations = 50 descriptors):")
        for desc in cupl[cls][:3]:
            print(f"  • {desc}")
        print(f"  ... and {len(cupl[cls]) - 3} more")

        print("\n🤖 DSPy Baseline (1 signature x 10 generations):")
        for desc in dspy[cls][:3]:
            print(f"  • {desc}")
        print(f"  ... and {len(dspy[cls]) - 3} more")

        print("\n✨ DSPy-MIPRO Optimized (10 generations):")
        for desc in mipro[cls][:3]:
            print(f"  • {desc}")
        print(f"  ... and {len(mipro[cls]) - 3} more")

    # Statistics
    print(f"\n{'='*70}")
    print("STATISTICS")
    print("=" * 70)
    print(f"CuPL: {sum(len(v) for v in cupl.values())} total descriptors ({sum(len(v) for v in cupl.values()) / len(cupl):.1f} per class)")
    print(f"DSPy Baseline: {sum(len(v) for v in dspy.values())} total descriptors ({sum(len(v) for v in dspy.values()) / len(dspy):.1f} per class)")
    print(f"DSPy-MIPRO: {sum(len(v) for v in mipro.values())} total descriptors ({sum(len(v) for v in mipro.values()) / len(mipro):.1f} per class)")

    # Average length
    cupl_avg_len = sum(sum(len(d) for d in v) for v in cupl.values()) / sum(len(v) for v in cupl.values())
    dspy_avg_len = sum(sum(len(d) for d in v) for v in dspy.values()) / sum(len(v) for v in dspy.values())
    print(f"\nCuPL avg description length: {cupl_avg_len:.1f} chars")
    print(f"DSPy avg description length: {dspy_avg_len:.1f} chars")


def analyze_differences():
    """Analyze qualitative differences between methods."""
    cupl = load_descriptors('output/cupl_descriptors/eurosat_descriptors.json')
    dspy = load_descriptors('output/dspy_baseline/eurosat_descriptors.json')
    mipro = load_descriptors('output/dspy_mipro/eurosat_descriptors.json')

    print("\n" + "=" * 70)
    print("QUALITATIVE ANALYSIS")
    print("=" * 70)

    # Check for key terms
    satellite_terms = ['satellite', 'aerial', 'view from above', 'overhead', 'remote sensing']
    visual_terms = ['green', 'texture', 'pattern', 'color', 'shape', 'arrangement']

    for method_name, descriptors in [('CuPL', cupl), ('DSPy Baseline', dspy), ('DSPy-MIPRO', mipro)]:
        all_text = ' '.join([' '.join(v) for v in descriptors.values()]).lower()

        print(f"\n{method_name}:")
        sat_count = sum(all_text.count(term) for term in satellite_terms)
        vis_count = sum(all_text.count(term) for term in visual_terms)
        avg_len = sum(sum(len(d) for d in v) for v in descriptors.values()) / sum(len(v) for v in descriptors.values())
        print(f"  Satellite-related terms: {sat_count}")
        print(f"  Visual descriptor terms: {vis_count}")
        print(f"  Avg description length: {avg_len:.1f} chars")


if __name__ == "__main__":
    compare_methods()
    analyze_differences()
