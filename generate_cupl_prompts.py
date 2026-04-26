import argparse
import json
import clip
from datasets import load_dataset


def clean(name):
    return name.replace("_", " ").replace("-", " ")


def make_prompts(class_name, dataset):
    cls = clean(class_name)

    if dataset == "eurosat":
        return [
            f"A satellite image showing {cls}.",
            f"A top-down aerial image of {cls}.",
            f"A remote sensing image containing {cls}.",
            f"A land cover satellite photo of {cls}.",
            f"An overhead view showing {cls}.",
            f"A Sentinel-2 satellite image of {cls}.",
            f"Aerial imagery showing the spatial pattern of {cls}.",
            f"A satellite scene dominated by {cls}.",
            f"A remote sensing photograph with visual features of {cls}.",
            f"A top-down image representing {cls}.",
            f"A land surface image classified as {cls}.",
            f"A satellite view with texture and colour patterns of {cls}.",
            f"An overhead land cover image showing {cls}.",
            f"A geospatial image containing {cls}.",
            f"Aerial remote sensing data showing {cls}."
        ]

    if dataset == "dtd":
        return [
            f"A close-up photo of a {cls} texture.",
            f"A surface with a {cls} pattern.",
            f"A material texture that appears {cls}.",
            f"A detailed texture image showing {cls} features.",
            f"A visual surface pattern classified as {cls}.",
            f"A repeated texture pattern that looks {cls}.",
            f"A close-up material surface with {cls} appearance.",
            f"A texture sample dominated by {cls} visual structure.",
            f"A fabric or surface pattern that is {cls}.",
            f"An image showing {cls} texture characteristics.",
            f"A material surface with visible {cls} details.",
            f"A pattern image containing {cls} texture.",
            f"A close-up view of a {cls} surface.",
            f"A textured material with {cls} visual quality.",
            f"A photo focused on {cls} texture."
        ]
    if dataset == "oxfordpets":
        return [
            f"A clear photo of a {cls} pet.",
            f"A close-up image showing the face and fur of a {cls}.",
            f"A detailed image showing the fur texture of a {cls}.",
            f"A high-quality photo of a {cls} breed.",
            f"A realistic image of a {cls} cat or dog.",
            f"A photograph showing the ears, eyes, and fur texture of a {cls}.",
            f"A side view of a {cls} showing its body shape.",
            f"A natural image of a {cls} in an everyday setting.",
            f"A pet image highlighting the coat pattern of a {cls}.",
            f"A well-lit photograph of a {cls} animal."
            f"A photo capturing the posture of a {cls}.",
            f"A close-up image focusing on the fur details of a {cls}.",
            f"A typical {cls} pet in a home or outdoor environment.",
            f"A clean image showing the appearance of a {cls}.",
            f"A breed classification image of a {cls}."
        ]
    if dataset == "flowers102":
        return [
            f"A clear photo of a {cls} flower.",
            f"A close-up image showing the petals and colours of a {cls}.",
            f"A detailed image showing the shape and texture of a {cls} flower.",
            f"A high-quality photo of a {cls} in bloom.",
            f"A realistic image of a {cls} flower in a natural setting.",
            f"A photograph showing the visual features of a {cls} flower.",
            f"A natural image containing a {cls} flower.",
            f"A typical example of a {cls} flower in nature.",
            f"A photo showing the main parts of a {cls} flower.",
            f"A well-lit image of a {cls} flower.",
            f"A close-up view focusing on the details of a {cls} flower.",
            f"A clean image showing the appearance of a {cls} flower.",
            f"A floral arrangement featuring a {cls} flower.",
            f"A botanical image classified as a {cls} flower.",
            f"An outdoor photo showcasing the beauty of a {cls} flower."
        ] 
    return [
        f"A clear photo of a {cls}.",
        f"A close-up image of a {cls}.",
        f"A natural image containing a {cls}.",
        f"A realistic photo showing a {cls}.",
        f"A typical example of a {cls}.",
        f"A photo showing the shape of a {cls}.",
        f"An image showing the visual features of a {cls}.",
        f"A recognizable object image of a {cls}.",
        f"A photo of a {cls} in a real-world setting.",
        f"A detailed image showing the appearance of a {cls}.",
        f"A centered photo of a {cls}.",
        f"A high quality image of a {cls}.",
        f"A visual example of a {cls}.",
        f"A photograph showing the main parts of a {cls}.",
        f"A clear object photo classified as {cls}."
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["eurosat", "dtd", "caltech101", "oxfordpets", "flowers102"], help="Dataset name")
    args = parser.parse_args()

    _, preprocess = clip.load("ViT-B/16", device="cpu")
    _, _, class_names = load_dataset(args.dataset, preprocess)

    prompts = {}
    for cls in class_names:
        prompts[cls] = make_prompts(cls, args.dataset)

    output_path = f"prompts/{args.dataset}_cupl_prompts.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4)

    print(f"Saved stronger CuPL prompts to {output_path}")
    print(f"Classes: {len(class_names)}")


if __name__ == "__main__":
    main()