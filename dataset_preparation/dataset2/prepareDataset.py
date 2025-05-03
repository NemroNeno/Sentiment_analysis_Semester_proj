import json

def convert_labeled_sentences(data_lines):
    label_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    result = []

    for line in data_lines:
        if not line.strip():
            continue  # skip empty lines
        try:
            sentence, label = line.rsplit(",", 1)
            sentence = sentence.strip().strip('"')
            label = label.strip().lower()
            polarity = label_map.get(label)
            if polarity is None:
                raise ValueError(f"Unknown label: {label}")
            result.append({
                "sentence": sentence,
                "polarity": polarity
            })
        except ValueError as e:
            print(f"Skipping line due to error: {e} | Line: {line}")

    return result

if __name__ == "__main__":
    input_file = "dataset_preparation/dataset2/dataset.txt"
    output_file = "dataset_preparation/dataset2/dataset.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = f.readlines()

    labeled_sentences = convert_labeled_sentences(data_lines)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labeled_sentences, f, ensure_ascii=False, indent=4)

    print(f"Converted {len(labeled_sentences)} sentences to JSON format.")