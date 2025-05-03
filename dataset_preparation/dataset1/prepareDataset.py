import json


file_names = ["train", "test", "dev"]
sentences = ".sen"
labels = ".lab"


def read_sen_lab_files(sen_path, lab_path):
    with open(sen_path, 'r', encoding='utf-8') as sen_file:
        sentences = [line.strip() for line in sen_file]

    with open(lab_path, 'r', encoding='utf-8') as lab_file:
        labels = [line.strip() for line in lab_file]

    if len(sentences) != len(labels):
        raise ValueError("The number of sentences and labels must match.")

    data = []
    for sentence, label in zip(sentences, labels):
        polarity = 1 if label == 'pos' else -1
        data.append({
            "sentence": sentence,
            "polarity": polarity
        })
    return data

if __name__ == "__main__":
    for i in file_names:
        sen_path = f"dataset_preparation/dataset1/{i}{sentences}"
        lab_path = f"dataset_preparation/dataset1/{i}{labels}"
        data = read_sen_lab_files(sen_path, lab_path)

        with open(f"dataset_preparation/dataset1/{i}.json", 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
