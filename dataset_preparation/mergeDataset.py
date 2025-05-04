import json



files = ['dataset.json','dev.json','test.json','train.json','sentiment_examples1.json','sentiment_examples2.json','sentiment_examples3.json','sentiment_examples4.json']

base_path = 'dataset_preparation/dataset_files'
output_path = 'dataset_preparation/final_dataset.json'

def merge_json_files(files, base_path, output_path):
    merged_data = []

    for file in files:
        with open(f'{base_path}/{file}', 'r') as f:
            data = json.load(f)
            merged_data.extend(data)

    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
    
if __name__ == "__main__":
    merge_json_files(files, base_path, output_path)
    print(f"Merged {len(files)} files into {output_path}")


