import csv
import requests
import argparse
from tqdm import tqdm

def bulk_import(project_name: str, csv_file: str, server_url: str = "http://127.0.0.1:8000"):
    """
    Imports prompt-response pairs from a CSV file into a Stria-LM project.
    The CSV should have 'prompt' and 'response' columns.
    """
    api_url = f"{server_url}/projects/{project_name}/add"
    
    try:
        with open(csv_file, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            for row in tqdm(rows, desc=f"Importing to '{project_name}'"):
                prompt = row.get("prompt")
                response = row.get("response")

                if not prompt or not response:
                    print(f"Skipping row due to missing prompt or response: {row}")
                    continue

                payload = {"prompt": prompt, "response": response}
                
                try:
                    res = requests.post(api_url, json=payload)
                    res.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Error importing row: {row}")
                    print(f"Error: {e}")
                    print(f"Response: {e.response.text if e.response else 'No response'}")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk import data into a Stria-LM project.")
    parser.add_argument("project_name", type=str, help="The name of the project.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file to import.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="URL of the Stria-LM server.")
    
    args = parser.parse_args()
    
    bulk_import(args.project_name, args.csv_file, args.url)
