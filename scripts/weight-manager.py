#TODO: integrate weight manager to CLI tooling
import os
import requests
import json
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
STRIA_LM_URL = "http://127.0.0.1:8000"

# Enable weights
os.environ["ENABLE_WEIGHTS"] = "true"

def list_projects():
    """List all available projects."""
    try:
        response = requests.get(f"{STRIA_LM_URL}/projects")
        response.raise_for_status()
        projects = response.json()
        return projects.get("projects", [])
    except requests.exceptions.RequestException as e:
        print(f"Error listing projects: {e}")
        return []

def get_project_qa_pairs(project_name: str):
    """Get all QA pairs with weights for a project."""
    try:
        response = requests.get(f"{STRIA_LM_URL}/projects/{project_name}/weights")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting QA pairs: {e}")
        return None

def update_qa_weight(project_name: str, qa_id: int, new_weight: float):
    """Update the weight of a specific QA pair."""
    try:
        payload = {"qa_id": qa_id, "weight": new_weight}
        response = requests.put(f"{STRIA_LM_URL}/projects/{project_name}/weights", json=payload)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error updating weight: {e}")
        return False

def get_weight_statistics(project_name: str):
    """Get weight statistics for a project."""
    try:
        response = requests.get(f"{STRIA_LM_URL}/projects/{project_name}/weights/stats")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting weight statistics: {e}")
        return None

def main():
    print("--- Weight Manager ---")
    print("This tool helps you manage weights for your Stria-LM projects.\n")
    
    # List projects
    projects = list_projects()
    if not projects:
        print("No projects found.")
        return
    
    print("Available projects:")
    for i, project in enumerate(projects, 1):
        print(f"  {i}. {project}")
    
    # Select project
    try:
        selection = int(input(f"\nSelect a project (1-{len(projects)}): ")) - 1
        if selection < 0 or selection >= len(projects):
            print("Invalid selection.")
            return
        project_name = projects[selection]
    except ValueError:
        print("Invalid input.")
        return
    
    print(f"\nSelected project: {project_name}")
    
    while True:
        print("\nOptions:")
        print("1. View weight statistics")
        print("2. List all QA pairs with weights")
        print("3. Update a QA pair weight")
        print("4. Exit")
        
        choice = input("Choose an option (1-4): ")
        
        if choice == "1":
            stats = get_weight_statistics(project_name)
            if stats:
                print(f"\nWeight Statistics for '{project_name}':")
                print(f"  Total QA pairs: {stats.get('total_pairs', 0)}")
                print(f"  Average weight: {stats.get('average_weight', 0):.2f}")
                print(f"  Min weight: {stats.get('min_weight', 0):.2f}")
                print(f"  Max weight: {stats.get('max_weight', 0):.2f}")
                print(f"  Weight distribution:")
                for range_key, count in stats.get('weight_distribution', {}).items():
                    print(f"    {range_key}: {count} pairs")
        
        elif choice == "2":
            qa_pairs = get_project_qa_pairs(project_name)
            if qa_pairs:
                print(f"\nQA Pairs for '{project_name}':")
                for pair in qa_pairs.get('qa_pairs', [])[:10]:  # Show first 10
                    print(f"  ID {pair['id']}: Weight {pair['weight']}")
                    print(f"    Prompt: {pair['prompt_text'][:60]}...")
                    print(f"    Response: {pair['response_text'][:60]}...")
                    print()
                
                total = len(qa_pairs.get('qa_pairs', []))
                if total > 10:
                    print(f"  ... and {total - 10} more pairs")
        
        elif choice == "3":
            try:
                qa_id = int(input("Enter QA pair ID to update: "))
                new_weight = float(input("Enter new weight (0.1 - 10.0): "))
                
                if 0.1 <= new_weight <= 10.0:
                    if update_qa_weight(project_name, qa_id, new_weight):
                        print(f"Successfully updated QA pair {qa_id} to weight {new_weight}")
                    else:
                        print("Failed to update weight.")
                else:
                    print("Weight must be between 0.1 and 10.0")
            except ValueError:
                print("Invalid input.")
        
        elif choice == "4":
            break
        
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()