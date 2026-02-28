import json
import os

SYNONYM_DICT_FILE = "knowledge_base/synonym_dict.json"

def add_synonyms():
    """
    Interactive script to add new synonyms to the dictionary
    """
    # Load existing synonyms
    if os.path.exists(SYNONYM_DICT_FILE):
        with open(SYNONYM_DICT_FILE, 'r') as f:
            synonym_dict = json.load(f)
    else:
        synonym_dict = {}
    
    print("Current synonyms:")
    for synonym, canonical in synonym_dict.items():
        print(f"  '{synonym}' -> '{canonical}'")
    
    print("\nAdd new synonyms (press Enter with empty input to finish):")
    
    while True:
        synonym = input("Enter synonym: ").strip()
        if not synonym:
            break
            
        canonical = input(f"Enter canonical form for '{synonym}': ").strip()
        if canonical:
            synonym_dict[synonym.lower()] = canonical.lower()
            print(f"Added: '{synonym}' -> '{canonical}'")
    
    # Save updated dictionary
    with open(SYNONYM_DICT_FILE, 'w') as f:
        json.dump(synonym_dict, f, indent=2)
    
    print(f"✅ Synonyms saved to {SYNONYM_DICT_FILE}")
    print(f"Total synonyms: {len(synonym_dict)}")

if __name__ == "__main__":
    add_synonyms()