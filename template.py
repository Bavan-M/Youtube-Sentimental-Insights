import os
from pathlib import Path

list_of_files = [
    ".github/workflows/.gitkeep",
    "data/", 
    "notebooks/",  
    "src/data/",
    "src/model/",
    "src/__init__.py",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    
    
    if str(filepath).endswith('/') or not filepath.suffix:
        os.makedirs(filepath, exist_ok=True)
    else:
        filedir = filepath.parent
        if filedir != Path('.'):
            os.makedirs(filedir, exist_ok=True)
        
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                pass