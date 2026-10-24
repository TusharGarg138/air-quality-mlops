"""
Automated Project Scaffolding Script.
Run this script to automatically create all directories and package marker files.
"""

from pathlib import Path

# Directories to create
DIRECTORIES = [
    ".github/workflows",
    "configs",
    "data/raw",
    "data/processed",
    "models",
    "reports/figures",
    "scripts",
    "src/api",
    "src/data",
    "src/features",
    "src/inference",
    "src/training",
    "tests",
]

# Package init files to create
INIT_FILES = [
    "src/__init__.py",
    "src/api/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/inference/__init__.py",
    "src/training/__init__.py",
    "tests/__init__.py",
]

# Gitkeep files to ensure empty tracked folders exist
GITKEEP_FILES = [
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "models/.gitkeep",
    "reports/figures/.gitkeep",
]

def main():
    print("Scaffolding MLOps Air Quality Project...")
    for dir_path in DIRECTORIES:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  [+] Created directory: {dir_path}")

    for init_file in INIT_FILES:
        p = Path(init_file)
        if not p.exists():
            p.write_text('"""Package initialization."""\n', encoding="utf-8")
            print(f"  [+] Created package file: {init_file}")

    for keep_file in GITKEEP_FILES:
        p = Path(keep_file)
        if not p.exists():
            p.touch()
            print(f"  [+] Created keep file: {keep_file}")

    print("\nProject structure initialized successfully!")
    print("You can now copy the code files into their respective folders.")

if __name__ == "__main__":
    main()