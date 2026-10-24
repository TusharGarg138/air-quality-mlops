## Run the API on Windows

Open PowerShell in the project directory and create the virtual environment once:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the API with the project interpreter:

```powershell
.\scripts\run_api.ps1
```

The equivalent command, when the current directory is the repository root, is:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000/docs> for the API documentation. If PowerShell blocks
the script, run it without changing the machine policy:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_api.ps1
```