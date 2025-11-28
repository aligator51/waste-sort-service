@echo off
REM === Run YOLOv8 Web API ===
cd /d I:\Projects\waste-sort-service

echo Activating virtual environment...
call venv\Scripts\activate

echo -----------------------------------
echo Starting FastAPI (YOLOv8 Web API)...
echo UI: http://localhost:8000/ui/
echo Docs: http://localhost:8000/docs
echo -----------------------------------

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
