@echo off
REM Sets up the Python 3.10 sidecar venv used by chord_detect.py.
REM Run once, after the main venv is already in place.
REM Window stays open at the end so you can read the output.

setlocal
cd /d "%~dp0"

set PYSIDECAR=C:\Users\admin\AppData\Local\Programs\Python\Python38\python.exe

echo ================================================================
echo  chord-scribe sidecar venv setup
echo ================================================================
echo  Working dir: %CD%
echo  Python:      %PYSIDECAR%
echo.

if not exist "%PYSIDECAR%" (
    echo ERROR: Python 3.8 not found at %PYSIDECAR%
    echo.
    echo Try:  where py    and    py -3.8 --version
    echo Or edit PYSIDECAR at the top of this script.
    goto :end
)

echo [1/4] Creating venv-chords with Python 3.8...
"%PYSIDECAR%" -m venv venv-chords
if errorlevel 1 (
    echo.
    echo ERROR: venv creation failed.
    goto :end
)
echo       OK
echo.

echo [2/4] Upgrading pip / setuptools / wheel...
call venv-chords\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: could not activate venv-chords.
    goto :end
)
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    goto :end
)
echo       OK
echo.

echo [3/4] Installing build deps (numpy, scipy, cython, mido)...
echo       madmom's setup.py imports numpy at install time, so build deps go first.
pip install "numpy<1.24" "scipy<1.13" "cython>=0.29,<3.0" "mido>=1.2,<1.3"
if errorlevel 1 (
    echo ERROR: build deps install failed.
    echo Most likely cause: no C/C++ compiler. Install MSVC Build Tools:
    echo     winget install Microsoft.VisualStudio.2022.BuildTools
    goto :end
)
echo       OK
echo.

echo [4/4] Installing madmom 0.16.1...
pip install madmom==0.16.1
if errorlevel 1 (
    echo ERROR: madmom install failed.
    goto :end
)
echo       OK
echo.

echo Smoke-testing the import...
python -c "from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor; print('madmom imports OK')"
if errorlevel 1 (
    echo ERROR: madmom imported but smoke test failed.
    goto :end
)

echo.
echo ================================================================
echo  SUCCESS. venv-chords is ready.
echo  The main pipeline will call it via subprocess.
echo ================================================================

:end
echo.
pause
endlocal
