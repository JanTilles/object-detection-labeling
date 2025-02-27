@echo off
:: Create a new virtual environment in the 'myenv' directory
python -m venv myenv

:: Activate the virtual environment
call myenv\Scripts\activate

:: Upgrade pip to the latest version
python -m pip install --upgrade pip

:: Install required packages
pip install ultralytics opencv-python-headless pyyaml

:: Deactivate the virtual environment
deactivate

echo Virtual environment setup complete.
