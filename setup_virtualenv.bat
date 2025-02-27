@echo off
:: Create a new virtual environment in the 'myenv' directory
python -m venv myenv

:: Activate the virtual environment
call myenv\Scripts\activate

:: Upgrade pip to the latest version
python -m pip install --upgrade pip

:: Install required packages from requirements.txt
pip install -r requirements.txt

:: Deactivate the virtual environment
deactivate

echo Virtual environment setup complete.
