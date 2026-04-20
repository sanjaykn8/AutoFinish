@echo off
cd /d %~dp0\backend
python train.py --data ..\data\Shakespeare.csv
python app.py
