@echo off
REM Installation location of activate.bat may vary across systems. 
REM Please update the line below according to your specific installation.
REM set PATH=%PATH%;c:\ProgramData\miniconda3\Scripts\;
set PATH=%PATH%;C:\Users\Benit\miniforge3\Scripts\;
REM set PATH=%PATH%;c:\Anaconda3\Scripts\;

call activate HBH
python bin\main_gui.py ""
pause