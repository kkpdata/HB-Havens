@echo off
REM Installation location of activate.bat may vary across systems. 
REM Please update the line below according to your specific installation.
set PATH=%PATH%;c:\ProgramData\miniconda3\Scripts\;
REM set PATH=%PATH%;c:\Anaconda3\Scripts\;

call conda config --add channels conda-forge
call conda config --set channel_priority strict 
call conda env create -f conda_HBH.yml
call activate HBH

REM REM Om deze omgeving weer te verwijderen:
REM conda remove --name HBH --all


