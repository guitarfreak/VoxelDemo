@echo off

call ".\code\buildWin.bat" -x64 -noRun -release -ship -fullOptimize
call ".\code\buildWin.bat" -x86 -noRun -release -ship -fullOptimize
