#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "rt_misc.h"
#include "rt_hotload.h"

MemoryBlock* globalMemory;

int CALLBACK WinMain(HINSTANCE instance, HINSTANCE prevInstance, LPSTR commandLine, int showCode) {

	globalMemory = (MemoryBlock*)malloc(sizeof(MemoryBlock));
	initMemorySizes(globalMemory, megaBytes(100), megaBytes(100), megaBytes(100), kiloBytes(10));
	initMemory(globalMemory);

	HotloadDll hotloadDll;
	initDll(&hotloadDll, "app.dll", "appTemp.dll", "lock.tmp");

	WindowsData wData = windowsData(instance, prevInstance, commandLine, showCode);

    bool firstFrame = true;
    bool secondFrame = false;
    bool isRunning = true;
    while(isRunning) {
        bool reload = updateDll(&hotloadDll);
     	platform_appMain = (appMainType*)getDllFunction(&hotloadDll, "appMain");
        platform_appMain(firstFrame, secondFrame, reload, &isRunning, globalMemory, wData);

        secondFrame = false;
        if(firstFrame) secondFrame = true;
        firstFrame = false;
    }

	return 0;
}