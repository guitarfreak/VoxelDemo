#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "rt_misc.h"
#include "rt_hotload.h"


#include "rt_misc_win32.h"

MemoryBlock* globalMemory;

LRESULT CALLBACK mainWindowCallBack(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    switch(message) {
        case WM_DESTROY: {
            PostMessage(window, message, wParam, lParam);
        } break;

        case WM_CLOSE: {
            PostMessage(window, message, wParam, lParam);
        } break;

        case WM_QUIT: {
            PostMessage(window, message, wParam, lParam);
        } break;

        default: {
            return DefWindowProc(window, message, wParam, lParam);
        } break;
    }

    return 1;
}

int CALLBACK WinMain(HINSTANCE instance, HINSTANCE prevInstance, LPSTR commandLine, int showCode) {

	globalMemory = (MemoryBlock*)malloc(sizeof(MemoryBlock));
	// initMemorySizes(globalMemory, megaBytes(100), megaBytes(100), megaBytes(100), kiloBytes(10));
	// initMemorySizes(globalMemory, megaBytes(500), megaBytes(100), megaBytes(100), kiloBytes(10));
	// initMemorySizes(globalMemory, megaBytes(2000), megaBytes(100), megaBytes(100), kiloBytes(10));
	// initMemorySizes(globalMemory, megaBytes(1000), megaBytes(100), megaBytes(100), kiloBytes(10));
	initMemorySizes(globalMemory, megaBytes(1500), megaBytes(10), megaBytes(100), kiloBytes(10));
	initMemory(globalMemory);

	HotloadDll hotloadDll;
	initDll(&hotloadDll, "app.dll", "appTemp.dll", "lock.tmp");

	WindowsData wData = windowsData(instance, prevInstance, commandLine, showCode);

	ThreadQueue threadQueue;
	threadInit(&threadQueue, 7);

    bool firstFrame = true;
    bool secondFrame = false;
    bool isRunning = true;
    while(isRunning) {
    	bool reload = false;
    	if(threadQueue.completionCount == threadQueue.completionGoal)
    		reload = updateDll(&hotloadDll);
        // if(reload) Sleep(1000);
     	platform_appMain = (appMainType*)getDllFunction(&hotloadDll, "appMain");
        platform_appMain(firstFrame, secondFrame, reload, &isRunning, globalMemory, wData, mainWindowCallBack, &threadQueue);

        secondFrame = false;
        if(firstFrame) secondFrame = true;
        firstFrame = false;
    }

	return 0;
}