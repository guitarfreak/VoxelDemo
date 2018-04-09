#pragma once

enum Keycode {
	KEYCODE_CTRL = 0,
	KEYCODE_CTRL_RIGHT,
	KEYCODE_SHIFT,
	KEYCODE_SHIFT_RIGHT,
	KEYCODE_ALT,
	KEYCODE_CAPS,
	KEYCODE_TAB,
	KEYCODE_SPACE,
	KEYCODE_RETURN,
	KEYCODE_ESCAPE,
	KEYCODE_BACKSPACE,
	KEYCODE_DEL,
	KEYCODE_HOME,
	KEYCODE_END,
	KEYCODE_PAGEUP,
	KEYCODE_PAGEDOWN,
	KEYCODE_UP,
	KEYCODE_DOWN,
	KEYCODE_LEFT,
	KEYCODE_RIGHT,

	KEYCODE_A,
	KEYCODE_B,
	KEYCODE_C,
	KEYCODE_D,
	KEYCODE_E,
	KEYCODE_F,
	KEYCODE_G,
	KEYCODE_H,
	KEYCODE_I,
	KEYCODE_J,
	KEYCODE_K,
	KEYCODE_L,
	KEYCODE_M,
	KEYCODE_N,
	KEYCODE_O,
	KEYCODE_P,
	KEYCODE_Q,
	KEYCODE_R,
	KEYCODE_S,
	KEYCODE_T,
	KEYCODE_U,
	KEYCODE_V,
	KEYCODE_W,
	KEYCODE_X,
	KEYCODE_Y,
	KEYCODE_Z,

	KEYCODE_0,
	KEYCODE_1,
	KEYCODE_2,
	KEYCODE_3,
	KEYCODE_4,
	KEYCODE_5,
	KEYCODE_6,
	KEYCODE_7,
	KEYCODE_8,
	KEYCODE_9,

	KEYCODE_F1,
	KEYCODE_F2,
	KEYCODE_F3,
	KEYCODE_F4,
	KEYCODE_F5,
	KEYCODE_F6,
	KEYCODE_F7,
	KEYCODE_F8,
	KEYCODE_F9,
	KEYCODE_F10,
	KEYCODE_F11,
	KEYCODE_F12,

	KEYCODE_COUNT,
};

struct Input {
	bool firstFrame;
	Vec2 mousePos;
	Vec2 mousePosNegative;
	Vec2 mousePosScreen;
	Vec2 mousePosNegativeScreen;

	Vec2 mousePosWindow;

	Vec2 mouseDelta;
	int mouseWheel;
	bool mouseButtonPressed[8];
	bool mouseButtonDown[8];
	bool mouseButtonReleased[8];
	bool doubleClick;
	Vec2 doubleClickPos;

	Vec2 lastMousePos;

	bool keysDown[KEYCODE_COUNT];
	bool keysPressed[KEYCODE_COUNT];
	char inputCharacters[32];
	int inputCharacterCount;

	bool anyKey;

	bool closeWindow;
	bool maximizeWindow;
	bool minimizeWindow;
	bool resize;

	bool altEnter;
};



#define WIN_KEY_NUMERIC_START 0x30
#define WIN_KEY_NUMERIC_END 0x39
#define WIN_KEY_LETTERS_START 0x41
#define WIN_KEY_LETTERS_END 0x5a
#define WIN_KEY_F_START 0x70
#define WIN_KEY_F_END 0x7B

int vkToKeycode(int vk) {

	switch(vk) {
		case VK_CONTROL: return KEYCODE_CTRL;
		case VK_RCONTROL: return KEYCODE_CTRL_RIGHT;
		case VK_SHIFT: return KEYCODE_SHIFT;
		case VK_RSHIFT: return KEYCODE_SHIFT_RIGHT;
		case VK_MENU: return KEYCODE_ALT;
		case VK_CAPITAL: return KEYCODE_CAPS;
		case VK_TAB: return KEYCODE_TAB;
		case VK_SPACE: return KEYCODE_SPACE;
		case VK_RETURN: return KEYCODE_RETURN;
		case VK_ESCAPE: return KEYCODE_ESCAPE;
		case VK_BACK: return KEYCODE_BACKSPACE;
		case VK_DELETE: return KEYCODE_DEL;
		case VK_HOME: return KEYCODE_HOME;
		case VK_END: return KEYCODE_END;
		case VK_PRIOR: return KEYCODE_PAGEUP;
		case VK_NEXT: return KEYCODE_PAGEDOWN;
		case VK_UP: return KEYCODE_UP;
		case VK_DOWN: return KEYCODE_DOWN;
		case VK_LEFT: return KEYCODE_LEFT;
		case VK_RIGHT: return KEYCODE_RIGHT;

		default: {
				 if(vk >= WIN_KEY_NUMERIC_START && vk <= WIN_KEY_NUMERIC_END) return KEYCODE_0 + vk - WIN_KEY_NUMERIC_START;
			else if(vk >= WIN_KEY_LETTERS_START && vk <= WIN_KEY_LETTERS_END) return KEYCODE_A + vk - WIN_KEY_LETTERS_START;
			else if(vk >= WIN_KEY_F_START 		&& vk <= WIN_KEY_F_END) 	  return KEYCODE_F1 + vk - WIN_KEY_F_START;
		}
	}

	return -1;
}

// int keycode = vkToKeycode(VK_UP);

void initInput(Input* input) {
    *input = {};

    input->firstFrame = true;

}

enum RequestType {
	REQUEST_TYPE_GET = 0,
	REQUEST_TYPE_POST,
	REQUEST_TYPE_COUNT,
};

struct HttpRequest {
	char* link;
	RequestType type;
	char* additionalBodyContent;

	char* contentBuffer;
	char* contentFile;
	char* contentFilePath;
	char* headerResponseFile;

	bool finished;
	int size;
	float progress;

	bool stopProcess;
};

enum ShellCommand {
	SHELLCOMMAND_FILE,
	SHELLCOMMAND_PATH,
	SHELLCOMMAND_URL,
	SHELLCOMMAND_COUNT,
};


// void atomicAdd(unsigned int* n) {
void atomicAdd(volatile unsigned int* n) {
	InterlockedIncrement(n);
}

void atomicSub(volatile unsigned int* n) {
	InterlockedDecrement(n);
}

struct ThreadQueue;
// struct PlatformData {
// 	int windowX, windowY;
// 	int windowWidth, windowHeight;
// 	int viewportWidth, viewportHeight;
// 	ThreadQueue* highQueue;
// 	ThreadQueue* lowQueue;

// 	// PlatformFunctions functions;
// };

int getThreadId() {
	int tId = GetCurrentThreadId();
	return tId;
}

// void platformDataInit(PlatformData* pd) {
	// *pd = {};
// }

#include <winsock2.h>
#include <WS2tcpip.h>
#include <Shellapi.h>
#include <Dwmapi.h>

struct SystemData {
	WindowsData windowsData;
	HINSTANCE instance;
	HDC deviceContext;
	HWND windowHandle;
	
	// 1. Misc, 2. CubeMaps, 3. Minecraft
	HANDLE folderHandles[3]; 

	//

	Input* input;
	void* mainFiber;
	void* messageFiber;

	int coreCount;
	int fontHeight;

	bool maximized;
	bool killedFocus;
	bool setFocus;
	bool windowIsFocused;

	bool vsyncTempTurnOff;
};

void systemDataInit(SystemData* sd, HINSTANCE instance) {
	sd->instance = instance;
}

bool mouseInClientArea(HWND windowHandle) {
	POINT point;    
	GetCursorPos(&point);
	ScreenToClient(windowHandle, &point);

	Vec2i mp = vec2i(point.x, point.y);

	RECT cr; 
	GetClientRect(windowHandle, &cr);
	bool result = (mp.x >= cr.left && mp.x < cr.right && 
				   mp.y >= cr.top  && mp.y < cr.bottom);

	return result;
}

LRESULT CALLBACK mainWindowCallBack(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {

	SystemData* sd = (SystemData*)GetWindowLongPtrA(window, GWLP_USERDATA);

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

        // #ifdef ENABLE_CUSTOM_WINDOW_FRAME
        // case WM_NCACTIVATE: {
        // 	sd->vsyncTempTurnOff = true;
        // 	SwitchToFiber(sd->mainFiber);
        // } break;
        // #endif

        case WM_SIZE: {
        	if(wParam == SIZE_MAXIMIZED) sd->maximized = true;
        	else if(wParam == SIZE_RESTORED) sd->maximized = false;

        	// sd->vsyncTempTurnOff = true;
        	sd->input->resize = true;
        } break;

        // case WM_NCPAINT: {
	       //  HDC hdc;
	       //  hdc = GetDCEx(window, (HRGN)wParam, DCX_WINDOW|DCX_INTERSECTRGN);
	       //  // Paint into this DC 
	       //  ReleaseDC(window, hdc);

	       //  // sd->vsyncTempTurnOff = true;
	       //  // SwitchToFiber(sd->mainFiber);

        // 	return 0;
        // } break;

        case WM_PAINT: {
        	PAINTSTRUCT ps;
        	HDC hdc = BeginPaint(window, &ps); 
        	EndPaint(window, &ps);

        	sd->vsyncTempTurnOff = true;
        	SwitchToFiber(sd->mainFiber);

        	return 0;
        } break;

        case WM_SETFOCUS: {
        	sd->setFocus = true;
        	sd->windowIsFocused = true;
        	// sd->vsyncTempTurnOff = true;
        	// SwitchToFiber(sd->mainFiber);
        } break;

        case WM_KILLFOCUS: {
		    // PostMessage(window, message, wParam, lParam);
		    sd->killedFocus = true;
        	sd->windowIsFocused = false;

        	sd->vsyncTempTurnOff = true;
        	SwitchToFiber(sd->mainFiber);
        } break;

        case WM_TIMER: {
        	sd->vsyncTempTurnOff = true;
        	SwitchToFiber(sd->mainFiber);
        } break;

        // Make alt+enter not beep....
        case WM_MENUCHAR: {
            if(LOWORD(wParam) & VK_RETURN) 
            	return MAKELRESULT(0, MNC_CLOSE);
            return DefWindowProc(window, message, wParam, lParam);
        } break;

        default: {
            return DefWindowProc(window, message, wParam, lParam);
        } break;
    }

    return 1;
}

struct MonitorData {
	Rect fullRect;
	Rect workRect;
	HMONITOR handle;
};

struct WindowSettings {
	Vec2i res;
	Vec2i fullRes;
	bool fullscreen;
	uint style;
	WINDOWPLACEMENT g_wpPrev;
	Rect previousWindowRect;

	MonitorData monitors[3];
	int monitorCount;
	Vec2i biggestMonitorSize;
	int refreshRate;

	Vec2i currentRes;
	float aspectRatio;
	float windowScale;

	bool dontUpdateCursor;
	bool customCursor;
	POINT lastMousePosition;

	bool vsync;
	int frameRate;
};

void updateCursor(WindowSettings* ws) {
	if(!ws->customCursor) {
		SetCursor(LoadCursor(0, IDC_ARROW));
	}
	ws->customCursor = false;
}

void setCursor(WindowSettings* ws, LPCSTR type) {
	SetCursor(LoadCursor(0, type));
	ws->customCursor = true;
}

void showCursor(bool show) {
	if(show) {
		while(ShowCursor(true) < 0) {};
	} else {
		while(ShowCursor(false) >= 0) {};
	}
}

void makeWindowTopmost(SystemData* sd) {
    SetWindowPos(sd->windowHandle, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
}

void showWindow(HWND windowHandle) {
    ShowWindow(windowHandle, SW_SHOW);
}

BOOL CALLBACK monitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {

	MONITORINFO mi = { sizeof(MONITORINFO) };
	GetMonitorInfo(hMonitor, &mi);

	WindowSettings* ws = (WindowSettings*)(dwData);
	MonitorData* md = ws->monitors + ws->monitorCount;
	md->fullRect = rect(mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom);
	md->workRect = rect(mi.rcWork.left, mi.rcWork.top, mi.rcWork.right, mi.rcWork.bottom);
	md->handle = hMonitor;
	ws->monitorCount++;

	return true;
}

Vec2 getMousePos(HWND windowHandle, bool yInverted = true) {
	POINT point;    
	GetCursorPos(&point);
	ScreenToClient(windowHandle, &point);
	Vec2 mousePos = vec2(0,0);
	mousePos.x = point.x;
	mousePos.y = point.y;
	if(yInverted) mousePos.y = -mousePos.y;

	return mousePos;
}

Vec2 getMousePosS(bool yInverted = true) {
	POINT point;    
	GetCursorPos(&point);
	Vec2 mousePos = vec2(0,0);
	mousePos.x = point.x;
	mousePos.y = point.y;
	if(yInverted) mousePos.y = -mousePos.y;

	return mousePos;
}

void inputPrepare(Input* input) {
	input->anyKey = false;
    input->mouseWheel = 0;
    for(int i = 0; i < arrayCount(input->mouseButtonPressed); i++) input->mouseButtonPressed[i] = 0;
    for(int i = 0; i < arrayCount(input->mouseButtonReleased); i++) input->mouseButtonReleased[i] = 0;
    for(int i = 0; i < arrayCount(input->keysPressed); i++) input->keysPressed[i] = 0;
    input->inputCharacterCount = 0;
    input->mouseDelta = vec2(0,0);

    input->doubleClick = false;

    input->closeWindow = false;
	input->maximizeWindow = false;
	input->minimizeWindow = false;

	input->altEnter = false;
}

void CALLBACK updateInput(SystemData* sd) {
	for(;;) {

		Input* input = sd->input;
		HWND windowHandle = sd->windowHandle;

		SetTimer(windowHandle, 1, 1, 0);

	    bool mouseInClient = mouseInClientArea(windowHandle);

	    MSG message;
	    while(PeekMessage(&message, windowHandle, 0, 0, PM_REMOVE)) {
	        switch(message.message) {
		        case WM_LBUTTONDBLCLK: {
		        	input->doubleClick = true;
					input->doubleClickPos = getMousePos(windowHandle, true);
		        } break;

	            case WM_KEYDOWN:
	            case WM_KEYUP: {
	                uint vk = uint(message.wParam);

	                bool keyDown = (message.message == WM_KEYDOWN);
	                int keycode = vkToKeycode(vk);
	                input->keysDown[keycode] = keyDown;
	                input->keysPressed[keycode] = keyDown;
	                // input->mShift = ((GetKeyState(VK_SHIFT) & 0x80) != 0);
	                // input->mCtrl = ((GetKeyState(VK_CONTROL) & 0x80) != 0);
	                bool alt = ((GetKeyState(VK_MENU) & 0x80) != 0);
	                if(keyDown && keycode == KEYCODE_RETURN && alt) {
	                	input->altEnter = true;
	                }

	                if(keyDown) {
	                	input->anyKey = true;
	                }

	                TranslateMessage(&message); 
	                DispatchMessage(&message); 
	            } break;

	            case WM_SYSKEYDOWN:
	            case WM_SYSKEYUP: {
	                uint vk = uint(message.wParam);
	            	bool keyDown = (message.message == WM_SYSKEYDOWN);

	            	if(keyDown) {
		            	if(vk == VK_RETURN) {
		            		input->altEnter = true;
		            	}
	            	}

	            	TranslateMessage(&message); 
	            	DispatchMessage(&message); 
	            };

	            case WM_CHAR: {
	                // input->inputCharacters[input->inputCharacterCount] = (char)uint(message.wParam);
	            	uint charIndex = uint(message.wParam);
	            	if(charIndex < ' ' || charIndex > '~') break;
	            	char c = (char)charIndex;
	                input->inputCharacters[input->inputCharacterCount] = c;
	                input->inputCharacterCount++;
	            } break;

	            case WM_INPUT: {
	            	RAWINPUT inputBuffer;
	            	UINT rawInputSize = sizeof(inputBuffer);
	            	GetRawInputData((HRAWINPUT)(message.lParam), RID_INPUT, &inputBuffer, &rawInputSize, sizeof(RAWINPUTHEADER));
	            	RAWINPUT* raw = (RAWINPUT*)(&inputBuffer);
	            	
	            	if (raw->header.dwType == RIM_TYPEMOUSE && raw->data.mouse.usFlags == MOUSE_MOVE_RELATIVE) {

	            	    input->mouseDelta += vec2(raw->data.mouse.lLastX, raw->data.mouse.lLastY);

	            	    USHORT buttonFlags = raw->data.mouse.usButtonFlags;

	            	    if(mouseInClient) {
							if(buttonFlags & RI_MOUSE_LEFT_BUTTON_DOWN) {
								// SetCapture(windowHandle);
								input->mouseButtonPressed[0] = true; 
								input->mouseButtonDown[0] = true; 
							} else if(buttonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN) {
								// SetCapture(windowHandle);
								input->mouseButtonPressed[1] = true; 
								input->mouseButtonDown[1] = true; 
							} else if(buttonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN) {
								// SetCapture(windowHandle);
								input->mouseButtonPressed[2] = true; 
								input->mouseButtonDown[2] = true; 
							} else if(buttonFlags & RI_MOUSE_WHEEL) {
								input->mouseWheel += ((SHORT)raw->data.mouse.usButtonData) / WHEEL_DELTA;
							}
	            	    }

	            	    if(buttonFlags & RI_MOUSE_LEFT_BUTTON_UP) {
	            	    	// ReleaseCapture();
	            	    	input->mouseButtonDown[0] = false; 
	            	    	input->mouseButtonReleased[0] = true; 
	            	    } else if(buttonFlags & RI_MOUSE_RIGHT_BUTTON_UP) {
	            	    	// ReleaseCapture();
	            	    	input->mouseButtonDown[1] = false; 
	            	    	input->mouseButtonReleased[1] = true; 
	            	    } else if(buttonFlags & RI_MOUSE_MIDDLE_BUTTON_UP) {
	            	    	// ReleaseCapture();
	            	    	input->mouseButtonDown[2] = false; 
	            	    	input->mouseButtonReleased[2] = true; 
	            	    }

	            	} break;

	            	TranslateMessage(&message);
	            	DispatchMessage(&message);
	            } break;

	            case WM_DESTROY: 
	            case WM_CLOSE: 
	            case WM_QUIT: 
	            	input->closeWindow = true;
	            	break;

	            default: {
	                TranslateMessage(&message); 
	                DispatchMessage(&message); 
	            } break;
	        }
	    }

	    if(!sd->windowIsFocused) {
	    	for(int i = 0; i < arrayCount(input->mouseButtonPressed); i++) input->mouseButtonPressed[i] = 0;
	    	for(int i = 0; i < arrayCount(input->mouseButtonReleased); i++) input->mouseButtonReleased[i] = 0;
	    	input->mouseWheel = 0;
	    }
	    sd->setFocus = false;

	    // In case we clear because of focus.
	    bool closeWindowTemp = input->closeWindow;

	    if(sd->killedFocus) {
	    	for(int i = 0; i < KEYCODE_COUNT; i++) {
	    		input->keysDown[i] = false;
	    	}
	    	*input = {};

	    	for(int i = 0; i < arrayCount(input->mouseButtonReleased); i++) {
		    	input->mouseButtonReleased[i] = true;
	    	}

	    	sd->killedFocus = false;
	    }

	    if(input->altEnter) {
	    	input->keysPressed[KEYCODE_RETURN] = false;
	    }

	    input->closeWindow = closeWindowTemp;

	    input->mousePos = getMousePos(windowHandle, false);
	    input->mousePosNegative = getMousePos(windowHandle, true);

	    input->mousePosScreen = getMousePosS(false);
	    input->mousePosNegativeScreen = getMousePosS(true);

	    input->lastMousePos = input->mousePos;

	    input->firstFrame = false;

	    SwitchToFiber(sd->mainFiber);
	}
}


#include <Mmsystem.h>

int getSystemFontHeight(HWND windowHandle);
void initSystem(SystemData* systemData, WindowSettings* ws, WindowsData wData, Vec2i res, int style, int , int monitor = 0) {
	systemData->windowsData = wData;

	EnumDisplayMonitors(0, 0, monitorEnumProc, ((LPARAM)ws));

	DEVMODE devMode;
	EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &devMode);
	ws->refreshRate = devMode.dmDisplayFrequency;


	ws->currentRes = res;
	ws->fullscreen = false;
	ws->aspectRatio = (float)res.w / (float)res.h;

	ws->style = style;

	RECT cr = {0, 0, res.w, res.h};
	AdjustWindowRectEx(&cr, ws->style, 0, 0);

	int ww = cr.right - cr.left;
	int wh = cr.bottom - cr.top;
	int wx, wy;
	{
		MonitorData* md = ws->monitors + monitor;
		wx = rectCen(md->workRect).x - res.w/2;
		wy = rectCen(md->workRect).y - res.h/2;
	}
	ws->res = vec2i(ww, wh);

    WNDCLASS windowClass = {};
    windowClass.style = CS_OWNDC|CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS;
     
    windowClass.lpfnWndProc = mainWindowCallBack;
    windowClass.hInstance = systemData->instance;
    windowClass.lpszClassName = "App";

    if(!RegisterClass(&windowClass)) {
        DWORD errorCode = GetLastError();
        int dummy = 2;   
    }

    // systemData->windowClass = windowClass;
    systemData->windowHandle = CreateWindowEx(0, windowClass.lpszClassName, "", ws->style, wx,wy,ww,wh, 0, 0, systemData->instance, 0);

    HWND windowHandle = systemData->windowHandle;
    if(!windowHandle) {
        DWORD errorCode = GetLastError();
    }

    SetWindowLongPtr(windowHandle, GWLP_USERDATA, (LONG_PTR)systemData);

    PIXELFORMATDESCRIPTOR pixelFormatDescriptor =
    {
        +    sizeof(PIXELFORMATDESCRIPTOR),
        1,
        /*PFD_SUPPORT_COMPOSITION |*/ PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    //Flags
        PFD_TYPE_RGBA,            //The kind of framebuffer. RGBA or palette.
        24,                        //Colordepth of the framebuffer.
        0, 0, 0, 0, 0, 0,
        0, //8
        0,
        0,
        0, 0, 0, 0,
        24,                        //Number of bits for the depthbuffer
        // 32,                        //Number of bits for the depthbuffer
        // 0,                        //Number of bits for the depthbuffer
        // 8,                        //Number of bits for the stencilbuffer
        0,                        //Number of bits for the stencilbuffer
        0,                        //Number of Aux buffers in the framebuffer.
        PFD_MAIN_PLANE,
        0,
        0, 0, 0
    };    
    
    HDC deviceContext = GetDC(windowHandle);
    systemData->deviceContext = deviceContext;
    int pixelFormat;
    pixelFormat = ChoosePixelFormat(deviceContext, &pixelFormatDescriptor);
	SetPixelFormat(deviceContext, pixelFormat, &pixelFormatDescriptor);
	
    HGLRC openglContext = wglCreateContext(systemData->deviceContext);
    bool result = wglMakeCurrent(systemData->deviceContext, openglContext);
    if(!result) { printf("Could not set Opengl Context.\n"); }

    #ifndef HID_USAGE_PAGE_GENERIC
    #define HID_USAGE_PAGE_GENERIC         ((USHORT) 0x01)
    #endif
    #ifndef HID_USAGE_GENERIC_MOUSE
    #define HID_USAGE_GENERIC_MOUSE        ((USHORT) 0x02)
    #endif

    RAWINPUTDEVICE Rid[1];
    Rid[0].usUsagePage = HID_USAGE_PAGE_GENERIC; 
    Rid[0].usUsage = HID_USAGE_GENERIC_MOUSE; 
    Rid[0].hwndTarget = windowHandle;
    Rid[0].dwFlags = RIDEV_INPUTSINK;   
    // Rid[0].dwFlags = 0;   
    bool r = RegisterRawInputDevices(Rid, 1, sizeof(Rid[0]));
    assert(r);

    systemData->mainFiber = ConvertThreadToFiber(0);
    systemData->messageFiber = CreateFiber(0, (PFIBER_START_ROUTINE)updateInput, systemData);

    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    systemData->coreCount = sysinfo.dwNumberOfProcessors;

    // Set icon.
    {
    	char* rs = MAKEINTRESOURCE(1);

    	HANDLE hbicon = LoadImage(GetModuleHandle(0), rs, IMAGE_ICON, GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), 0);
    	if(hbicon) SendMessage(windowHandle, WM_SETICON, ICON_BIG, (LPARAM)hbicon);

    	HANDLE hsicon = LoadImage(GetModuleHandle(0), rs, IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0);
    	if(hsicon) SendMessage(windowHandle, WM_SETICON, ICON_SMALL, (LPARAM)hsicon);
    }

    // Set minimal sleep timer resolution.
    {
    	TIMECAPS timecaps;
    	timeGetDevCaps(&timecaps, sizeof(TIMECAPS));
    	int error = timeBeginPeriod(timecaps.wPeriodMin);
    	if(error != TIMERR_NOERROR) printf("Timer error.\n");
    }

	systemData->fontHeight = getSystemFontHeight(windowHandle);
	
	SetFocus(windowHandle);
	systemData->windowIsFocused = true;
}




// MetaPlatformFunction();
// const char* getClipboard(MemoryBlock* memory) {
//     BOOL result = OpenClipboard(0);
//     HANDLE clipBoardData = GetClipboardData(CF_TEXT);

//     return (char*)clipBoardData;
// }

const char* getClipboard() {
    BOOL result = OpenClipboard(0);
    HANDLE clipBoardData = GetClipboardData(CF_TEXT);

    return (char*)clipBoardData;
}

// MetaPlatformFunction();
void setClipboard(char* text) {
    int textSize = strLen(text) + 1;
    HANDLE clipHandle = GlobalAlloc(GMEM_MOVEABLE | GMEM_DDESHARE, textSize);
    char* pointer = (char*)GlobalLock(clipHandle);
    memCpy(pointer, (char*)text, textSize);
    GlobalUnlock(clipHandle);

    OpenClipboard(0);
    EmptyClipboard();
    SetClipboardData(CF_TEXT, clipHandle);
    CloseClipboard();
}

Rect getWindowWindowRect(HWND windowHandle) {
	RECT r; 
	GetWindowRect(windowHandle, &r);
	Rect windowRect = rect(r.left, r.bottom, r.right, r.top);
	
	return windowRect;
}

void getWindowProperties(HWND windowHandle, int* viewWidth, int* viewHeight, int* width, int* height, int* x, int* y) {
    RECT cr; 
    GetClientRect(windowHandle, &cr);
    *viewWidth = cr.right - cr.left;
    *viewHeight = cr.bottom - cr.top;

    if(width && height) {
    	RECT wr; 
    	GetWindowRect(windowHandle, &wr);
    	*width = wr.right - wr.left;
    	*height = wr.bottom - wr.top;
    }

    if(x && y) {
    	WINDOWPLACEMENT placement;
    	GetWindowPlacement(windowHandle, &placement);
    	RECT r; 
    	r = placement.rcNormalPosition; 
    	*x = r.left;
    	*y = r.top;    	
    }
}

// MetaPlatformFunction();
void setWindowProperties(HWND windowHandle, int width, int height, int x, int y) {
    WINDOWPLACEMENT placement;
    GetWindowPlacement(windowHandle, &placement);
    RECT r = placement.rcNormalPosition;

    if(width != -1) r.right = r.left + width;
    if(height != -1) r.bottom = r.top + height;
    if(x != -1) {
        int width = r.right - r.left;
        r.left = x;
        r.right = x + width;
    }
    if(y != -1) {
        int height = r.bottom - r.top;
        r.top = y;
        r.bottom = y + height;
    }

    placement.rcNormalPosition = r;
    SetWindowPlacement(windowHandle, &placement);
}

enum WindowMode {
	WINDOW_MODE_WINDOWED = 0,
	WINDOW_MODE_FULLBORDERLESS,

	WINDOW_MODE_COUNT,
};

void setWindowStyle(HWND hwnd, DWORD dwStyle) {
	SetWindowLong(hwnd, GWL_STYLE, dwStyle);
}

DWORD getWindowStyle(HWND hwnd) {
	return GetWindowLong(hwnd, GWL_STYLE);
}

void updateResolution(HWND windowHandle, WindowSettings* ws) {
	getWindowProperties(windowHandle, &ws->currentRes.x, &ws->currentRes.y,0,0,0,0);
	ws->aspectRatio = ws->currentRes.x / (float)ws->currentRes.y;

	{
		MONITORINFO monitorInfo;
		monitorInfo.cbSize = sizeof(MONITORINFO);
		bool result = GetMonitorInfo(MonitorFromWindow(windowHandle, MONITOR_DEFAULTTONEAREST), &monitorInfo);
		RECT rWork = monitorInfo.rcMonitor;
		Vec2i monitorRes = vec2i(rWork.right - rWork.left, rWork.bottom - rWork.top);

		ws->windowScale = (float)ws->currentRes.h / monitorRes.h;
	}
}

void setWindowMode(HWND hwnd, WindowSettings* wSettings, int mode) {
	if(mode == WINDOW_MODE_FULLBORDERLESS && !wSettings->fullscreen) {
		wSettings->previousWindowRect = getWindowWindowRect(hwnd);
		
		wSettings->g_wpPrev = {};

		DWORD dwStyle = getWindowStyle(hwnd);
		if (dwStyle & WS_OVERLAPPEDWINDOW) {
		  MONITORINFO mi = { sizeof(mi) };
		  if (GetWindowPlacement(hwnd, &wSettings->g_wpPrev) &&
		      GetMonitorInfo(MonitorFromWindow(hwnd,
		                     MONITOR_DEFAULTTOPRIMARY), &mi)) {
		    SetWindowLong(hwnd, GWL_STYLE,
		                  dwStyle & ~WS_OVERLAPPEDWINDOW);
			setWindowStyle(hwnd, dwStyle & ~WS_OVERLAPPEDWINDOW);

		    SetWindowPos(hwnd, HWND_TOP,
		                 mi.rcMonitor.left, mi.rcMonitor.top,
		                 mi.rcMonitor.right - mi.rcMonitor.left,
		                 mi.rcMonitor.bottom - mi.rcMonitor.top,
		                 SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		  }
		}

		wSettings->fullscreen = true;

	} else if(mode == WINDOW_MODE_WINDOWED && wSettings->fullscreen) {
		setWindowStyle(hwnd, wSettings->style);
		SetWindowPlacement(hwnd, &wSettings->g_wpPrev);

		wSettings->fullscreen = false;

		InvalidateRect(NULL, NULL, FALSE);
	}
}

void swapBuffers(SystemData* systemData) {
    SwapBuffers(systemData->deviceContext);
}

void captureMouse(HWND windowHandle, bool t) {
	if(t) {
		int w,h;
		Vec2i wPos;
		getWindowProperties(windowHandle, &w, &h, 0, 0, &wPos.x, &wPos.y);
		SetCursorPos(wPos.x + w/2, wPos.y + h/2);

		while(ShowCursor(false) >= 0);
	} else {
		while(ShowCursor(true) < 0);
	}
}

bool windowHasFocus(HWND windowHandle) {
	bool result = GetFocus() == windowHandle;
	return result;
}

bool windowSizeChanged(HWND windowHandle, WindowSettings* ws) {
	Vec2i cr;
	getWindowProperties(windowHandle, &cr.x, &cr.y, 0, 0, 0, 0);

	bool result = cr != ws->currentRes;
	return result;
}

int getSystemFontHeight(HWND windowHandle) {
	HDC dc = GetDC(windowHandle);

	TEXTMETRIC textMetric;
	GetTextMetrics(dc, &textMetric);

	return textMetric.tmHeight;
}

// MetaPlatformFunction();
uint getTicks() {
    uint result = GetTickCount();

    return result;
}

__int64 getCycleStamp() {
	return __rdtsc();
}

struct MSTimer {
	double frequency;
	LARGE_INTEGER timeStamp;

	double dt;
};

void timerInit(MSTimer* timer) {
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency); 

	timer->frequency = (double)frequency.QuadPart;
}

void timerStart(MSTimer* timer) {
	QueryPerformanceCounter(&timer->timeStamp);
}

double timerStop(MSTimer* timer) {	
	LARGE_INTEGER newTimeStamp;
	QueryPerformanceCounter(&newTimeStamp);

	timer->dt = newTimeStamp.QuadPart - timer->timeStamp.QuadPart;

	// In seconds.
	timer->dt /= timer->frequency;

	return timer->dt;
}

double timerUpdate(MSTimer* timer) {	
	double time = timerStop(timer);
	timerStart(timer);

	return time;
}

// i64 timerInit() {
// 	LARGE_INTEGER counter;
// 	QueryPerformanceCounter(&counter);

// 	return counter.QuadPart;
// }

// // Returns time in milliseconds;
// f64 timerUpdate(i64 lastTimeStamp, i64* setTimeStamp = 0) {
// 	LARGE_INTEGER counter;
// 	LARGE_INTEGER frequency;
// 	QueryPerformanceFrequency(&frequency); 
// 	QueryPerformanceCounter(&counter);

// 	i64 timeStamp = counter.QuadPart;
// 	f64 dt = (timeStamp - lastTimeStamp);
// 	dt *= 1000000;
// 	dt /= frequency.QuadPart;
// 	dt /= 1000000;
	
// 	if(setTimeStamp) *setTimeStamp = timeStamp;

// 	return dt;
// }



// MetaPlatformFunction();
// void shellExecute(MemoryBlock* memory, char* pathOrFile, int shellCommand) {
//     char* command = "xdg-open ";
//     char* totalCommand = getTArray(memory, char, strLen(command) + strLen(pathOrFile));
//     totalCommand[0] = '\0';
//     strAppend(totalCommand, command);
//     strAppend(totalCommand, pathOrFile);

//     // printf("%s \n", totalCommand);
//     system(totalCommand);
// }

void shellExecute(char* command) {
	system(command);
}

bool windowIsMinimized(HWND windowHandle) {
	return IsIconic(windowHandle);
}

void shellExecuteNoWindow(char* command) {
	STARTUPINFO si = {};
	PROCESS_INFORMATION pi = {};
	si.cb = sizeof(si);

	if (CreateProcess(NULL, command, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
	    WaitForSingleObject(pi.hProcess, INFINITE);
	    CloseHandle(pi.hProcess);
	    CloseHandle(pi.hThread);
	}
}

// MetaPlatformFunction();
void sleep(int milliseconds) {
    Sleep(milliseconds);
}


struct FolderSearchData {
	WIN32_FIND_DATA findData;
	HANDLE folderHandle;

	char* fileName;
};

bool folderSearchStart(FolderSearchData* fd, char* folder) {	
	// Remember, for searching folder add "*" at the end of path

	fd->folderHandle = FindFirstFile(folder, &fd->findData);

	if(fd->folderHandle != INVALID_HANDLE_VALUE) return true;
	else return false;
}

bool folderSearchNextFile(FolderSearchData* fd) {
	if(FindNextFile(fd->folderHandle, &fd->findData) == 0) return false;

	if(strLen(fd->findData.cFileName) <= 2) {
		return folderSearchNextFile(fd); // Skip ".."
	}

	fd->fileName = fd->findData.cFileName;

	return true;
}