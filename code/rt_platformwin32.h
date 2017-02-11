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
	int mouseDeltaX, mouseDeltaY;
	int mouseWheel;
	bool mouseButtonPressed[8];
	bool mouseButtonDown[8];

	bool keysDown[KEYCODE_COUNT];
	bool keysPressed[KEYCODE_COUNT];
	char inputCharacters[32];
	int inputCharacterCount;

	bool mShift, mCtrl, mAlt;
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
	// LONG __cdecl InterlockedIncrement(
	//   _Inout_ LONG volatile *Addend
	// );

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

	HANDLE folderHandles[3]; // 1. Misc, 2. CubeMaps, 3. Minecraft
};

void systemDataInit(SystemData* sd, HINSTANCE instance) {
	sd->instance = instance;
}

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

void initSystem(SystemData* systemData, WindowsData wData, uint style, int x, int y, int w, int h) {
	systemData->windowsData = wData;

    WNDCLASS windowClass = {};
    windowClass.style = CS_OWNDC|CS_HREDRAW|CS_VREDRAW;
    windowClass.lpfnWndProc = mainWindowCallBack;
    windowClass.hInstance = systemData->instance;
    windowClass.lpszClassName = "App";
    windowClass.hCursor = LoadCursor(0, IDC_ARROW);

    if(!RegisterClass(&windowClass)) {
        DWORD errorCode = GetLastError();
        int dummy = 2;   
    }

    if(!style) style = WS_OVERLAPPEDWINDOW;
    // if(!style) style = WS_VISIBLE;
    if(!x) x = CW_USEDEFAULT;
    if(!y) y = CW_USEDEFAULT;
    if(!w) w = CW_USEDEFAULT;
    if(!h) h = CW_USEDEFAULT;
    systemData->windowHandle = CreateWindowEx(0, windowClass.lpszClassName, "", style, x,y,w,h, 0, 0, systemData->instance, 0);

    if(!systemData->windowHandle) {
        DWORD errorCode = GetLastError();
    }

    PIXELFORMATDESCRIPTOR pixelFormatDescriptor =
    {
        +    sizeof(PIXELFORMATDESCRIPTOR),
        1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    //Flags
        PFD_TYPE_RGBA,            //The kind of framebuffer. RGBA or palette.
        32,                        //Colordepth of the framebuffer.
        0, 0, 0, 0, 0, 0,
        0, //8
        0,
        0,
        0, 0, 0, 0,
        24,                        //Number of bits for the depthbuffer
        // 0,                        //Number of bits for the depthbuffer
        8,                        //Number of bits for the stencilbuffer
        // 0,                        //Number of bits for the stencilbuffer
        0,                        //Number of Aux buffers in the framebuffer.
        PFD_MAIN_PLANE,
        0,
        0, 0, 0
    };    
    
    HDC deviceContext = GetDC(systemData->windowHandle);
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
    Rid[0].dwFlags = RIDEV_INPUTSINK;   
    Rid[0].hwndTarget = systemData->windowHandle;
    bool r = RegisterRawInputDevices(Rid, 1, sizeof(Rid[0]));

    printf("%Opengl Version: %s\n", (char*)glGetString(GL_VERSION));
}

void makeWindowVisible(SystemData* systemData) {
    ShowWindow(systemData->windowHandle, SW_SHOW);
}

void updateInput(Input* input, bool* isRunning, HWND windowHandle) {
    input->mouseWheel = 0;
    for(int i = 0; i < arrayCount(input->mouseButtonPressed); i++) input->mouseButtonPressed[i] = 0;
    for(int i = 0; i < arrayCount(input->keysPressed); i++) input->keysPressed[i] = 0;
    input->mShift = 0;
    input->mCtrl = 0;
    input->mAlt = 0;
    input->inputCharacterCount = 0;

    input->mouseDeltaX = 0;
    input->mouseDeltaY = 0;

    MSG message;
    while(PeekMessage(&message, 0, 0, 0, PM_REMOVE)) {
        switch(message.message) {
            case WM_MOUSEWHEEL: {
                short wheelDelta = HIWORD(message.wParam);
                if (wheelDelta > 0) input->mouseWheel = 1;
                if (wheelDelta < 0) input->mouseWheel = -1;
            } break;

            case WM_LBUTTONDOWN: { 
            	SetCapture(windowHandle);
            	input->mouseButtonPressed[0] = true; 
				input->mouseButtonDown[0] = true; 
			} break;
            case WM_RBUTTONDOWN: { 
            	SetCapture(windowHandle);
            	input->mouseButtonPressed[1] = true; 
				input->mouseButtonDown[1] = true; 
			} break;
            case WM_MBUTTONDOWN: { 
            	SetCapture(windowHandle);
            	input->mouseButtonPressed[2] = true; 
				input->mouseButtonDown[2] = true; 
			} break;

	        case WM_LBUTTONUP: { 
            	ReleaseCapture();
				input->mouseButtonDown[0] = false; 
			} break;
	        case WM_RBUTTONUP: { 
            	ReleaseCapture();
				input->mouseButtonDown[1] = false; 
			} break;
	        case WM_MBUTTONUP: { 
            	ReleaseCapture();
				input->mouseButtonDown[2] = false; 
			} break;

            case WM_KEYDOWN:
            case WM_KEYUP: {
                uint vk = uint(message.wParam);

                bool keyDown = (message.message == WM_KEYDOWN);
                int keycode = vkToKeycode(vk);
                input->keysDown[keycode] = keyDown;
                input->keysPressed[keycode] = keyDown;
                input->mShift = ((GetKeyState(VK_SHIFT) & 0x80) != 0);
                input->mCtrl = ((GetKeyState(VK_CONTROL) & 0x80) != 0);
                input->mAlt = ((GetKeyState(VK_MENU) & 0x80) != 0);

                if(keyDown) {
                    TranslateMessage(&message); 
                }
            } break;

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
            	
            	if (raw->header.dwType == RIM_TYPEMOUSE) {
            	    int xPosRelative = raw->data.mouse.lLastX;
            	    int yPosRelative = raw->data.mouse.lLastY;

            	    input->mouseDeltaX = -xPosRelative;
            	    input->mouseDeltaY = -yPosRelative;
            	} break;
            } break;

            // case WM_SIZE: {
            // } break;

            case WM_DESTROY: {
                *isRunning = false;
            } break;

            case WM_CLOSE: {
                *isRunning = false;
            } break;

            case WM_QUIT: {
                *isRunning = false;
            } break;

            case WM_ACTIVATEAPP: {

            } break;

            default: {
                TranslateMessage(&message); 
                DispatchMessage(&message); 
            } break;
        }
    }

    POINT point;    
    GetCursorPos(&point);
    ScreenToClient(windowHandle, &point);
    input->mousePos.x = point.x;
    input->mousePos.y = point.y;

    input->mousePosNegative = vec2(input->mousePos.x, -input->mousePos.y);

    input->firstFrame = false;
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

struct WindowSettings {
	Vec2i res;
	Vec2i fullRes;
	bool fullscreen;
	uint style;
	WINDOWPLACEMENT g_wpPrev;

	Vec2i currentRes;
	float aspectRatio;
};

void setWindowStyle(HWND hwnd, DWORD dwStyle) {
	SetWindowLong(hwnd, GWL_STYLE, dwStyle);
}

DWORD getWindowStyle(HWND hwnd) {
	return GetWindowLong(hwnd, GWL_STYLE);
}

void setWindowMode(HWND hwnd, WindowSettings* wSettings, int mode) {
	if(mode == WINDOW_MODE_FULLBORDERLESS && !wSettings->fullscreen) {
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
		SetWindowPos(hwnd, NULL, 0,0, wSettings->res.w, wSettings->res.h, SWP_NOZORDER | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		wSettings->fullscreen = false;
	}

	getWindowProperties(hwnd, &wSettings->currentRes.x, &wSettings->currentRes.y,0,0,0,0);
	wSettings->aspectRatio = wSettings->currentRes.x / (float)wSettings->currentRes.y;
}

void swapBuffers(SystemData* systemData) {
    SwapBuffers(systemData->deviceContext);
}

// MetaPlatformFunction();
uint getTicks() {
    uint result = GetTickCount();

    return result;
}

__int64 getTimestamp() {
	return __rdtsc();
}

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

// MetaPlatformFunction();
void sleep(int milliseconds) {
    Sleep(milliseconds);
}

#define REQUEST_DEBUG_OUTPUT 0

// MetaPlatformFunction();
void sendHttpRequest(void* data) {
    HttpRequest* request = (HttpRequest*)data;

    char tempLink[128];
    strCpy(tempLink, request->link);

    char host[128];
    char hostAndPort[128];
    char hirarchy[128];
    int port = -1;

    int findStart = strFind(tempLink, "://");
    if(findStart != -1) strCpy(tempLink, tempLink+findStart+3);

    int hasContent = strFind(tempLink, '/');
    if(hasContent != -1) {
        copySubstring(hostAndPort, tempLink, 0, strFind(tempLink, '/')-1);
        copySubstring(hirarchy, tempLink, strFind(tempLink, '/'), strLen(tempLink));
    } else {
        strCpy(hostAndPort, tempLink);  
        hirarchy[0] = '\\';
    }           

    int portSign = strFind(hostAndPort,':');
    if(portSign != -1) {
        copySubstring(host, hostAndPort, 0, portSign-1);
        port = strToInt(hostAndPort+portSign+1);
    } else {
        strCpy(host, hostAndPort);
    }

    char httpRequest[512]; httpRequest[0] = '\0';
    if(request->type == REQUEST_TYPE_GET) strAppend(httpRequest, "GET ");
    if(request->type == REQUEST_TYPE_POST) strAppend(httpRequest, "POST ");
    strAppend(httpRequest, hirarchy);
    strAppend(httpRequest, " HTTP/1.1\r\n");
    strAppend(httpRequest, "Host: ");
    strAppend(httpRequest, hostAndPort);
    strAppend(httpRequest, "\r\n");
    strAppend(httpRequest, "Accept: */*\r\n");
    strAppend(httpRequest, "Accept-Encoding: */*\r\n");
    strAppend(httpRequest, "Accept-Language: */*\r\n");
    strAppend(httpRequest, "Connection: keep-alive\r\n");
    strAppend(httpRequest, "Range: bytes 0-\r\n");
    if(request->type == REQUEST_TYPE_POST) {
        char b[16];
        int contentLength = strLen(request->additionalBodyContent);
        intToStr(b, contentLength);

        strAppend(httpRequest, "Content-Length: ");
        strAppend(httpRequest, b);
        strAppend(httpRequest, "\r\n");
    }
    strAppend(httpRequest, "\r\n");
    if(request->type == REQUEST_TYPE_POST) {
        strAppend(httpRequest, request->additionalBodyContent);
        // strAppend(httpRequest, "\r\n");
    }

    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
        printf("WSAStartup failed.\n");
    }

    struct addrinfo hints;
    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* targetAdressInfo = 0;
    DWORD getAddrRes = getaddrinfo(host, 0, &hints, &targetAdressInfo);
    if (getAddrRes != 0 || targetAdressInfo == 0) {
        printf("Could not resolve the Host Name");
        WSACleanup();
    }

    SOCKADDR_IN sockAddr;
    sockAddr.sin_addr = ((struct sockaddr_in*) targetAdressInfo->ai_addr)->sin_addr;
    sockAddr.sin_family = AF_INET;
    if(port == -1) sockAddr.sin_port = htons(80);
    else sockAddr.sin_port = htons(port);

    freeaddrinfo(targetAdressInfo);

    SOCKET webSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (webSocket == INVALID_SOCKET) {
        printf("Creation of the Socket Failed");
        WSACleanup();
    }

    timeval timeOut;
    // timeOut.tv_sec = 10;
    // timeOut.tv_usec = 0;
    // if(setsockopt(webSocket, SOL_SOCKET, SO_SNDTIMEO, (char*)(&timeOut), sizeof(timeOut))) {
    //  printf("Could not set socket settings.\n");
    // }
    timeOut.tv_sec = 500;
    timeOut.tv_usec = 0;
    if(setsockopt(webSocket, SOL_SOCKET, SO_RCVTIMEO, (char*)(&timeOut), sizeof(timeOut))) {
        printf("Could not set socket settings.\n");
    }

    if(REQUEST_DEBUG_OUTPUT) printf("Connecting...\n");
    if(connect(webSocket, (SOCKADDR*)&sockAddr, sizeof(sockAddr)) != 0) {
        printf("Could not connect\n");
        closesocket(webSocket);
        WSACleanup();
    }
    if(REQUEST_DEBUG_OUTPUT) printf("Connected\n\n");

    int sentBytes = send(webSocket, httpRequest, strlen(httpRequest),0);
    if (sentBytes < strlen(httpRequest) || sentBytes == SOCKET_ERROR) {
        printf("Could not send the request to the Server\n");
        closesocket(webSocket);
        WSACleanup();
    }
    if(REQUEST_DEBUG_OUTPUT) {
        printf("HTTP REQUEST: \n");
        printf("%s", httpRequest);
        if(request->type == REQUEST_TYPE_POST) printf("\n\n", httpRequest);
    }

    // char message[20000];
    char message[8000];
    int messageTotalSize = arrayCount(message);
    ZeroMemory(message, messageTotalSize);



    bool storeInFile = false;
    bool storeInBuffer = false;


    FILE *dataFile;
    if(request->contentFile) {
        storeInFile = true;

        dataFile = fopen(request->contentFile, "wb");
        fclose(dataFile);
        dataFile = fopen(request->contentFile, "a+b");
        if(!dataFile) printf("Cant open data file\n");
    }

    char* contentBuffer = request->contentBuffer;
    if(contentBuffer) {
        storeInBuffer = true;
    }

    int totalBytesReceived = 0;
    int totalContentBytesReceived = 0;
    bool parseHeader = true;

    char header[4048];
    int headerSize = 0;
    int headerContentLength;

    int chunkCurrentSize = 0;
    int chunkTotalSize = 0;

    int receivedCode;
    do {
        receivedCode = recv(webSocket, message, messageTotalSize, 0);
        
        if(receivedCode > 0) {
            int messageSize = receivedCode;
            totalBytesReceived += messageSize;

            char* buffer = message;
            int bufferSize = messageSize;

            if(parseHeader) {
                headerSize = strFindRight(message, "\r\n\r\n");
                strCpy(header, message, headerSize);

                char b[16];
                int infoPos = strFindRight(header, "Content-Length: ", headerSize);
                if(infoPos != -1) {
                    strCpy(b, header+infoPos, strFind(header+infoPos, "\r\n"));
                    headerContentLength = strToInt(b);
                    if(headerContentLength == 0) headerContentLength = -1;
                } else {
                    headerContentLength = -1;
                }

                parseHeader = false;
                
                totalBytesReceived -= headerSize;
                buffer += headerSize;
                bufferSize -= headerSize;

                if(REQUEST_DEBUG_OUTPUT) printf("HTTP RESPONSE: \n%s", header);
            }

            float progress = 0;
            if(headerContentLength != -1) {
                progress = ((float)totalBytesReceived/headerContentLength) * 100;
                if(REQUEST_DEBUG_OUTPUT) printf("Receiving... Size/Bytes: %i Progress: %.3f\n", receivedCode, progress);

                if(storeInFile) fwrite(buffer, 1, bufferSize, dataFile);
                if(storeInBuffer) memCpy(contentBuffer+totalContentBytesReceived, buffer, bufferSize);
                totalContentBytesReceived += bufferSize;

                if(totalBytesReceived == headerContentLength) {
                    if(REQUEST_DEBUG_OUTPUT) printf("File download successfull.\n");
                    break;
                }
            } else {
                bool streamEnd = false;

                while(bufferSize > 0) {
                    if(chunkCurrentSize < chunkTotalSize) {
                        int chunkSizeRemaining = chunkTotalSize - chunkCurrentSize;
                        if(chunkSizeRemaining <= bufferSize) {
                            // load in remaining and begin new chunk
                            if(storeInFile) fwrite(buffer, 1, chunkSizeRemaining, dataFile);
                            if(storeInBuffer) memCpy(contentBuffer+totalContentBytesReceived, buffer, chunkSizeRemaining);
                            totalContentBytesReceived += chunkSizeRemaining;

                            chunkCurrentSize = 0;
                            chunkTotalSize = 0;

                            buffer += chunkSizeRemaining + 2; //\r\n at end 
                            bufferSize -= chunkSizeRemaining + 2; //\r\n at end
                        } else {
                            // load in difference
                            int chunkSizeAvailable = bufferSize;
                            if(storeInFile) fwrite(buffer, 1, chunkSizeAvailable, dataFile);
                            if(storeInBuffer) memCpy(contentBuffer+totalContentBytesReceived, buffer, chunkSizeAvailable);
                            totalContentBytesReceived += chunkSizeAvailable;
                            
                            chunkCurrentSize += chunkSizeAvailable;
                            break;
                        }
                    } else {
                        // load next chunk
                        if(chunkCurrentSize == 0) {
                            char b[16];
                            int chunkInfoPos = strFind(buffer, "\r\n");
                            strCpy(b, buffer, chunkInfoPos);
                            int chunkSize = strHexToInt(b);
                            int chunkInfoSize = strLen(b) + 2;

                            // printf("aaa: %i %*.*s %i\n", chunkInfoPos, 4,4,b, chunkSize);

                            buffer += chunkInfoSize;
                            bufferSize -= chunkInfoSize;

                            chunkCurrentSize = 0;
                            chunkTotalSize = chunkSize;

                            // close stream if no more chunks available
                            if(chunkTotalSize == 0) {
                                streamEnd = true;
                                break;
                            }
                        }
                    }
                }

                if(REQUEST_DEBUG_OUTPUT) printf("Loading webpage: %i bytes\n", messageSize);

                if(streamEnd) {
                    if(REQUEST_DEBUG_OUTPUT) printf("Webpage successfully loaded.\n");
                    break;
                }
            }

            request->size = totalContentBytesReceived;
            request->progress = progress;
            if(request->stopProcess) {
                break;
            }

        } else if(receivedCode == 0) {
            if(REQUEST_DEBUG_OUTPUT) printf("Connection closed.\n");
        } else {
            int error = WSAGetLastError();
            if(error == 10060) printf("Timeout Receiving.\n");
            else printf("Connection error has ocurred: %i\n", error);
        }
    } while (receivedCode > 0);

    if(REQUEST_DEBUG_OUTPUT) {
        printf("Total Bytes: %i Content Bytes: %i\n", totalBytesReceived, totalContentBytesReceived);
        printf("\n");
    }

    if(storeInFile) fclose(dataFile);

    if(request->headerResponseFile) {
        FILE *file = fopen(request->headerResponseFile, "w");
        fwrite(header, 1, headerSize, file);
        fclose(file);
    }

    closesocket(webSocket);
    WSACleanup();

    request->size = totalContentBytesReceived;
    request->finished = true;
}



