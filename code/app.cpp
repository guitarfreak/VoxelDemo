/*
//-----------------------------------------
//				WHAT TO DO
//-----------------------------------------
- Joysticks, Keyboard, Mouse, Xinput-DirectInput
- Sound
- Data Package - Streamingw
- Expand Font functionality
- Gui
- create simpler windows.h
- remove c runtime library, implement sin, cos...
- savestates, hotrealoading
- pre and post functions to Main
- memory dynamic expansion
- 32x32 voxel chunks
- ballistic motion on jumping
- round collision
- screen space reflections

- advance timestep when window not in focus (stop game when not in focus)
- reduce number of thread queue pushes
- insert and init mesh not thread proof, make sure every mesh is initialized before generating
- implement sun and clouds that block beams of light
- glowstone emmiting light
- make voxel drop in tutorial code for stb_voxel
- stb_voxel push alpha test in voxel vertex shader
- rocket launcher
- antialiased pixel graphics with neighbour sampling 
- macros for array stuff so i dont have to inline updateMeshList[updateMeshListSize++] every time 
- level of detail for world gen back row								
- 32x32 gen chunks
  
- put in spaces for fillString
- save mouse position at startup and place the mouse there when the app closes
- simplex noise instead of perlin noise
- make thread push copy the thead specific data on push in a seperate buffer for all possible threadjobs
- frametime timer and fps counter

- simd voxel generation
- simd on vectors as a test first?

- experiment with directx 11
- font drawing bold

gui stuff: 
additionally:
- 2d drawing
- 3d drawing
- vector representation
- sound 
- entities
- make columns stack
- drag element to seperate column elements
- struct as defer for section begin and such
- make settings pushable as a whole
  -> doesn't that remove the need for explicit stacking in scrollbars and such?
- top menu bar
- Clean up the gui to make it more usable.
- Make entities watchable and changeable in Gui.

Changing course for now:
 - In executeCommandList: Remember state, and only switch if a change occured. (Like shaders, colors, linewidth, etc.)

 - 3d animation system. (Search Opengl vertex skinning.)
 - Sound perturbation. (Whatever that is.) 
 - Clean up gui.
 - Using makros and defines to make templated vectors and hashtables and such.
 - When switching between text editor and debugger, synchronize open files.
 - Entity introspection in gui.
 - Open devenv from within sublime.
 - Shadow mapping, start with cloud texture projection.

- Main Menu.
- New game, load game, save game.
- Settings.

//-------------------------------------
//               BUGS
//-------------------------------------
- look has to be negative to work in view projection matrix
- distance jumping collision bug, possibly precision loss in distances
- gpu fucks up at some point making swapBuffers alternates between time steps 
  which makes the game stutter, restart is required
- game input gets stuck when buttons are pressed right at the start
- sort key assert firing randomly
- draw line crashes
- release build takes forever
- Crashing once in a while at startup. (NOTE: Has not happened in quite a while.)

*/

/*
	switching shader -> 550 ticks
	using namedBufferSubData vs uniforms for vertices -> 2400 ticks vs 400 ticks
*/



// Intrinsics.

// #include <iacaMarks.h>
#include <xmmintrin.h>
#include <emmintrin.h>

// External.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl\gl.h>
// #include "external\glext.h"

#include <Mmdeviceapi.h>
#include <Audioclient.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG
#include "external\stb_image.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_PARAMETER_TAGS_H
#include FT_MODULE_H


#define STB_VOXEL_RENDER_IMPLEMENTATION
// #define STBVOX_CONFIG_LIGHTING_SIMPLE
#define STBVOX_CONFIG_FOG_SMOOTHSTEP
// #define STBVOX_CONFIG_MODE 0
#define STBVOX_CONFIG_MODE 1
#include "external\stb_voxel_render.h"



//

struct ThreadQueue;
struct GraphicsState;
struct AudioState;
struct DrawCommandList;
struct MemoryBlock;
struct DebugState;
struct Timer;
ThreadQueue* globalThreadQueue;
GraphicsState* globalGraphicsState;
AudioState* theAudioState;
DrawCommandList* globalCommandList;
MemoryBlock* globalMemory;
DebugState* globalDebugState;
Timer* globalTimer;

// Internal.

#include "rt_types.cpp"
#include "rt_timer.cpp"
#include "rt_misc.cpp"
#include "rt_math.cpp"
#include "rt_hotload.cpp"
#include "rt_misc_win32.cpp"
#include "rt_platformWin32.cpp"

#include "memory.cpp"
#include "openglDefines.cpp"
#include "userSettings.cpp"

#include "audio.cpp"

#include "rendering.cpp"
#include "gui.cpp"

#include "entity.cpp"
#include "voxel.cpp"

#include "debug.cpp"




struct GameSettings {
	float mouseSensitivity;
};

void defaultGameSettings(GameSettings* settings) {
	settings->mouseSensitivity = 0.1f;
}

enum Game_Mode {
	GAME_MODE_MENU,
	GAME_MODE_LOAD,
	GAME_MODE_MAIN,
};

enum Menu_Screen {
	MENU_SCREEN_MAIN,
	MENU_SCREEN_SETTINGS,
};

struct MainMenu {
	bool gameRunning;

	int screen;
	int activeId;
	int currentId;

	float buttonAnimState;

	Font* font;
	Vec4 cOption;
	Vec4 cOptionActive;
	Vec4 cOptionShadow;
	Vec4 cOptionShadowActive;

	float optionShadowSize;

	bool pressedEnter;
};

bool menuOption(MainMenu* menu, char* text, Vec2 pos, Vec2i alignment) {

	bool isActive = menu->activeId == menu->currentId;
	Vec4 textColor = isActive ? menu->cOptionActive : menu->cOption;
	Vec4 shadowColor = isActive ? menu->cOptionShadowActive : menu->cOptionShadow;

	dcText(text, menu->font, pos, textColor, alignment, 0, menu->optionShadowSize, shadowColor);

	bool result = menu->pressedEnter && menu->activeId == menu->currentId;

	menu->currentId++;

	return result;
}


struct AppData {
	
	// General.

	SystemData systemData;
	Input input;
	WindowSettings wSettings;
	GraphicsState graphicsState;

	AudioState audioState;

	f64 dt;
	f64 time;
	int frameCount;

	DrawCommandList commandList2d;
	DrawCommandList commandList3d;

	bool updateFrameBuffers;

	// 

	bool captureMouseKeepCenter;
	bool captureMouse;
	bool fpsMode;
	bool fpsModeFixed;
	bool lostFocusWhileCaptured;

	Camera activeCam;

	Vec2i cur3dBufferRes;
	int msaaSamples;
	Vec2i fboRes;
	bool useNativeRes;

	float aspectRatio;
	float fieldOfView;
	float nearPlane;
	float farPlane;

	// Game.

	int gameMode;
	MainMenu menu;

	bool loading;
	bool newGame;
	float startFade;

	GameSettings settings;

	//

	EntityList entityList;
	Entity* player;
	Entity* cameraEntity;

	bool* treeNoise;

	bool playerMode;
	bool pickMode;

	bool playerOnGround;
	int blockMenu[10];
	int blockMenuSelected;

	float bombFireInterval;
	bool bombButtonDown;
	float bombSpawnTimer;

	bool blockSelected;
	Vec3 selectedBlock;
	Vec3 selectedBlockFaceDir;

	VoxelData voxelData;

	uchar* voxelCache[8];
	uchar* voxelLightingCache[8];

	MakeMeshThreadedData threadData[256];

	bool reloadWorld;

	int voxelDrawCount;
	int voxelTriangleCount;

	int skyBoxId;

	// Particles.

	GLuint testBufferId;
	char* testBuffer;
	int testBufferSize;
};





// void debugMain(DebugState* ds, AppMemory* appMemory, AppData* ad, bool reload, bool* isRunning, bool init);
void debugMain(DebugState* ds, AppMemory* appMemory, AppData* ad, bool reload, bool* isRunning, bool init, ThreadQueue* threadQueue);
// void debugUpdatePlayback(DebugState* ds, AppMemory* appMemory);

#ifdef FULL_OPTIMIZE
#pragma optimize( "", on )
#else 
#pragma optimize( "", off )
#endif

extern "C" APPMAINFUNCTION(appMain) {

	if(init) {

		// Init memory.

		SYSTEM_INFO info;
		GetSystemInfo(&info);

		char* baseAddress = (char*)gigaBytes(8);
	    VirtualAlloc(baseAddress, gigaBytes(40), MEM_RESERVE, PAGE_READWRITE);

		ExtendibleMemoryArray* pMemory = &appMemory->extendibleMemoryArrays[appMemory->extendibleMemoryArrayCount++];
		initExtendibleMemoryArray(pMemory, megaBytes(512), info.dwAllocationGranularity, baseAddress);

		ExtendibleBucketMemory* dMemory = &appMemory->extendibleBucketMemories[appMemory->extendibleBucketMemoryCount++];
		initExtendibleBucketMemory(dMemory, megaBytes(1), megaBytes(512), info.dwAllocationGranularity, baseAddress + gigaBytes(16));

		MemoryArray* tMemoryDebug = &appMemory->memoryArrays[appMemory->memoryArrayCount++];
		initMemoryArray(tMemoryDebug, megaBytes(30), baseAddress + gigaBytes(33));



		MemoryArray* pDebugMemory = &appMemory->memoryArrays[appMemory->memoryArrayCount++];
		initMemoryArray(pDebugMemory, megaBytes(50), 0);

		MemoryArray* tMemory = &appMemory->memoryArrays[appMemory->memoryArrayCount++];
		initMemoryArray(tMemory, megaBytes(30), 0);

		ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[appMemory->extendibleMemoryArrayCount++];
		initExtendibleMemoryArray(debugMemory, megaBytes(512), info.dwAllocationGranularity, baseAddress + gigaBytes(34));
	}

	// Setup memory and globals.

	MemoryBlock gMemory = {};
	gMemory.pMemory = &appMemory->extendibleMemoryArrays[0];
	gMemory.tMemory = &appMemory->memoryArrays[0];
	gMemory.dMemory = &appMemory->extendibleBucketMemories[0];
	gMemory.pDebugMemory = &appMemory->memoryArrays[1];
	gMemory.tMemoryDebug = &appMemory->memoryArrays[2];
	gMemory.debugMemory = &appMemory->extendibleMemoryArrays[1];
	globalMemory = &gMemory;

	DebugState* ds = (DebugState*)getBaseMemoryArray(gMemory.pDebugMemory);
	AppData* ad = (AppData*)getBaseExtendibleMemoryArray(gMemory.pMemory);
	GraphicsState* gs = &ad->graphicsState;

	Input* input = &ad->input;
	SystemData* sd = &ad->systemData;
	HWND windowHandle = sd->windowHandle;
	WindowSettings* ws = &ad->wSettings;

	globalThreadQueue = threadQueue;
	globalGraphicsState = &ad->graphicsState;
	theAudioState = &ad->audioState;
	globalDebugState = ds;
	globalTimer = ds->timer;

	// AppGlobals.

	voxelThreadData = ad->threadData;
	treeNoise = ad->treeNoise;

	for(int i = 0; i < 8; i++) {
		voxelCache[i] = ad->voxelCache[i];
		voxelLightingCache[i] = ad->voxelLightingCache[i];
	}

	// Init.

	if(init) {

		//
		// DebugState.
		//

		getPMemoryDebug(sizeof(DebugState));
		*ds = {};
		ds->assets = getPArrayDebug(Asset, 100);

		ds->inputCapacity = 600;
		ds->recordedInput = (Input*)getPMemoryDebug(sizeof(Input) * ds->inputCapacity);

		ds->timer = getPStructDebug(Timer);
		globalTimer = ds->timer;
		int timerSlots = 50000;
		ds->timer->bufferSize = timerSlots;
		ds->timer->timerBuffer = (TimerSlot*)getPMemoryDebug(sizeof(TimerSlot) * timerSlots);

		ds->savedBufferMax = 20000;
		ds->savedBufferIndex = 0;
		ds->savedBufferCount = 0;
		ds->savedBuffer = (GraphSlot*)getPMemoryDebug(sizeof(GraphSlot) * ds->savedBufferMax);


		ds->gui = getPStructDebug(Gui);
		// gui->init(rectCenDim(vec2(0,1), vec2(300,800)));
		// gui->init(rectCenDim(vec2(1300,1), vec2(300,500)));
		ds->gui->init(rectCenDim(vec2(1300,1), vec2(300, ws->currentRes.h)), 0);

		// ds->gui->cornerPos = 

		ds->gui2 = getPStructDebug(Gui);
		// ds->gui->init(rectCenDim(vec2(1300,1), vec2(400, ws->currentRes.h)), -1);
		ds->gui2->init(rectCenDim(vec2(1300,1), vec2(300, ws->currentRes.h)), 3);

		ds->input = getPStructDebug(Input);
		ds->showMenu = false;
		ds->showStats = false;
		ds->showConsole = false;
		ds->showHud = false;
		// ds->guiAlpha = 0.95f;
		ds->guiAlpha = 1;

		for(int i = 0; i < arrayCount(ds->notificationStack); i++) {
			ds->notificationStack[i] = getPStringDebug(DEBUG_NOTE_LENGTH+1);
		}

		ds->fontScale = 1.0f;

		TIMER_BLOCK_NAMED("Init");

		//
		// AppData.
		//

		getPMemory(sizeof(AppData));
		*ad = {};
		
		// int windowStyle = WS_OVERLAPPEDWINDOW & ~WS_SYSMENU;
		int windowStyle = WS_OVERLAPPEDWINDOW;
		initSystem(sd, ws, windowsData, vec2i(1920*0.85f, 1080*0.85f), windowStyle, 1);

		windowHandle = sd->windowHandle;

		SetWindowText(windowHandle, APP_NAME);

		loadFunctions();

		if(true) {
			wglSwapIntervalEXT(1);
			ws->vsync = true;
			ws->frameRate = ws->refreshRate;
		} else {
			wglSwapIntervalEXT(0);
			ws->vsync = false;
			ws->frameRate = 200;
		}

		initInput(&ad->input);
		sd->input = &ad->input;

		#ifndef SHIPPING_MODE
		if(!IsDebuggerPresent()) {
			makeWindowTopmost(sd);
		}
		#endif

		//
		// Init Folder Handles.
		//

		initWatchFolders(sd->folderHandles, ds->assets, &ds->assetCount);

		//
		// Setup Textures.
		//

		for(int i = 0; i < TEXTURE_SIZE; i++) {
			Texture tex;
			loadTextureFromFile(&tex, texturePaths[i], -1, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE);
			addTexture(tex);
		}

		for(int i = 0; i < CUBEMAP_SIZE; i++) {
			loadCubeMapFromFile(gs->cubeMaps + i, (char*)cubeMapPaths[i], 5, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE);
		}

		loadVoxelTextures((char*)minecraftTextureFolderPath, INTERNAL_TEXTURE_FORMAT);

		//
		// Setup shaders and uniforms.
		//

		loadShaders();

		//
		// Setup Meshs.
		//

		uint vao = 0;
		glCreateVertexArrays(1, &vao);
		glBindVertexArray(vao);

		for(int i = 0; i < MESH_SIZE; i++) {
			Mesh* mesh = getMesh(i);

			MeshMap* meshMap = meshArrays +i;

			glCreateBuffers(1, &mesh->bufferId);
			glNamedBufferData(mesh->bufferId, meshMap->size, meshMap->vertexArray, GL_STATIC_DRAW);
			mesh->vertCount = meshMap->size / sizeof(Vertex);
		}

		ad->testBufferSize = megaBytes(10);
		ad->testBuffer = getPArray(char, ad->testBufferSize);
		glCreateBuffers(1, &ad->testBufferId);
		glNamedBufferData(ad->testBufferId, ad->testBufferSize, ad->testBuffer, GL_STREAM_DRAW);

		// 
		// Samplers.
		//

		gs->samplers[SAMPLER_NORMAL] = createSampler(16.0f, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
		// gs->samplers[SAMPLER_NORMAL] = createSampler(16.0f, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST);

		gs->samplers[SAMPLER_VOXEL_1] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR);
		gs->samplers[SAMPLER_VOXEL_2] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR);
		gs->samplers[SAMPLER_VOXEL_3] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);

		//
		//
		//

		ad->fieldOfView = 60;
		ad->msaaSamples = 4;
		ad->fboRes = vec2i(0, 120);
		ad->useNativeRes = true;
		ad->nearPlane = 0.1f;
		ad->farPlane = 3000;
		ad->dt = 1/(float)60;

		//
		// FrameBuffers.
		//

		{
			for(int i = 0; i < FRAMEBUFFER_SIZE; i++) {
				FrameBuffer* fb = getFrameBuffer(i);
				initFrameBuffer(fb);
			}

			attachToFrameBuffer(FRAMEBUFFER_3dMsaa, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0, ad->msaaSamples);
			attachToFrameBuffer(FRAMEBUFFER_3dMsaa, FRAMEBUFFER_SLOT_DEPTH_STENCIL, GL_DEPTH_STENCIL, 0, 0, ad->msaaSamples);

			attachToFrameBuffer(FRAMEBUFFER_3dNoMsaa, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);
			attachToFrameBuffer(FRAMEBUFFER_3dNoMsaa, FRAMEBUFFER_SLOT_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, 0, 0);

			attachToFrameBuffer(FRAMEBUFFER_Reflection, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);
			attachToFrameBuffer(FRAMEBUFFER_Reflection, FRAMEBUFFER_SLOT_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, 0, 0);

			attachToFrameBuffer(FRAMEBUFFER_2d, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);

			attachToFrameBuffer(FRAMEBUFFER_DebugMsaa, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0, ad->msaaSamples);
			attachToFrameBuffer(FRAMEBUFFER_DebugNoMsaa, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);

			ad->updateFrameBuffers = true;
		}

		//
		// @AudioInit.
		//

		AudioState* as = &ad->audioState;
		(*as) = {};

		{
			int hr;

			const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
			const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
			hr = CoCreateInstance(
			       CLSID_MMDeviceEnumerator, NULL,
			       CLSCTX_ALL, IID_IMMDeviceEnumerator,
			       (void**)&as->deviceEnumerator);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			hr = as->deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &as->immDevice);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			hr = as->immDevice->Activate(__uuidof(IAudioClient),CLSCTX_ALL,NULL,(void**)&as->audioClient);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			int referenceTimeToSeconds = 10 * 1000 * 1000;
			REFERENCE_TIME referenceTime = referenceTimeToSeconds; // 100 nano-seconds -> 1 second.
			hr = as->audioClient->GetMixFormat(&as->waveFormat);

			{
				WAVEFORMATEX* format = as->waveFormat;
				format->wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
				format->nChannels = 2;
				format->wBitsPerSample = 32;
				format->cbSize = 0;

				WAVEFORMATEX what = {};
				WAVEFORMATEX* formatClosest = &what;
				hr = as->audioClient->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, format, &formatClosest);
				if(hr) { printf("Failed to initialise sound."); assert(!hr); };
			}

			as->audioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, referenceTime, 0, as->waveFormat, 0);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			REFERENCE_TIME latency;
			as->audioClient->GetStreamLatency(&latency);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			as->latencyInSeconds = (float)latency / referenceTimeToSeconds;

			hr = as->audioClient->GetBufferSize(&as->bufferFrameCount);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			hr = as->audioClient->GetService(__uuidof(IAudioRenderClient), (void**)&as->renderClient);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			// Fill with silence before starting.

			float* buffer;
			hr = as->renderClient->GetBuffer(as->bufferFrameCount, (BYTE**)&buffer);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			// hr = as->renderClient->ReleaseBuffer(as->bufferFrameCount, AUDCLNT_BUFFERFLAGS_SILENT);
			hr = as->renderClient->ReleaseBuffer(0, AUDCLNT_BUFFERFLAGS_SILENT);
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };

			// // Calculate the actual duration of the allocated buffer.
			// hnsActualDuration = (double)REFTIMES_PER_SEC *
			//                     as->bufferFrameCount / pwfx->nSamplesPerSec;

			hr = as->audioClient->Start();  // Start playing.
			if(hr) { printf("Failed to initialise sound."); assert(!hr); };
		}

		as->masterVolume = 1.0f;

		as->fileCountMax = 100;
		as->files = getPArray(Audio, as->fileCountMax);

		char* audioFolderPath = fillString("%s*", App_Audio_Folder);

		FolderSearchData fd;
		folderSearchStart(&fd, audioFolderPath);

		while(folderSearchNextFile(&fd)) {

			if(fd.type == FILE_TYPE_FOLDER) {
				char* subFolder = fd.fileName;
				char* subFolderPath = fillString("%s%s\\*", App_Audio_Folder, subFolder);

				FolderSearchData fdSub;
				folderSearchStart(&fdSub, subFolderPath);

				while(folderSearchNextFile(&fdSub)) {

					char* filePath = fillString("%s%s\\%s", App_Audio_Folder, subFolder, fdSub.fileName);
					char* name = fillString("%s\\%s", subFolder, fdSub.fileName);
					addAudio(as, filePath, name);
				}

			} else {
				char* folderPath = fillString("%s%s", App_Audio_Folder, fd.fileName);
				addAudio(as, folderPath, fd.fileName);
			}
		}

		//
		//
		//

		globalGraphicsState->fontFolders[globalGraphicsState->fontFolderCount++] = getPStringCpy(App_Font_Folder);
		char* windowsFontFolder = fillString("%s%s", getenv(Windows_Font_Path_Variable), Windows_Font_Folder);
		globalGraphicsState->fontFolders[globalGraphicsState->fontFolderCount++] = getPStringCpy(windowsFontFolder);

		// Setup app temp settings.
		AppSessionSettings appSessionSettings = {};
		{
			// @AppSessionDefaults
			if(!fileExists(App_Session_File)) {
				AppSessionSettings at = {};

				Rect r = ws->monitors[0].workRect;
				Vec2 center = vec2(rectCenX(r), (r.top - r.bottom)/2);
				Vec2 dim = vec2(rectW(r), -rectH(r));
				at.windowRect = rectCenDim(center, dim*0.85f);

				appWriteSessionSettings(App_Session_File, &at);
			}

			// @AppSessionLoad
			{
				AppSessionSettings at = {};
				appReadSessionSettings(App_Session_File, &at);

				Recti r = rectiRound(at.windowRect);
				MoveWindow(windowHandle, r.left, r.top, r.right-r.left, r.bottom-r.top, true);

				appSessionSettings = at;
			}
		}

		pcg32_srandom(0, __rdtsc());

		//
		// @AppInit.
		//

		timerInit(&ds->swapTimer);
		timerInit(&ds->frameTimer);
		timerInit(&ds->tempTimer);

		ad->gameMode = GAME_MODE_LOAD;
		ad->newGame = false;
		// ad->menu.activeId = 0;

		ad->captureMouse = false;
		showCursor(false);

		folderExistsCreate(SAVES_FOLDER);	

		//

		defaultGameSettings(&ad->settings);

		// Entity.

		ad->playerMode = true;
		ad->pickMode = true;

		ad->entityList.size = 1000;
		ad->entityList.e = (Entity*)getPMemory(sizeof(Entity)*ad->entityList.size);
		for(int i = 0; i < ad->entityList.size; i++) ad->entityList.e[i].init = false;

		// Voxel.

		ad->skyBoxId = CUBEMAP_5;
		ad->bombFireInterval = 0.1f;
		ad->bombButtonDown = false;

		// Hashes and thread data.
		{
			VoxelData* vd = &ad->voxelData;
			vd->voxelHashSize = 1024;
			vd->voxelHash = getPArray(DArray<int>, vd->voxelHashSize);

			for(int i = 0; i < arrayCount(ad->threadData); i++) {
				ad->threadData[i] = {};
			} 

			for(int i = 0; i < arrayCount(ad->voxelCache); i++) {
				ad->voxelCache[i] = (uchar*)getPMemory(sizeof(uchar)*VOXEL_CACHE_SIZE);
				ad->voxelLightingCache[i] = (uchar*)getPMemory(sizeof(uchar)*VOXEL_CACHE_SIZE);
			}

			for(int i = 0; i < 8; i++) {
				voxelCache[i] = ad->voxelCache[i];
				voxelLightingCache[i] = ad->voxelLightingCache[i];
			}

			ad->voxelData.voxels.reserve(10000);
		}

		// Trees.
		{
			int treeRadius = 4;
			ad->treeNoise = (bool*)getPMemory(VOXEL_X*VOXEL_Y);
			zeroMemory(ad->treeNoise, VOXEL_X*VOXEL_Y);

			Rect bounds = rect(0, 0, 64, 64);
			Vec2* noiseSamples;
			// int noiseSamplesSize = blueNoise(bounds, 5, &noiseSamples);
			int noiseSamplesSize = blueNoise(bounds, treeRadius, &noiseSamples);
			// int noiseSamplesSize = 10;
			for(int i = 0; i < noiseSamplesSize; i++) {
				Vec2 s = noiseSamples[i];
				// Vec2i p = vec2i((int)(s.x/gridCell) * gridCell, (int)(s.y/gridCell) * gridCell);
				// drawRect(rectCenDim(vec2(p), vec2(5,5)), rect(0,0,1,1), vec4(1,0,1,1), ad->textures[0]);
				Vec2i index = vec2i(s);
				ad->treeNoise[index.y*VOXEL_X + index.x] = 1;
			}
			free(noiseSamples);
			treeNoise = ad->treeNoise;
		}
	}

	// @AppStart.

	TIMER_BLOCK_BEGIN_NAMED(reload, "Reload");

	if(reload) {
		loadFunctions();
		SetWindowLongPtr(sd->windowHandle, GWLP_WNDPROC, (LONG_PTR)mainWindowCallBack);
	    SetWindowLongPtr(sd->windowHandle, GWLP_USERDATA, (LONG_PTR)sd);

	    DeleteFiber(sd->messageFiber);
	    sd->messageFiber = CreateFiber(0, (PFIBER_START_ROUTINE)updateInput, sd);

		gs->screenRes = ws->currentRes;

		if(HOTRELOAD_SHADERS) {
			loadShaders();
		}

		// Bad news.
		for(int i = 0; i < arrayCount(globalGraphicsState->fonts); i++) {
			for(int j = 0; j < arrayCount(globalGraphicsState->fonts[0]); j++) {
				Font* font = &globalGraphicsState->fonts[i][j];
				if(font->heightIndex != 0) {
					freeFont(font);
				} else break;
			}
		}
	}

	TIMER_BLOCK_END(reload);

	// Update timer.
	{
		if(init) {
			timerStart(&ds->frameTimer);
			ds->dt = 1/(float)60;
		} else {
			ds->dt = timerUpdate(&ds->frameTimer);
			ds->time += ds->dt;

			ds->fpsTime += ds->dt;
			ds->fpsCounter++;
			if(ds->fpsTime >= 1) {
				ds->avgFps = 1 / (ds->fpsTime / (f64)ds->fpsCounter);
				ds->fpsTime = 0;
				ds->fpsCounter = 0;
			}

			// timerStart(&ad->frameTimer);
			// printf("%f\n", ad->dt);
		}
	}

	clearTMemory();

	// Allocate drawCommandlist.

	int clSize = kiloBytes(1000);
	drawCommandListInit(&ad->commandList3d, (char*)getTMemory(clSize), clSize);
	drawCommandListInit(&ad->commandList2d, (char*)getTMemory(clSize), clSize);
	globalCommandList = &ad->commandList3d;

	// Hotload changed files.

	reloadChangedFiles(sd->folderHandles, ds->assets, ds->assetCount);

	// Update input.
	{
		TIMER_BLOCK_NAMED("Input");

		inputPrepare(input);
		SwitchToFiber(sd->messageFiber);

		if(ad->input.closeWindow) *isRunning = false;

		// ad->input = *ds->input;
		*ds->input = ad->input;

		if(ds->console.isActive) {
			memSet(ad->input.keysPressed, 0, sizeof(ad->input.keysPressed));
			memSet(ad->input.keysDown, 0, sizeof(ad->input.keysDown));
		}

		if(mouseInClientArea(windowHandle)) {
			updateCursor(ws);
			showCursor(false);
		} else {
			showCursor(true);
		}

		ad->dt = ds->dt;
		ad->time = ds->time;

		ad->frameCount++;

		sd->fontHeight = getSystemFontHeight(sd->windowHandle);
		ds->fontHeight = roundInt(ds->fontScale*sd->fontHeight);
	}

	// Handle recording.
	{
		if(ds->recordingInput) {
			memCpy(ds->recordedInput + ds->inputIndex, &ad->input, sizeof(Input));
			ds->inputIndex++;
			if(ds->inputIndex >= ds->inputCapacity) {
				ds->recordingInput = false;
			}
		}

		if(ds->playbackInput && !ds->playbackPause) {
			ad->input = ds->recordedInput[ds->playbackIndex];
			ds->playbackIndex = (ds->playbackIndex+1)%ds->inputIndex;
			if(ds->playbackIndex == 0) ds->playbackSwapMemory = true;
		}
	} 

	if(ds->playbackPause) goto endOfMainLabel;

	if(ds->playbackBreak) {
		if(ds->playbackBreakIndex == ds->playbackIndex) {
			ds->playbackPause = true;
		}
	}

    if((input->keysPressed[KEYCODE_F11] || input->altEnter) && !sd->maximized) {
    	if(ws->fullscreen) setWindowMode(windowHandle, ws, WINDOW_MODE_WINDOWED);
    	else setWindowMode(windowHandle, ws, WINDOW_MODE_FULLBORDERLESS);
    }

	// if(input->keysPressed[KEYCODE_F2]) {
	// 	static bool switchMonitor = false;

	// 	setWindowMode(windowHandle, ws, WINDOW_MODE_WINDOWED);

	// 	if(!switchMonitor) setWindowProperties(windowHandle, 1, 1, 1920, 0);
	// 	else setWindowProperties(windowHandle, 1920, 1080, -1920, 0);
	// 	switchMonitor = !switchMonitor;

	// 	setWindowMode(windowHandle, ws, WINDOW_MODE_FULLBORDERLESS);
	// }

	if(input->resize || init) {
		if(!windowIsMinimized(windowHandle)) {
			updateResolution(windowHandle, ws);
			ad->updateFrameBuffers = true;
		}
		input->resize = false;
	}

	if(ad->updateFrameBuffers) {
		TIMER_BLOCK_NAMED("Upd FBOs");

		ad->updateFrameBuffers = false;
		ad->aspectRatio = ws->aspectRatio;
		
		ad->fboRes.x = ad->fboRes.y*ad->aspectRatio;

		if(ad->useNativeRes) ad->cur3dBufferRes = ws->currentRes;
		else ad->cur3dBufferRes = ad->fboRes;

		Vec2i s = ad->cur3dBufferRes;
		Vec2 reflectionRes = vec2(s);

		setDimForFrameBufferAttachmentsAndUpdate(FRAMEBUFFER_3dMsaa, s.w, s.h);
		setDimForFrameBufferAttachmentsAndUpdate(FRAMEBUFFER_3dNoMsaa, s.w, s.h);
		setDimForFrameBufferAttachmentsAndUpdate(FRAMEBUFFER_Reflection, reflectionRes.w, reflectionRes.h);
		setDimForFrameBufferAttachmentsAndUpdate(FRAMEBUFFER_2d, ws->currentRes.w, ws->currentRes.h);

		setDimForFrameBufferAttachmentsAndUpdate(FRAMEBUFFER_DebugMsaa, ws->currentRes.w, ws->currentRes.h);
		setDimForFrameBufferAttachmentsAndUpdate(FRAMEBUFFER_DebugNoMsaa, ws->currentRes.w, ws->currentRes.h);
	}
	


	TIMER_BLOCK_BEGIN_NAMED(openglInit, "Opengl Init");

	// Opengl Debug settings.
	{
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

		const int count = 1;
		GLenum sources;
		GLenum types;
		GLuint ids;
		GLenum severities;
		GLsizei lengths;

		int bufSize = 1000;
		char* messageLog = getTString(bufSize);

		memSet(messageLog, 0, bufSize);

		uint fetchedLogs = 1;
		while(fetchedLogs = glGetDebugMessageLog(count, bufSize, &sources, &types, &ids, &severities, &lengths, messageLog)) {
			if(severities == GL_DEBUG_SEVERITY_NOTIFICATION) continue;

			if(severities == GL_DEBUG_SEVERITY_HIGH) printf("HIGH: \n");
			else if(severities == GL_DEBUG_SEVERITY_MEDIUM) printf("MEDIUM: \n");
			else if(severities == GL_DEBUG_SEVERITY_LOW) printf("LOW: \n");
			else if(severities == GL_DEBUG_SEVERITY_NOTIFICATION) printf("NOTE: \n");

			printf("\t%s \n", messageLog);
		}
	}

	// Clear all the framebuffers and window backbuffer.
	{

		// for(int i = 0; i < arrayCount(gs->frameBuffers); i++) {
		// 	FrameBuffer* fb = getFrameBuffer(i);
		// 	bindFrameBuffer(i);

		// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		// }

		glClearColor(0,0,0,1);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		bindFrameBuffer(FRAMEBUFFER_3dMsaa);
		glClearColor(1,1,1,1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		bindFrameBuffer(FRAMEBUFFER_3dNoMsaa);
		glClearColor(1,1,1,1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		bindFrameBuffer(FRAMEBUFFER_Reflection);
		glClearColor(1,1,1,1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		bindFrameBuffer(FRAMEBUFFER_2d);
		glClearColor(0,0,0,0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		bindFrameBuffer(FRAMEBUFFER_DebugMsaa);
		glClearColor(0,0,0,0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		bindFrameBuffer(FRAMEBUFFER_DebugNoMsaa);
		glClearColor(0,0,0,0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	}

	// Setup opengl.
	{
		// glDepthRange(-1.0,1.0);
		glFrontFace(GL_CW);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glDisable(GL_SCISSOR_TEST);
		// glEnable(GL_LINE_SMOOTH);
		// glEnable(GL_POLYGON_SMOOTH);
		// glDisable(GL_POLYGON_SMOOTH);
		// glDisable(GL_SMOOTH);
		
		glEnable(GL_TEXTURE_2D);
		// glEnable(GL_ALPHA_TEST);
		// glAlphaFunc(GL_GREATER, 0.9);
		// glDisable(GL_LIGHTING);
		// glDepthFunc(GL_LESS);
		// glClearDepth(1);
		// glDepthMask(GL_TRUE);
		glEnable(GL_MULTISAMPLE);
		// glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		glViewport(0,0, ad->cur3dBufferRes.x, ad->cur3dBufferRes.y);
	}

	TIMER_BLOCK_END(openglInit);

	// Mouse capture.
	if(!init)
	{
		if(sd->killedFocus && ad->captureMouse) {
			ad->captureMouse = false;
			ad->lostFocusWhileCaptured = true;
		}

		if(ad->lostFocusWhileCaptured && windowHasFocus(windowHandle)) {
			if(input->mouseButtonPressed[0] && mouseInClientArea(windowHandle)) {
				ad->captureMouse = true;
				input->mouseButtonPressed[0] = false;

				ad->lostFocusWhileCaptured = false;
			}
		}

		if(!ad->captureMouse) {
			// if(input->keysPressed[KEYCODE_F3] || 
			//    (input->mouseButtonPressed[1]) && !ad->fpsModeFixed) {
				// if(input->keysPressed[KEYCODE_F3]) ad->fpsModeFixed = true;

			if(input->keysPressed[KEYCODE_F3]) {
				input->mouseButtonPressed[1] = false;
				ad->captureMouse = true;

				GetCursorPos(&ws->lastMousePosition);
			}
		} else {
			// if(input->keysPressed[KEYCODE_F3] || 
			   // (input->mouseButtonReleased[1] && !ad->fpsModeFixed)) {
				// if(input->keysPressed[KEYCODE_F3]) ad->fpsModeFixed = false;

			if(input->keysPressed[KEYCODE_F3]) {
				ad->captureMouse = false;
				SetCursorPos(ws->lastMousePosition.x, ws->lastMousePosition.y);
			}
		}

		ad->fpsMode = ad->captureMouse && windowHasFocus(windowHandle);
		if(ad->fpsMode) {
			int w,h;
			Vec2i wPos;
			getWindowProperties(windowHandle, &w, &h, 0, 0, &wPos.x, &wPos.y);

			SetCursorPos(wPos.x + w/2, wPos.y + h/2);
			input->lastMousePos = getMousePos(windowHandle,false);

			showCursor(false);
		} else {
			showCursor(true);
		}
	}

	// @AppLoop.

	// @GameMenu.

	// // Save.
	// if(input->keysPressed[KEYCODE_U]) {
	// 	char* saveName = "saveState1.sav";
	// 	char* saveFile = fillString("%s%s", SAVES_FOLDER, saveName);

	// 	DArray<VoxelMesh>* voxels = &ad->voxelData.voxels;

	// 	FILE* file = fopen(saveFile, "wb");
	// 	if(file) {
	// 		fwrite(ad->entityList.e, ad->entityList.size * sizeof(Entity), 1, file);

	// 		fwrite(&voxels->count, sizeof(int), 1, file);
	// 		fwrite(voxels->data, voxels->count * sizeof(VoxelMesh), 1, file);
	// 	}
	// }

	// // Load.
	// if(input->keysPressed[KEYCODE_I]) {
	// 	char* saveName = "saveState1.sav";
	// 	char* saveFile = fillString("%s%s", SAVES_FOLDER, saveName);

	// 	DArray<VoxelMesh>* voxels = &ad->voxelData.voxels;
	// 	voxels->clear();

	// 	FILE* file = fopen(saveFile, "wb");
	// 	if(file) {
	// 		fread(ad->entityList.e, ad->entityList.size * sizeof(Entity), 1, file);

	// 		int count = 0;
	// 		fread(&count, sizeof(int), 1, file);

	// 		voxels->reserve(count);
	// 		voxels->count = count;
	// 		fwrite(voxels->data, voxels->count * sizeof(VoxelMesh), 1, file);
	// 	}

	// 	for(int i = 0; i < ad->voxelData.voxelHashSize; i++) {
	// 		ad->voxelData.voxelHash[i].clear();
	// 	}
	// }

	if(ad->gameMode == GAME_MODE_MENU) {
		globalCommandList = &ad->commandList2d;

		Rect sr = getScreenRect(ws);
		Vec2 top = rectT(sr);
		float rHeight = rectH(sr);
		float rWidth = rectW(sr);

		int titleFontHeight = ds->fontHeight * ws->windowScale * 6.0f;
		int optionFontHeight = titleFontHeight * 0.5f;
		Font* titleFont = getFont(FONT_SOURCESANS_PRO, titleFontHeight);
		Font* font = getFont(FONT_SOURCESANS_PRO, optionFontHeight);

		Vec4 cBackground = vec4(hslToRgbFloat(0.63f,0.3f,0.13f),1);
		Vec4 cTitle = vec4(1,1);
		Vec4 cTitleShadow = vec4(0,0,0,1);
		Vec4 cOption = vec4(0.5f,1);
		Vec4 cOptionActive = vec4(0.9f,1);
		Vec4 cOptionShadow1 = vec4(0,1);
		Vec4 cOptionShadow2 = vec4(hslToRgbFloat(0.0f,0.5f,0.5f), 1);
		Vec4 cOptionShadow = vec4(0,1);

		float titleShadowSize = titleFontHeight * 0.07f;
		float optionShadowSize = optionFontHeight * 0.07f;

		float buttonAnimSpeed = 4;

		float optionOffset = optionFontHeight*1.2f;
		float settingsOffset = rWidth * 0.15f;


		MainMenu* menu = &ad->menu;

		float volumeMid = 0.5f;

		bool selectionChange = false;

		if(input->keysPressed[KEYCODE_DOWN]) {
			addTrack("ui\\select.wav", volumeMid);
			menu->activeId++;
			selectionChange = true;
		}
		if(input->keysPressed[KEYCODE_UP]) {
			addTrack("ui\\select.wav", volumeMid);
			menu->activeId--;
			selectionChange = true;
		}

		if(menu->currentId > 0)
			menu->activeId = mod(menu->activeId, menu->currentId);

		{
			if(selectionChange) {
				menu->buttonAnimState = 0;
			}

			menu->buttonAnimState += ad->dt * buttonAnimSpeed;
			float anim = (cos(menu->buttonAnimState) + 1)/2.0f;
			anim = powf(anim, 0.5f);
			Vec4 cOptionShadowActive = vec4(0,1);
			cOptionShadowActive.rgb = lerp(anim, cOptionShadow1.rgb, cOptionShadow2.rgb);

			menu->currentId = 0;
			menu->font = font;
			menu->cOption = cOption;
			menu->cOptionActive = cOptionActive;
			menu->cOptionShadow = cOptionShadow;
			menu->cOptionShadowActive = cOptionShadowActive;
			menu->optionShadowSize = optionShadowSize;
			menu->pressedEnter = input->keysPressed[KEYCODE_RETURN];
		}

		dcRect(sr, cBackground);

		if(menu->screen == MENU_SCREEN_MAIN) {

			Vec2 p = top - vec2(0, rHeight*0.2f);
			dcText("Voxel Game", titleFont, p, cTitle, vec2i(0,0), 0, titleShadowSize, cTitleShadow);

			bool gameRunning = menu->gameRunning;

			int optionCount = gameRunning ? 4 : 3;
			p.y = rectCen(sr).y + ((optionCount-1)*optionOffset)/2;

			if(gameRunning) {
				if(menuOption(menu, "Resume", p, vec2i(0,0)) || 
				   input->keysPressed[KEYCODE_ESCAPE]) {
					input->keysPressed[KEYCODE_ESCAPE] = false;

					ad->gameMode = GAME_MODE_MAIN;
				}
				p.y -= optionOffset;
			}

			if(menuOption(menu, "New Game", p, vec2i(0,0))) {
				addTrack("ui\\start.wav");
				ad->gameMode = GAME_MODE_LOAD;
				ad->newGame = true;
			}

			p.y -= optionOffset;
			if(menuOption(menu, "Settings", p, vec2i(0,0))) {
				addTrack("ui\\menuPush.wav", volumeMid);

				menu->screen = MENU_SCREEN_SETTINGS;
				menu->activeId = 0;
			}

			p.y -= optionOffset;
			if(menuOption(menu, "Exit", p, vec2i(0,0))) {
				*isRunning = false;
			}

		} else if(menu->screen == MENU_SCREEN_SETTINGS) {

			Vec2 p = top - vec2(0, rHeight*0.2f);
			dcText("Settings", titleFont, p, cTitle, vec2i(0,0), 0, titleShadowSize, cTitleShadow);

			int optionCount = 4;
			p.y = rectCen(sr).y + ((optionCount-1)*optionOffset)/2;

			// List settings.

			p.x = settingsOffset;
			if(menuOption(menu, "Field of view", p, vec2i(-1,0))) {
			}

			p.y -= optionOffset;
			if(menuOption(menu, "Resolution", p, vec2i(-1,0))) {
			}

			p.y -= optionOffset;
			if(menuOption(menu, "Mouse Sensitivity", p, vec2i(-1,0))) {
			}

			// 

			p.x = rWidth * 0.5f;
			p.y -= optionOffset;
			if(menuOption(menu, "Back", p, vec2i(0,0)) || 
			      input->keysPressed[KEYCODE_ESCAPE] ||
			      input->keysPressed[KEYCODE_BACKSPACE]) {
				addTrack("ui\\menuPop.wav", volumeMid);

				menu->screen = MENU_SCREEN_MAIN;
				menu->activeId = 0;
			}

		}
	}

	if(ad->gameMode == GAME_MODE_LOAD) {

		globalCommandList = &ad->commandList2d;

		int titleFontHeight = ds->fontHeight * ws->windowScale * 8.0f;
		Font* titleFont = getFont(FONT_SOURCESANS_PRO, titleFontHeight);

		Rect sr = getScreenRect(ws);

		dcRect(sr, vec4(0,1));
		dcText("Loading", titleFont, rectCen(sr), vec4(1,1), vec2i(0,-1));

		// @InitNewGame.

		if(!ad->loading) {
			ad->loading = true;

			bool hasSaveState;
			char* saveFile = fillString("%s%s", SAVES_FOLDER, SAVE_STATE1);
			if(fileExists(saveFile)) hasSaveState = true;

			// Pre work.
			{
				ad->blockMenuSelected = 0;

				// Clear voxel hash and meshes.
				{
					DArray<int>* voxelHash = ad->voxelData.voxelHash;
					for(int i = 0; i < ad->voxelData.voxelHashSize; i++) {
						voxelHash[i].clear();
					}

					DArray<VoxelMesh>* voxels = &ad->voxelData.voxels;
					for(int i = 0; i < voxels->count; i++) {
						freeVoxelMesh(voxels->data + i);
					}

					voxels->clear();
				}
			}

			// Load SaveState.
			if(!ad->newGame && hasSaveState) {
				DArray<VoxelMesh>* voxels = &ad->voxelData.voxels;
				voxels->clear();

				FILE* file = fopen(saveFile, "rb");
				if(file) {
					fread(ad->entityList.e, ad->entityList.size * sizeof(Entity), 1, file);

					int count = 0;
					fread(&count, sizeof(int), 1, file);

					voxels->reserve(count);
					voxels->count = count;

					for(int i = 0; i < voxels->count; i++) {
						VoxelMesh* m = voxels->data + i;
						initVoxelMesh(m, vec2i(0,0));

						fread(&m->coord, sizeof(Vec2i), 1, file);
						fread(m->voxels, VOXEL_SIZE * sizeof(uchar), 1, file);
						fread(m->lighting, VOXEL_SIZE * sizeof(uchar), 1, file);
					}

					fread(&startX, sizeof(int), 1, file);
					fread(&startY, sizeof(int), 1, file);
				}

				for(int i = 0; i < voxels->count; i++) {
					VoxelMesh* m = voxels->data + i;
					m->generated = true;
					m->upToDate = false;
					m->meshUploaded = false;

					addVoxelMesh(&ad->voxelData, m->coord, i);
				}

				EntityList* entityList = &ad->entityList;
				Entity* player = 0;
				Entity* camera = 0;
				for(int i = 0; i < entityList->size; i++) {
					Entity* e = entityList->e + i;
					if(e->type == ET_Player) player = e;
					else if(e->type == ET_Camera) camera = e;

					if(player && camera) break;
				}

				ad->player = player;
				ad->cameraEntity = camera;

			} else {

				// New Game.

				for(int i = 0; i < ad->entityList.size; i++) ad->entityList.e[i].init = false;

				// Init player.
				{
					float v = randomFloatPCG(0,M_2PI,0.001f);
					Vec3 startRot = vec3(v,0,0);

					Entity player;
					Vec3 playerDim = vec3(0.8f, 0.8f, 1.8f);
					float camOff = playerDim.z*0.5f - playerDim.x*0.25f;
					initEntity(&player, ET_Player, vec3(0,0,40), playerDim, vec3(0,0,camOff));
					player.rot = startRot;
					player.playerOnGround = false;
					strCpy(player.name, "Player");
					
					ad->player = addEntity(&ad->entityList, &player);
				}

				// Debug cam.
				{
					Entity freeCam;
					initEntity(&freeCam, ET_Camera, vec3(35,35,32), vec3(0,0,0), vec3(0,0,0));
					strCpy(freeCam.name, "Camera");
					ad->cameraEntity = addEntity(&ad->entityList, &freeCam);
				}

				startX = randomIntPCG(0,1000000);
				startY = randomIntPCG(0,1000000);
			}

			// Load voxel meshes around the player at startup.
			{
				Vec2i pPos = coordToMesh(ad->player->pos);
				for(int y = -1; y < 2; y++) {
					for(int x = -1; x < 2; x++) {
						Vec2i coord = pPos - vec2i(x,y);

						VoxelMesh* m = getVoxelMesh(&ad->voxelData, coord);
						makeMesh(m, &ad->voxelData);
					}
				}
			}
		}

		if(threadQueueFinished(globalThreadQueue)) {

			// Push the player up until he is right above the ground.

			Entity* player = ad->player;
			while(collisionVoxelWidthBox(&ad->voxelData, player->pos, player->dim)) {
				player->pos.z += 2;
			}

			ad->loading = false;
			ad->gameMode = GAME_MODE_MAIN;
			ad->startFade = 0;

			input->keysPressed[KEYCODE_F3] = true;
		}

	}

	if(ad->gameMode == GAME_MODE_MAIN) {

	if(input->keysPressed[KEYCODE_ESCAPE]) {
		ad->gameMode = GAME_MODE_MENU;
		ad->menu.gameRunning = true;
		ad->menu.activeId = 0;
	}

	Entity* player = ad->player;
	Entity* camera = ad->cameraEntity;

	if(input->keysPressed[KEYCODE_F4]) {
		if(ad->playerMode) {
			camera->pos = player->pos + player->camOff;
			camera->dir = player->dir;
			camera->rot = player->rot;
			camera->rotAngle = player->rotAngle;
		}
		ad->playerMode = !ad->playerMode;
	}

	if(input->mouseWheel) {
		ad->blockMenuSelected += -input->mouseWheel;
		ad->blockMenuSelected = mod(ad->blockMenuSelected, arrayCount(ad->blockMenu));
	}

	if(input->keysPressed[KEYCODE_0]) ad->blockMenuSelected = 9;
	for(int i = 0; i < 9; i++) {
		if(input->keysPressed[KEYCODE_0 + i+1]) ad->blockMenuSelected = i;
	}

	if(ad->playerMode) {
		camera->vel = vec3(0,0,0);
	} else {
		player->vel = vec3(0,0,0);
	}

	if(!ad->playerMode && input->keysPressed[KEYCODE_SPACE]) {
		player->pos = camera->pos;
		player->dir = camera->dir;
		player->rot = camera->rot;
		player->rotAngle = camera->rotAngle;
		player->vel = camera->vel;
		ad->playerMode = true;
		input->keysPressed[KEYCODE_SPACE] = false;
		input->keysDown[KEYCODE_SPACE] = false;
	}

	// spawn bomb
	#if 0
	{
		bool spawnBomb = false;
		if(input->mouseButtonPressed[2]) {
			spawnBomb = true;
			ad->bombSpawnTimer = 0;
		}

		if(input->mouseButtonDown[2]) {
			ad->bombSpawnTimer += ad->dt;
		}

		if(ad->bombSpawnTimer >= ad->bombFireInterval) {
			spawnBomb = true;
			ad->bombSpawnTimer = 0;
		}

		if(spawnBomb) {
			Entity b;
			Vec3 bombPos = ad->activeCam.pos + ad->activeCam.look*4;
			initEntity(&b, ET_Rocket, bombPos, ad->activeCam.look, vec3(0.5f), vec3(0,0,0));
			b.vel = ad->activeCam.look*300;
			b.acc = ad->activeCam.look*200;
			b.isMoving = true;

			addEntity(&ad->entityList, &b);
		}
	}
	#endif


	TIMER_BLOCK_BEGIN_NAMED(entities, "Upd Entities");
	// @update entities
	for(int i = 0; i < ad->entityList.size; i++) {
		Entity* e = &ad->entityList.e[i];
		if(!e->init) continue;

		float dt = ad->dt;
		Vec3 up = vec3(0,0,1);

		switch(e->type) {
			case ET_Player: {
				if(ad->playerMode == false) continue;

				Camera cam = getCamData(e->pos, e->rot);
				e->acc = vec3(0,0,0);

				bool rightLock = true;
				float runBoost = 1.5f;
				float speed = 30;

				if((!ad->fpsMode && input->mouseButtonDown[1]) || ad->fpsMode) {
					float turnRate = ad->dt*ad->settings.mouseSensitivity;
					e->rot.y -= turnRate * input->mouseDelta.y;
					e->rot.x -= turnRate * input->mouseDelta.x;

					float margin = 0.00001f;
					clamp(&e->rot.y, -M_PI_2+margin, M_PI_2-margin);
					e->rot.x = modFloat(e->rot.x, (float)M_PI*2);
				}

				if( input->keysDown[KEYCODE_W] || input->keysDown[KEYCODE_A] || input->keysDown[KEYCODE_S] || 
					input->keysDown[KEYCODE_D]) {

					if(rightLock || input->keysDown[KEYCODE_CTRL]) cam.look = cross(up, cam.right);

					Vec3 acceleration = vec3(0,0,0);
					if(input->keysDown[KEYCODE_SHIFT]) speed *= runBoost;
					if(input->keysDown[KEYCODE_W]) acceleration +=  normVec3(cam.look);
					if(input->keysDown[KEYCODE_S]) acceleration += -normVec3(cam.look);
					if(input->keysDown[KEYCODE_D]) acceleration +=  normVec3(cam.right);
					if(input->keysDown[KEYCODE_A]) acceleration += -normVec3(cam.right);
					e->acc += normVec3(acceleration)*speed;
				}

				e->acc.z = 0;

				if(ad->playerMode) {
					// if(input->keysPressed[KEYCODE_SPACE]) {
					if(input->keysDown[KEYCODE_SPACE]) {
						if(player->playerOnGround) {
							player->vel += up*7.0f;
							player->playerOnGround = false;
						}
					}
				}

				float gravity = 20.0f;
				if(!e->playerOnGround) e->acc += -up*gravity;
				e->vel = e->vel + e->acc*dt;
				float friction = 0.01f;
				e->vel.x *= pow(friction,dt);
				e->vel.y *= pow(friction,dt);
				// e->vel *= 0.9f;

				if(e->playerOnGround) e->vel.z = 0;

				bool playerGroundCollision = false;
				bool playerCeilingCollision = false;
				bool playerSideCollision = false;

				if(e->vel != vec3(0,0,0)) {
					Vec3 pPos = e->pos;
					Vec3 pSize = e->dim;

					Vec3 nPos = pPos + -0.5f*e->acc*dt*dt + e->vel*dt;

					int collisionCount = 0;
					bool collision = true;
					while(collision) {

						float minDistance;
						Vec3 collisionBox;
						collision = collisionVoxelWidthBox(&ad->voxelData, nPos, pSize, &minDistance, &collisionBox);

						if(collision) {
							collisionCount++;

							float minDistance;
							Vec3 dir = vec3(0,0,0);
							int face;

								// check all the 6 planes and take the one with the shortest distance
							for(int i = 0; i < 6; i++) {
								Vec3 n;
								if(i == 0) 		n = vec3(1,0,0);
								else if(i == 1) n = vec3(-1,0,0);
								else if(i == 2) n = vec3(0,1,0);
								else if(i == 3) n = vec3(0,-1,0);
								else if(i == 4) n = vec3(0,0,1);
								else if(i == 5) n = vec3(0,0,-1);

									// assuming voxel size is 1
									// this could be simpler because the voxels are axis aligned
								Vec3 p = collisionBox + (n * ((pSize/2) + 0.5));
								float d = -dot(p, n);
								float d2 = dot(nPos, n);

									// distances are lower then zero in this case where the point is 
									// not on the same side as the normal
								float distance = d + d2;

								if(i == 0 || distance > minDistance) {
									minDistance = distance;
									dir = n;
									face = i;
								}
							}

							float error = 0.0001f;
								// float error = 0;
							nPos += dir*(-minDistance + error);

							if(face == 5) playerCeilingCollision = true;
							else if(face == 4) playerGroundCollision = true;
							else playerSideCollision = true;
						}

						// something went wrong and we reject the move, for now
						if(collisionCount > 5) {
							// nPos = ad->playerPos;

							nPos.z += 5;
							// ad->playerVel = vec3(0,0,0);
							break;	
						}
					}

					float stillnessThreshold = 0.0001f;
					if(valueBetween(e->vel.z, -stillnessThreshold, stillnessThreshold)) {
						e->vel.z = 0;
					}

					if(playerCeilingCollision) {
						e->vel.z = 0;
					}

					if(playerSideCollision) {
						float sideFriction = 0.0010f;
						e->vel.x *= pow(sideFriction,dt);
						e->vel.y *= pow(sideFriction,dt);
					}

					if(collisionCount > 5) {
						e->vel = vec3(0,0,0);
					}

					e->pos = nPos;
				}

				// raycast for touching ground
				if(ad->playerMode) {
					if(playerGroundCollision && e->vel.z <= 0) {
						e->playerOnGround = true;
						e->vel.z = 0;
					}

					if(e->playerOnGround) {
						Vec3 pos = e->pos;
						Vec3 size = e->dim;
						Rect3 box = rect3CenDim(pos, size);

						bool groundCollision = false;

						for(int i = 0; i < 4; i++) {
							Vec3 gp;
							if(i == 0) 		gp = box.min + size*vec3(0,0,0);
							else if(i == 1) gp = box.min + size*vec3(1,0,0);
							else if(i == 2) gp = box.min + size*vec3(0,1,0);
							else if(i == 3) gp = box.min + size*vec3(1,1,0);

							// drawCube(&ad->pipelineIds, block, vec3(1,1,1)*1.01f, vec4(1,0,1,1), 0, vec3(0,0,0));

							float raycastThreshold = 0.01f;
							gp -= up*raycastThreshold;

							Vec3 block = coordToVoxelCoord(gp);
							uchar* blockType = getBlockFromCoord(&ad->voxelData, gp);

							if(*blockType > 0) {
								groundCollision = true;
								break;
							}
						}

						if(groundCollision) {
							if(e->vel.z <= 0) e->playerOnGround = true;
						} else {
							e->playerOnGround = false;
						}
					}
				}


			} break;

			case ET_Camera: {
				Camera cam = getCamData(e->pos, e->rot);

				e->acc = vec3(0,0,0);

				bool rightLock = false;
				float runBoost = 2.0f;
				float speed = 150;
				if(input->keysDown[KEYCODE_T]) speed = 1000;

				if((!ad->fpsMode && input->mouseButtonDown[1]) || ad->fpsMode) {
					float turnRate = ad->dt*ad->settings.mouseSensitivity;
					e->rot.y -= turnRate * input->mouseDelta.y;
					e->rot.x -= turnRate * input->mouseDelta.x;

					float margin = 0.00001f;
					clamp(&e->rot.y, -M_PI_2+margin, M_PI_2-margin);
					e->rot.x = modFloat(e->rot.x, (float)M_PI*2);
				}

				if( input->keysDown[KEYCODE_W] || input->keysDown[KEYCODE_A] || input->keysDown[KEYCODE_S] || 
					input->keysDown[KEYCODE_D] || input->keysDown[KEYCODE_E] || input->keysDown[KEYCODE_Q]) {

					if(rightLock || input->keysDown[KEYCODE_CTRL]) cam.look = cross(up, cam.right);

					Vec3 acceleration = vec3(0,0,0);
					if(input->keysDown[KEYCODE_SHIFT]) speed *= runBoost;
					if(input->keysDown[KEYCODE_W]) acceleration +=  normVec3(cam.look);
					if(input->keysDown[KEYCODE_S]) acceleration += -normVec3(cam.look);
					if(input->keysDown[KEYCODE_D]) acceleration +=  normVec3(cam.right);
					if(input->keysDown[KEYCODE_A]) acceleration += -normVec3(cam.right);
					if(input->keysDown[KEYCODE_E]) acceleration +=  normVec3(up);
					if(input->keysDown[KEYCODE_Q]) acceleration += -normVec3(up);
					e->acc += normVec3(acceleration)*speed;
				}

				e->vel = e->vel + e->acc*dt;
				float friction = 0.01f;
				e->vel = e->vel * pow(friction,dt);

				if(e->vel != vec3(0,0,0)) {
					e->pos = e->pos - 0.5f*e->acc*dt*dt + e->vel*dt;
				}

			} break;

			case ET_Rocket: {
				// float gravity = 20.0f;
				float gravity = 1.0f;

				e->acc += -up*gravity;
				e->vel = e->vel + e->acc*dt;

				float friction = 0.01f;
				e->vel.x *= pow(friction,dt);
				e->vel.y *= pow(friction,dt);
				e->vel.z *= pow(friction,dt);

				if(e->vel != vec3(0,0,0)) {
					Vec3 pPos = e->pos;
					Vec3 pSize = e->dim;

					Vec3 nPos = pPos + -0.5f*e->acc*dt*dt + e->vel*dt;

					bool collision = false;

					Rect3 box = rect3CenDim(nPos, pSize);
					Vec3i voxelMin = coordToVoxel(box.min);
					Vec3i voxelMax = coordToVoxel(box.max+1);

					collision = false;
					Vec3 collisionBox;

					for(int x = voxelMin.x; x < voxelMax.x; x++) {
						for(int y = voxelMin.y; y < voxelMax.y; y++) {
							for(int z = voxelMin.z; z < voxelMax.z; z++) {
								Vec3i coord = vec3i(x,y,z);
								uchar* block = getBlockFromVoxel(&ad->voxelData, coord);

								if(*block > 0) {
									collisionBox = voxelToVoxelCoord(coord);
									collision = true;
									goto forGoto;
								}
							}
						}
					} forGoto:

					if(collision) {
						e->init = false;
						e->exploded = true;
					}

					Vec2i updateMeshList[8];
					int updateMeshListSize = 0;

					if(e->exploded) {
						Vec3 startPos = collisionBox;
						float sRad = 10;

						// float resolution = M_2PI;
						// float vStep = 0.5f;
						float resolution = M_PI;
						float vStep = 0.75f;

						int itCount = sRad*resolution;
						for(int it = 0; it < itCount+1; it++) {
							float off = degreeToRadian(it * (360/(float)itCount));
							Vec3 dir = rotateVec3(normVec3(vec3(0,1,0)), off, vec3(1,0,0));
							float off2 = sin(off/(float)2)*sRad;

							float rad = (dir*sRad).y;
							for(int i = 0; i < 2; i++) {
								Vec3 pos;
								if(i == 0) pos = startPos + vec3(0,off2,0);
								else pos = startPos + vec3(0,-off2,0);

								int itCount = rad*resolution;
								for(int it = 0; it < itCount+1; it++) {
									float off = degreeToRadian(it * (360/(float)itCount));
									Vec3 dir = rotateVec3(normVec3(vec3(1,0,0)), off, vec3(0,-1,0));
									Vec3 p = pos + dir*rad;

									float cubeSize = 1.0f;

									// dcCube({coordToVoxelCoord(pos + dir*rad), vec3(cubeSize), vec4(1,0.5f,0,1), 0, vec3(0,0,0)});
									// dcCube({coordToVoxelCoord(pos - dir*rad), vec3(cubeSize), vec4(1,0.5f,0,1), 0, vec3(0,0,0)});

									*getBlockFromCoord(&ad->voxelData, pos+dir*rad) = 0; 
									*getLightingFromCoord(&ad->voxelData, pos+dir*rad) = globalLumen; 
									*getBlockFromCoord(&ad->voxelData, pos-dir*rad) = 0; 
									*getLightingFromCoord(&ad->voxelData, pos-dir*rad) = globalLumen; 

									for(int it = 0; it < 2; it++) {
										bool found = false;
										Vec2i mc;
										if(it == 0) mc = coordToMesh(pos+dir*rad);
										else mc = coordToMesh(pos-dir*rad);
										for(int i = 0; i < updateMeshListSize; i++) {
											if(updateMeshList[i] == mc) {
												found = true;
												break;
											}
										}
										if(!found) {
											updateMeshList[updateMeshListSize++] = mc;
										}

									}

									float off2 = sin(off/(float)2)*rad;
									for(float z = 0; z < (dir*rad).x; z += vStep) {
										// dcCube({coordToVoxelCoord(pos + vec3(off2,0, z)), vec3(cubeSize), vec4(0,0.5f,1,1), 0, vec3(0,0,0)});
										// dcCube({coordToVoxelCoord(pos + vec3(off2,0,-z)), vec3(cubeSize), vec4(0,0.5f,1,1), 0, vec3(0,0,0)});
										// dcCube({coordToVoxelCoord(pos - vec3(off2,0,z)), vec3(cubeSize), vec4(0,0.5f,1,1), 0, vec3(0,0,0)});
										// dcCube({coordToVoxelCoord(pos - vec3(off2,0,-z)), vec3(cubeSize), vec4(0,0.5f,1,1), 0, vec3(0,0,0)});

										*getBlockFromCoord(&ad->voxelData,    pos + vec3(off2,0, z)) = 0; 
										*getLightingFromCoord(&ad->voxelData, pos + vec3(off2,0, z)) = globalLumen; 
										*getBlockFromCoord(&ad->voxelData,    pos + vec3(off2,0,-z)) = 0; 
										*getLightingFromCoord(&ad->voxelData, pos + vec3(off2,0,-z)) = globalLumen; 
										*getBlockFromCoord(&ad->voxelData,    pos - vec3(off2,0, z)) = 0; 
										*getLightingFromCoord(&ad->voxelData, pos - vec3(off2,0, z)) = globalLumen; 
										*getBlockFromCoord(&ad->voxelData,    pos - vec3(off2,0,-z)) = 0; 
										*getLightingFromCoord(&ad->voxelData, pos - vec3(off2,0,-z)) = globalLumen; 
									}
								}
							}
						}

						for(int i = 0; i < updateMeshListSize; i++) {
							VoxelMesh* m = getVoxelMesh(&ad->voxelData, updateMeshList[i]);
							m->upToDate = false;
							m->meshUploaded = false;
							m->modifiedByUser = true;
						}
					}

					e->pos = nPos;

					dcCube(e->pos, e->dim, vec4(1,0.5f,0,1), 0, vec3(0,0,0));
				}
			} break;

			default: {

			};
		}
	}
	TIMER_BLOCK_END(entities);

	if(ad->playerMode) {
		ad->activeCam = getCamData(ad->player->pos, ad->player->rot, ad->player->camOff);
	} else {
		ad->activeCam = getCamData(ad->cameraEntity->pos, ad->cameraEntity->rot, ad->cameraEntity->camOff);
	}

	// Selecting blocks and modifying them.
	if(ad->playerMode) {
		ad->blockSelected = false;

		// get block in line
		Vec3 startDir = ad->activeCam.look;
		Vec3 startPos = player->pos + player->camOff;

		Vec3 newPos = startPos;
		int smallerAxis[2];
		int biggestAxis = getBiggestAxis(startDir, smallerAxis);

		bool intersection = false;
		Vec3 intersectionBox;

		int intersectionFace;

		for(int i = 0; i < SELECTION_RADIUS; i++) {
			newPos = newPos + normVec3(startDir);

			Vec3 coords[9];
			int coordsSize = 0;

			Vec3 blockCoords = voxelToVoxelCoord(coordToVoxel(newPos));

			// we populate 8 blocks around the biggest axis
			for(int y = -1; y < 2; y++) {
				for(int x = -1; x < 2; x++) {
					Vec3 dir = vec3(0,0,0);
					dir.e[smallerAxis[0]] = y;
					dir.e[smallerAxis[1]] = x;

					coords[coordsSize++] = blockCoords + dir;
				}
			}

			bool firstIntersection = true;
			float minDistance = -1;

			for(int i = 0; i < coordsSize; i++) {
				Vec3 block = coords[i];

				uchar* blockType = getBlockFromCoord(&ad->voxelData, block);
				Vec3 temp = voxelToVoxelCoord(coordToVoxel(block));

				// Vec4 c;
				// if(i == 0) c = vec4(1,0,1,1);
				// else c = vec4(0,0,1,1);
				// glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				// drawCube(&ad->pipelineIds, temp, vec3(1.0f,1.0f,1.0f), c, 0, vec3(0,0,0));

				// if(blockType > 0) {
				if(*blockType > 0) {
					Vec3 iBox = voxelToVoxelCoord(coordToVoxel(block));
					float distance;
					int face;
					bool inter = boxRaycast(startPos, startDir, rect3CenDim(iBox, vec3(1,1,1)), &distance, &face);

					if(inter) {
						if(firstIntersection) {
							minDistance = distance;
							intersectionBox = iBox;
							firstIntersection = false;
							intersectionFace = face;
						} else if(distance < minDistance) {
							minDistance = distance;
							intersectionBox = iBox;
							intersectionFace = face;
						}

						intersection = true;
					}
				}
			}

			if(intersection) break;
		}

		if(intersection) {
			ad->selectedBlock = intersectionBox;
			ad->blockSelected = true;

			Vec3 faceDir = vec3(0,0,0);
			if(intersectionFace == 0) faceDir = vec3(-1,0,0);
			else if(intersectionFace == 1) faceDir = vec3(1,0,0);
			else if(intersectionFace == 2) faceDir = vec3(0,-1,0);
			else if(intersectionFace == 3) faceDir = vec3(0,1,0);
			else if(intersectionFace == 4) faceDir = vec3(0,0,-1);
			else if(intersectionFace == 5) faceDir = vec3(0,0,1);
			ad->selectedBlockFaceDir = faceDir;

			if(ad->playerMode && ad->fpsMode) {
				VoxelMesh* vm = getVoxelMesh(&ad->voxelData, coordToMesh(intersectionBox));

				uchar* block = getBlockFromCoord(&ad->voxelData, intersectionBox);
				uchar* lighting = getLightingFromCoord(&ad->voxelData, intersectionBox);

				bool mouse1 = input->mouseButtonPressed[0];
				bool mouse2 = input->mouseButtonPressed[1];
				bool placeBlock = (!ad->fpsMode && ad->pickMode && mouse1) || (ad->fpsMode && mouse1);
				bool removeBlock = (!ad->fpsMode && !ad->pickMode && mouse1) || (ad->fpsMode && mouse2);

				if(placeBlock || removeBlock) {
					vm->upToDate = false;
					vm->meshUploaded = false;
					vm->modifiedByUser = true;

					// if block at edge of mesh, we have to update the mesh on the other side too
					Vec2i currentCoord = coordToMesh(intersectionBox);
					for(int i = 0; i < 4; i++) {
						Vec3 offset;
						if(i == 0) offset = vec3(1,0,0);
						else if(i == 1) offset = vec3(-1,0,0);
						else if(i == 2) offset = vec3(0,1,0);
						else if(i == 3) offset = vec3(0,-1,0);

						Vec2i mc = coordToMesh(intersectionBox + offset);
						if(mc != currentCoord) {
							VoxelMesh* edgeMesh = getVoxelMesh(&ad->voxelData, mc);
							edgeMesh->upToDate = false;
							edgeMesh->meshUploaded = false;
							edgeMesh->modifiedByUser = true;
						}
					}
				}

				if(placeBlock) {
					Vec3 boxToCamDir = startPos - intersectionBox;
					Vec3 sideBlock = coordToVoxelCoord(intersectionBox + faceDir);
					Vec3i voxelSideBlock = coordToVoxel(sideBlock);

					// CodeDuplication:
					// get mesh coords that touch the player box
					Rect3 box = rect3CenDim(player->pos, player->dim);
					Vec3i voxelMin = coordToVoxel(box.min);
					Vec3i voxelMax = coordToVoxel(box.max+1);
					bool collision = false;

					for(int x = voxelMin.x; x < voxelMax.x; x++) {
						for(int y = voxelMin.y; y < voxelMax.y; y++) {
							for(int z = voxelMin.z; z < voxelMax.z; z++) {
								Vec3i coord = vec3i(x,y,z);

								if(coord == voxelSideBlock) {
									collision = true;
									goto forBreak;
								}
							}
						}
					} forBreak:

					if(!collision) {
						uchar* sideBlockType = getBlockFromVoxel(&ad->voxelData, voxelSideBlock);
						uchar* sideBlockLighting = getLightingFromVoxel(&ad->voxelData, voxelSideBlock);

						*sideBlockType = ad->blockMenu[ad->blockMenuSelected];
						*sideBlockLighting = 0;
					}
				} else if(removeBlock) {
					if(*block > 0) {
						*block = 0;
						*lighting = globalLumen;
					}
				}
			}
		}
	}

	// Main view proj setup.
	Mat4 view, proj;
	{
		viewMatrix(&view, ad->activeCam.pos, -ad->activeCam.look, ad->activeCam.up, ad->activeCam.right);
		projMatrix(&proj, degreeToRadian(ad->fieldOfView), ad->aspectRatio, ad->nearPlane, ad->farPlane);

		bindFrameBuffer(FRAMEBUFFER_3dMsaa);

		pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VIEW, view);
		pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_PROJ, proj);
	}	

	// Draw cubemap.
	{
		bindShader(SHADER_CUBEMAP);
		glBindTextures(0, 1, &getCubemap(ad->skyBoxId)->id);
		glBindSamplers(0, 1, gs->samplers);

		Vec3 skyBoxRot;
		if(ad->playerMode) skyBoxRot = ad->player->rot;
		else skyBoxRot = ad->cameraEntity->rot;
		skyBoxRot.x += M_PI;

		Camera skyBoxCam = getCamData(vec3(0,0,0), skyBoxRot, vec3(0,0,0), vec3(0,1,0), vec3(0,0,1));
		pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_VIEW, viewMatrix(skyBoxCam.pos, -skyBoxCam.look, skyBoxCam.up, skyBoxCam.right));
		pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_PROJ, projMatrix(degreeToRadian(ad->fieldOfView), ad->aspectRatio, 0.001f, 2));
		pushUniform(SHADER_CUBEMAP, 2, CUBEMAP_UNIFORM_CLIPPLANE, false);

		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
		glDepthMask(false);
		glFrontFace(GL_CCW);
			glDrawArrays(GL_TRIANGLES, 0, 6*6);
		glFrontFace(GL_CW);
		glDepthMask(true);
		glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	}

	if(ad->reloadWorld) {
		ad->reloadWorld = false;

		if(threadQueueFinished(threadQueue)) {
			int radius = VIEW_DISTANCE/VOXEL_X;

			Vec3 pp = ad->activeCam.pos;
			Vec2i worldPos = coordToMesh(pp);

			for(int y = worldPos.y-radius; y < worldPos.y+radius; y++) {
				for(int x = worldPos.x-radius; x < worldPos.x+radius; x++) {
					Vec2i coord = vec2i(x, y);
					VoxelMesh* m = getVoxelMesh(&ad->voxelData, coord);
					m->upToDate = false;
					m->meshUploaded = false;
					m->generated = false;

					makeMesh(m, &ad->voxelData);
				}
			}
		} 
	}

	TIMER_BLOCK_BEGIN_NAMED(world, "Upd World");

	Vec2i* coordList = (Vec2i*)getTMemory(sizeof(Vec2i)*2000);
	int coordListSize = 0;

	// Collect voxel meshes to draw.
	{
		int meshGenerationCount = 0;
		int radCounter = 0;

		ad->voxelTriangleCount = 0;
		ad->voxelDrawCount = 0;

		Vec2i pPos = coordToMesh(ad->activeCam.pos);
		int radius = VIEW_DISTANCE/VOXEL_X;

		// generate the meshes around the player in a spiral by drawing lines and rotating
		// the directing every time we reach a corner
		for(int r = 0; r < radius; r++) {
			int lineLength = r == 0? 1 : 8*r;
			int segment = r*2;

			Vec2i lPos = pPos+r;

			int lLength = 0;
			Vec2i lDir = vec2i(0,-1);

			for(int lineId = 0; lineId < lineLength; lineId++) {
				if(r == 0) lPos = pPos;
				else {
					if(lLength == segment) {
						lLength = 0;
						lDir = vec2i(lDir.y, -lDir.x);
					}
					lLength++;

					lPos += lDir;
				}

				radCounter++;

				Vec2i coord = lPos;
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coord);

				if(!m->meshUploaded) {
					makeMesh(m, &ad->voxelData);
					meshGenerationCount++;

					if(!m->modifiedByUser) continue;
				}

				// frustum culling
				Vec3 cp = ad->activeCam.pos;
				Vec3 cl = ad->activeCam.look;
				Vec3 cu = ad->activeCam.up;
				Vec3 cr = ad->activeCam.right;

				float ar = ad->aspectRatio;
				float fov = degreeToRadian(ad->fieldOfView);
				// float ne = ad->nearPlane;
				// float fa = ad->farPlane;

				Vec3 left = rotateVec3(cl, fov*ar, cu);
				Vec3 right = rotateVec3(cl, -fov*ar, cu);
				Vec3 top = rotateVec3(cl, fov, cr);
				Vec3 bottom = rotateVec3(cl, -fov, cr);

				Vec3 normalLeftPlane = cross(cu, left);
				Vec3 normalRightPlane = cross(right, cu);
				Vec3 normalTopPlane = cross(cr, top);
				Vec3 normalBottomPlane = cross(bottom, cr);

				Vec3 boxPos = vec3(coord.x*VOXEL_X+VOXEL_X*0.5f, coord.y*VOXEL_Y+VOXEL_Y*0.5f, VOXEL_Z*0.5f);
				Vec3 boxSize = vec3(VOXEL_X, VOXEL_Y, VOXEL_Z);

				bool isIntersecting = true;	
				for(int test = 0; test < 4; test++) {

					Vec3 testNormal;
					if(test == 0) testNormal = normalLeftPlane;
					else if(test == 1) testNormal = normalRightPlane;
					else if(test == 2) testNormal = normalTopPlane;
					else if(test == 3) testNormal = normalBottomPlane;

					bool inside = false;
					for(int i = 0; i < 8; i++) {
						Vec3 off;
						switch (i) {
							case 0: off = vec3( 0.5f,  0.5f, -0.5f); break;
							case 1: off = vec3(-0.5f,  0.5f, -0.5f); break;
							case 2: off = vec3( 0.5f, -0.5f, -0.5f); break;
							case 3: off = vec3(-0.5f, -0.5f, -0.5f); break;
							case 4: off = vec3( 0.5f,  0.5f,  0.5f); break;
							case 5: off = vec3(-0.5f,  0.5f,  0.5f); break;
							case 6: off = vec3( 0.5f, -0.5f,  0.5f); break;
							case 7: off = vec3(-0.5f, -0.5f,  0.5f); break;
						}

						Vec3 boxPoint = boxPos + boxSize*off;
						Vec3 p = boxPoint - cp;

						if(dot(p, testNormal) < 0) {
							inside = true;
							break;
						}
					}

					if(!inside) {
						isIntersecting = false;
						break;
					}
				}

				if(isIntersecting) {
					// drawVoxelMesh(m);
					coordList[coordListSize++] = m->coord;

					// triangleCount += m->quadCount*4;
					ad->voxelTriangleCount += m->quadCount/(float)2;
					ad->voxelDrawCount++;
				}
			}
		}
	}

	SortPair* sortList = (SortPair*)getTMemory(sizeof(SortPair)*coordListSize);
	int sortListSize = 0;

	// Sort voxel meshes.
	{
		for(int i = 0; i < coordListSize; i++) {
			Vec2 c = meshToMeshCoord(coordList[i]).xy;
			float distanceToCamera = lenVec2(ad->activeCam.pos.xy - c);
			sortList[sortListSize++] = {distanceToCamera, i};
		}

		radixSortPair(sortList, sortListSize);

		for(int i = 0; i < sortListSize-1; i++) {
			assert(sortList[i].key <= sortList[i+1].key);
		}
	}

	TIMER_BLOCK_END(world);


	TIMER_BLOCK_BEGIN_NAMED(worldDraw, "Draw World");

	// Draw voxel world and reflection.
	{
		setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, voxelFogColor);

		// TIMER_BLOCK_NAMED("D World");
		// draw world without water
		{
			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, 2);
			}
		}

		// draw stencil
		{
			glEnable(GL_STENCIL_TEST);
			glStencilMask(0xFF);
			glClear(GL_STENCIL_BUFFER_BIT);
			glStencilFunc(GL_ALWAYS, 1, 0xFF);
			glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
			glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
			glDepthMask(GL_FALSE);

			glEnable(GL_CLIP_DISTANCE0);
			glEnable(GL_CLIP_DISTANCE1);

			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CLIPPLANE, true);
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE1, 0,0,1,-WATER_LEVEL_HEIGHT);
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE2, 0,0,-1,WATER_LEVEL_HEIGHT);

			pushUniform(SHADER_VOXEL, 1, VOXEL_UNIFORM_ALPHATEST, 0.5f);

			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, 1);
			}

			glDisable(GL_CLIP_DISTANCE0);
			glDisable(GL_CLIP_DISTANCE1);
			glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
			glDepthMask(GL_TRUE);
		}

		// draw reflection
		{	
			glStencilMask(0x00);
			glStencilFunc(GL_EQUAL, 1, 0xFF);

			bindFrameBuffer(FRAMEBUFFER_Reflection);
			glClearColor(0,0,0,0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

			Vec2i reflectionRes = ad->cur3dBufferRes;
			blitFrameBuffers(FRAMEBUFFER_3dMsaa, FRAMEBUFFER_Reflection, ad->cur3dBufferRes, GL_STENCIL_BUFFER_BIT, GL_NEAREST);

			glEnable(GL_CLIP_DISTANCE0);
			// glEnable(GL_CLIP_DISTANCE1);
			glEnable(GL_DEPTH_TEST);
			glFrontFace(GL_CCW);

				// draw cubemap reflection
				bindShader(SHADER_CUBEMAP);
				glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
				glBindTextures(0, 1, &getCubemap(ad->skyBoxId)->id);
				glBindSamplers(0, 1, gs->samplers);

				Vec3 skyBoxRot;
				if(ad->playerMode) skyBoxRot = ad->player->rot;
				else skyBoxRot = ad->cameraEntity->rot;
				skyBoxRot.x += M_PI;

				Camera skyBoxCam = getCamData(vec3(0,0,0), skyBoxRot, vec3(0,0,0), vec3(0,1,0), vec3(0,0,1));

				Mat4 viewMat; viewMatrix(&viewMat, skyBoxCam.pos, -skyBoxCam.look, skyBoxCam.up, skyBoxCam.right);
				Mat4 projMat; projMatrix(&projMat, degreeToRadian(ad->fieldOfView), ad->aspectRatio, 0.001f, 2);
				pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_VIEW, viewMat.e);
				pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_PROJ, projMat.e);

				pushUniform(SHADER_CUBEMAP, 2, CUBEMAP_UNIFORM_CLIPPLANE, true);
				pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_CPLANE1, 0,0,-1,WATER_LEVEL_HEIGHT);

				glDepthMask(false);
				// glFrontFace(GL_CCW);
				glDrawArrays(GL_TRIANGLES, 0, 6*6);
				// glFrontFace(GL_CW);
				glDepthMask(true);
				glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

			setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, voxelFogColor, vec3(0,0,WATER_LEVEL_HEIGHT*2 + 0.01f), vec3(1,1,-1));
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CLIPPLANE, true);
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE1, 0,0,-1,WATER_LEVEL_HEIGHT);

			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, 2);
			}
			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, 1);
			}

			glFrontFace(GL_CW);
			glDisable(GL_CLIP_DISTANCE0);
			// glDisable(GL_CLIP_DISTANCE1);
			glDisable(GL_STENCIL_TEST);
		}

		// draw reflection texture	
		{ 
			// 	// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			// 	// glBlendFunc(GL_SRC_COLOR, GL_ONE);
			// 	// glBlendFunc(GL_SRC_COLOR, GL_ONE_MINUS_SRC_ALPHA);
			// 	// glBlendFunc(GL_DST_COLOR, GL_ZERO);
			// 	// glBlendFunc(GL_SRC_COLOR, GL_ONE);

			// 	// glBlendFunc(GL_DST_COLOR, GL_ZERO);
			// 	// glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);
			// 	// glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);

			// 	// glBlendFunc(GL_ONE, GL_DST_COLOR);
			// 	// glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);
			// 	// glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);

			// 	// GL_DST_COLOR, GL_ZERO

			// 	// void glBlendFuncSeparate(	GLenum srcRGB,
			// 	//  	GLenum dstRGB,
			// 	//  	GLenum srcAlpha,
			// 	//  	GLenum dstAlpha);

			// // glBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ZERO, GL_ONE, GL_ONE);
			// // glBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ZERO, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			// glBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ZEROd, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


			// 	glBlendEquation(GL_FUNC_ADD);
			// 	// glBlendEquation(GL_FUNC_SUBTRACT);
			// 	// glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
			// 	// glBlendEquation(GL_MIN);
			// 	// glBlendEquation(GL_MAX);

			bindFrameBuffer(FRAMEBUFFER_3dMsaa);
			glDisable(GL_DEPTH_TEST);

			bindShader(SHADER_QUAD);
			drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), vec4(1,1,1,reflectionAlpha), rect(0,1,1,0), 
			         getFrameBuffer(FRAMEBUFFER_Reflection)->colorSlot[0]->id);

			glEnable(GL_DEPTH_TEST);

			// 	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			// 	glBlendEquation(GL_FUNC_ADD);
		}

		// draw water
		{
			setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, voxelFogColor);
			pushUniform(SHADER_VOXEL, 1, VOXEL_UNIFORM_ALPHATEST, 0.5f);

			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, 1);
			}
		}
	}

	TIMER_BLOCK_END(worldDraw);


	// Draw player and selected block.
	{
		dcState(STATE_LINEWIDTH, 3);
		if(!ad->playerMode) {
			Camera cam = getCamData(ad->player->pos, ad->player->rot);
			Vec3 pCamPos = player->pos + player->camOff;
			float lineLength = 0.5f;

			dcLine(pCamPos, pCamPos + cam.look*lineLength, vec4(1,0,0,1));
			dcLine(pCamPos, pCamPos + cam.up*lineLength, vec4(0,1,0,1));
			dcLine(pCamPos, pCamPos + cam.right*lineLength, vec4(0,0,1,1));

			dcState(STATE_POLYGONMODE, POLYGON_MODE_LINE);
			dcCube(player->pos, player->dim, vec4(1,1,1,1), 0, vec3(0,0,0));
			dcState(STATE_POLYGONMODE, POLYGON_MODE_FILL);
		} else {
			if(ad->blockSelected) {
				dcDisable(STATE_CULL);
				Vec3 vs[4];
				getPointsFromQuadAndNormal(ad->selectedBlock + ad->selectedBlockFaceDir*0.5f*1.01f, ad->selectedBlockFaceDir, 1, vs);

				dcQuad(vs[0], vs[1], vs[2], vs[3], vec4(1,1,1,0.025f));
				dcEnable(STATE_CULL);

				dcState(STATE_POLYGONMODE, POLYGON_MODE_LINE);
				dcCube(ad->selectedBlock, vec3(1.01f), vec4(0.9f), 0, vec3(0,0,0));
				dcState(STATE_POLYGONMODE, POLYGON_MODE_FILL);
			}
		}
	}

	// Block selection menu.
	if(ad->playerMode) {
		globalCommandList = &ad->commandList2d;

		for(int i = 0; i < 10; i++) {
			ad->blockMenu[i] = i+1;
		}

		Vec2 res = vec2(ws->currentRes);
		float bottom = 0.950f;
		float size = 0.0527f;
		float dist = 0.5f;
		Vec2 iconSize = vec2(size * res.h);
		float iconDist = iconSize.w * dist;
			// int count = arrayCount(ad->blockMenu);
		int count = 10;

		float selectColor = 1.5f;
		float trans = 0.8f;

		float start = res.x*0.5f - ((count-1)*0.5f)*(iconDist+iconSize.w);
		for(int i = 0; i < count; i++) {
			Vec4 color = vec4(1,1,1,trans);
			float iconOff = 1;
			if(ad->blockMenuSelected == i) {
				color = vec4(1*selectColor,1*selectColor,1*selectColor,trans);
				iconOff = 1.2f;
			}

			Vec2 pos = vec2(start + i*(iconSize.w + iconDist), -res.y*bottom);
			// dcRect({rectCenDim(pos, iconSize*1.2f*iconOff), rect(0,0,1,1), vec4(vec3(0.1f),trans), 1});
			dcRect(rectCenDim(pos, iconSize*1.2f*iconOff), rect(0,0,1,1), vec4(0,0,0,0.85f), 1);

			uint textureId = gs->textures3d[0].id;
			dcRect(rectCenDim(pos, iconSize*iconOff), rect(0,0,1,1), color, (int)textureId, texture1Faces[i+1][0]+1);
		}

		globalCommandList = &ad->commandList3d;
	}


	// Start fading.

	if(ad->startFade < 1.0f)
	{
		float v = ad->startFade;

		ad->startFade += ad->dt / 3.0f;

		// float p = 0.25f;

		// float r[] = {0, 0.3f, 0.9, 1};
		// int stage = 0;
		// for(int i = 0; i < arrayCount(r); i++) {
		// 	if(v >= r[i] && v < r[i+1]) { stage = i; break; }
		// }

		// float a = 0;
		// if(stage == 0) {
		// 	a = 1;
		// } else if(stage == 1) {
		// 	a = mapRange(v, r[stage], r[stage+1], 1, 0.1f);
		// 	a = powf(a, p);
		// } else if(stage == 2) {
		// 	float s = powf(0.1f, p);
		// 	a = mapRange(v, r[stage], r[stage+1], s, 0);
		// }

		float r[] = {0, 0.3f, 1};
		int stage = 0;
		for(int i = 0; i < arrayCount(r); i++) {
			if(v >= r[i] && v < r[i+1]) { stage = i; break; }
		}

		float a = 0;
		if(stage == 0) {
			a = 1;
		} else if(stage == 1) {
			a = mapRange(v, r[stage], r[stage+1], 1, 0);
		}

		Vec4 c = vec4(0,a);
		dcRect(getScreenRect(ws), c, &ad->commandList2d);
	}

	// Particle test.
	if(false)
	{
		bindShader(SHADER_CUBE);

		Vec3 ep = vec3(0,0,80);

		static ParticleEmitter emitter;
		static bool emitterInit = true; 
		if(emitterInit) {
			emitter = {};
			// emitter.particleListSize = 1024;
			// emitter.particleListSize = 100000;
			emitter.particleListSize = 100000;
			emitter.particleList = getPArray(Particle, emitter.particleListSize);
			// emitter.spawnRate = 0.0001f;
			// emitter.spawnRate = 0.001f;
			emitter.spawnRate = 0.005f;
			// emitter.spawnRate = 0.0001f;

			emitter.pos = vec3(0,0,70);
			emitter.friction = 0.5f;

			emitterInit = false;
		}

		static float dt = 0;
		// dt += ad->dt;
		emitter.pos = ep + vec3(sin(dt),0,0);
		// drawCube(emitter.pos, vec3(0.5f), vec4(0,0,0,0.2f), 0, vec3(0,0,0));

		if(0)
		{
			ParticleEmitter* e = &emitter;
			float dt = ad->dt;

			e->dt += dt;
			while(e->dt >= 0.1f) {
				e->dt -= e->spawnRate;

				if(e->particleListCount < e->particleListSize) {
					Particle p = {};

					p.pos = e->pos;
					Vec3 dir = normVec3(vec3(randomFloat(-1,1,0.01f), randomFloat(-1,1,0.01f), randomFloat(-1,1,0.01f)));
					// p.vel = dir * 1.0f;
					p.vel = dir * 5.0f;
					// p.acc = dir*0.2f;
					// p.acc = -dir*0.2f;
					p.acc = -dir*1.0f;

					// p.color = vec4(0.8f, 0.1f, 0.6f, 1.0f);
					// p.accColor = vec4(-0.15f,0,0.15f,-0.05f);
					p.color = vec4(0.8f, 0.8f, 0.1f, 1.0f);
					p.accColor = vec4(+0.10f,-0.15f,0,-0.05f);


					// p.size = vec3(0.1f);
					p.size = vec3(0.1f, 0.1f, 0.005f);

					p.rot = 0;
					p.rot2 = degreeToRadian(randomInt(0,360));
					p.velRot = 20.0f;
					p.accRot = -4.0f;

					p.timeToLive = 5;
					// p.timeToLive = randomFloat(2.0f,6.0f, 0.01f);

					e->particleList[e->particleListCount++] = p;
				}
			}
		}

			// particleEmitterUpdate(&emitter, ad->dt);

		// glDisable(GL_CULL_FACE);

		// Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
		// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs, 4);
		// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, true);

		// uint tex[2] = {getTexture(TEXTURE_CIRCLE)->id, 0};
		// glBindTextures(0,2,tex);

		// for(int i = 0; i < emitter.particleListCount; i++) {
		// 	Particle* p = emitter.particleList + i;

		// 	Vec3 normal = normVec3(p->vel);

		// 	float size = 0.1f;
		// 	Vec4 color = p->color;
		// 	Vec3 base = p->pos;

		// 	Vec3 dir1 = normVec3(cross(normal, vec3(1,0,0)));
		// 	rotateVec3(&dir1, p->rot2, normal);
		// 	rotateVec3(&normal, p->rot, dir1);
		// 	Vec3 dir2 = normVec3(cross(normal, dir1));

		// 	dir1 *= size*0.5f;
		// 	dir2 *= size*0.5f;

		// 	Vec3 verts[4] = {base + dir1+dir2, base + dir1+(-dir2), base + (-dir1)+(-dir2), base + (-dir1)+dir2};

		// 	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
		// 	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, &color);

		// 	glDrawArrays(GL_QUADS, 0, arrayCount(verts));
		// }
		// glEnable(GL_CULL_FACE);



		// glDisable(GL_CULL_FACE);

		// // Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
		// // pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs, 4);
		// // pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, true);

		// uint tex[2] = {getTexture(TEXTURE_CIRCLE)->id, 0};
		// glBindTextures(0,2,tex);

		// Mesh* mesh = getMesh(MESH_QUAD);
		// glBindBuffer(GL_ARRAY_BUFFER, mesh->bufferId);

		// glVertexAttribPointer(0, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)0);
		// glEnableVertexAttribArray(0);
		// glVertexAttribPointer(1, 2, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3)));
		// glEnableVertexAttribArray(1);
		// glVertexAttribPointer(2, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3) + sizeof(Vec2)));
		// glEnableVertexAttribArray(2);

		// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, false);

		// for(int i = 0; i < emitter.particleListCount; i++) {
		// 	Particle* p = emitter.particleList + i;

		// 	Vec3 normal = normVec3(p->vel);

		// 	float size = 0.1f;
		// 	Vec4 color = p->color;
		// 	Vec3 base = p->pos;

		// 	Vec3 dir1 = normVec3(cross(normal, vec3(1,0,0)));
		// 	rotateVec3(&dir1, p->rot2, normal);
		// 	rotateVec3(&normal, p->rot, dir1);
		// 	Vec3 dir2 = normVec3(cross(normal, dir1));

		// 	dir1 *= size*0.5f;
		// 	dir2 *= size*0.5f;

		// 	// Vec3 verts[4] = {base + dir1+dir2, base + dir1+(-dir2), base + (-dir1)+(-dir2), base + (-dir1)+dir2};

		// 	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
		// 	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, &color);

		// 	// glDrawArrays(GL_QUADS, 0, arrayCount(verts));



		// 	Mat4 model = modelMatrix(p->pos, p->size, p->rot, normal);
		// 	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODEL, model.e);
		// 	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, color.e);

		// 	glDrawArrays(GL_QUADS, 0, mesh->vertCount);

		// }
		// glEnable(GL_CULL_FACE);




		#if 1


			bindShader(SHADER_PARTICLE);
			pushUniform(SHADER_PARTICLE, 0, PARTICLE_UNIFORM_VIEW, view);
			pushUniform(SHADER_PARTICLE, 0, PARTICLE_UNIFORM_PROJ, proj);

			uint tex[2] = {getTexture(TEXTURE_CIRCLE)->id, 0};
			glBindTextures(0,1,tex);
			glBindSamplers(0,1,gs->samplers);


			static float timer = 1;
			timer += ad->dt;

			// if(timer >= 1) {

				int bufferOffset = 0;

				for(int i = 0; i < emitter.particleListCount; i++) {
					Particle* p = emitter.particleList + i;
					Vec3 normal = normVec3(p->vel);

					Mat4 model = modelMatrix(p->pos, p->size, p->rot, normal);
					// Mat4 model = modelMatrix(p->pos, vec3(p->size.x, p->size.x, p->size.x), p->rot, normal);
					rowToColumn(&model);

					memCpy(ad->testBuffer + bufferOffset, model.e, sizeof(model)); bufferOffset += sizeof(model);
					memCpy(ad->testBuffer + bufferOffset, p->color.e, sizeof(p->color)); bufferOffset += sizeof(p->color);
				}

				// printf("f%i \n", emitter.particleListCount);


				timer = 0;
				glNamedBufferData(ad->testBufferId, ad->testBufferSize, 0, GL_STREAM_DRAW);
				glNamedBufferSubData(ad->testBufferId, 0, bufferOffset, ad->testBuffer);
			// }

			// glNamedBufferData(ad->testBufferId, ad->testBufferSize, 0, GL_STREAM_DRAW);
			// glNamedBufferSubData(ad->testBufferId, 0, bufferOffset, ad->testBuffer);

			Vec3 verts[] = {vec3(-0.5f, -0.5f, 0),vec3(-0.5f, 0.5f, 0),vec3(0.5f, 0.5f, 0),vec3(0.5f, -0.5f, 0)};
			glProgramUniform3fv(getShader(SHADER_PARTICLE)->vertex, 0, 4, verts[0].e);
			
			Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
			glProgramUniform2fv(getShader(SHADER_PARTICLE)->vertex, 4, 4, quadUVs[0].e);

			glBindBuffer(GL_ARRAY_BUFFER, ad->testBufferId);

			glEnableVertexAttribArray(8);
			glVertexAttribPointer(8, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)(sizeof(Mat4)));

			glEnableVertexAttribArray(9);
			glVertexAttribPointer(9, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)0);
			glEnableVertexAttribArray(10);
			glVertexAttribPointer(10, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)(sizeof(Vec4)*1));
			glEnableVertexAttribArray(11);
			glVertexAttribPointer(11, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)(sizeof(Vec4)*2));
			glEnableVertexAttribArray(12);
			glVertexAttribPointer(12, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)(sizeof(Vec4)*3));

			glVertexAttribDivisor(8, 1);
			glVertexAttribDivisor(9, 1);
			glVertexAttribDivisor(10, 1);
			glVertexAttribDivisor(11, 1);
			glVertexAttribDivisor(12, 1);

			glDisable(GL_CULL_FACE);
			// glDrawArrays(GL_QUADS, 0, emitter.particleListCount * 4);
			glDrawArraysInstanced(GL_QUADS, 0, 4, emitter.particleListCount);
			// glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, emitter.particleListCount);
			glEnable(GL_CULL_FACE);

		#endif 
	}

	}

	endOfMainLabel:

	// Update Audio.
	{
		AudioState* as = &ad->audioState;

		as->masterVolume = 0.5f;

		int framesPerFrame = ad->dt * as->waveFormat->nSamplesPerSec;

		uint numFramesPadding;
		as->audioClient->GetCurrentPadding(&numFramesPadding);
		uint numFramesAvailable = as->bufferFrameCount - numFramesPadding;
		framesPerFrame = min(framesPerFrame, numFramesAvailable);

		if(numFramesAvailable) {

			float* buffer;
			as->renderClient->GetBuffer(numFramesAvailable, (BYTE**)&buffer);

			// Clear to zero.

			for(int i = 0; i < numFramesAvailable*2; i++) buffer[i] = 0.0f;

			for(int trackIndex = 0; trackIndex < arrayCount(as->tracks); trackIndex++) {
				Track* track = as->tracks + trackIndex;
				Audio* audio = track->audio;

				if(!track->used) continue;

				// Push all frames that are available.

				int index = track->index;
				int channels = audio->channels;
				int availableFrames = min(audio->frameCount - index, numFramesAvailable);

				for(int i = 0; i < availableFrames; i++) {
					short valueLeft = audio->data[(index*channels) + (i*channels)+0];
					short valueRight;
					if(channels > 1) valueRight = audio->data[(index*channels) + (i*channels)+1];
					else valueRight = valueLeft;

					float floatValueLeft = (float)valueLeft/SHRT_MAX;
					float floatValueRight = (float)valueRight/SHRT_MAX;

					buffer[(i*2)+0] += floatValueLeft * as->masterVolume * track->volume;
					buffer[(i*2)+1] += floatValueRight * as->masterVolume * track->volume;
				}

				track->index += framesPerFrame;

				if(track->index >= audio->frameCount) {
					track->used = false;
				}

			}

			as->renderClient->ReleaseBuffer(framesPerFrame, 0);

		}
	}

	// Render.
	{
		TIMER_BLOCK_NAMED("Render");

		bindShader(SHADER_CUBE);
		executeCommandList(&ad->commandList3d);

		bindShader(SHADER_QUAD);
		glDisable(GL_DEPTH_TEST);
		ortho(rect(0, -ws->currentRes.h, ws->currentRes.w, 0));
		blitFrameBuffers(FRAMEBUFFER_3dMsaa, FRAMEBUFFER_3dNoMsaa, ad->cur3dBufferRes, GL_COLOR_BUFFER_BIT, GL_LINEAR);


		bindFrameBuffer(FRAMEBUFFER_2d);
		glViewport(0,0, ws->currentRes.x, ws->currentRes.y);
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), vec4(1), rect(0,1,1,0), 
		         getFrameBuffer(FRAMEBUFFER_3dNoMsaa)->colorSlot[0]->id);
		// executeCommandList(&ad->commandList2d);


		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
		glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);

		bindFrameBuffer(FRAMEBUFFER_DebugMsaa);
		executeCommandList(&ad->commandList2d);

		static double tempTime = 0;
		timerInit(&ds->tempTimer);
			executeCommandList(&ds->commandListDebug, false, reload);
		tempTime += ds->dt;
		if(tempTime >= 1) {
			ds->debugRenderTime = timerStop(&ds->tempTimer);
			tempTime = 0;
		}

		// drawTextLineCulled("sdfWT34t3w4tSEr", getFont(FONT_CALIBRI, 30), vec2(200,-200), 20, vec4(1,0,1,1));

		blitFrameBuffers(FRAMEBUFFER_DebugMsaa, FRAMEBUFFER_DebugNoMsaa, ws->currentRes, GL_COLOR_BUFFER_BIT, GL_LINEAR);


		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		bindFrameBuffer(FRAMEBUFFER_2d);
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), vec4(1,1,1,ds->guiAlpha), rect(0,1,1,0), 
		         getFrameBuffer(FRAMEBUFFER_DebugNoMsaa)->colorSlot[0]->id);

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);



		#if USE_SRGB 
			glEnable(GL_FRAMEBUFFER_SRGB);
		#endif 

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), vec4(1), rect(0,1,1,0), 
		         getFrameBuffer(FRAMEBUFFER_2d)->colorSlot[0]->id);

		#if USE_SRGB
			glDisable(GL_FRAMEBUFFER_SRGB);
		#endif
	}

	// Swap window background buffer.
	{
		TIMER_BLOCK_NAMED("Swap");

		if(!ws->vsync) sd->vsyncTempTurnOff = false;

		// Sleep until monitor refresh.
		double frameTime = timerStop(&ds->swapTimer);
		int sleepTimeMS = 0;
		if(!init && ws->vsync && !sd->vsyncTempTurnOff) {
			double fullFrameTime = ((double)1/ws->frameRate);

			if(frameTime < fullFrameTime) {
				double sleepTime = fullFrameTime - frameTime;
				sleepTimeMS = sleepTime*1000.0 - 0.5f;

				if(sleepTimeMS > 0) {
	    			glFlush();
					Sleep(sleepTimeMS);
				}
			}
		}

		if(sd->vsyncTempTurnOff) {
			wglSwapIntervalEXT(0);
		}

		if(init) {
			showWindow(windowHandle);
			GLenum glError = glGetError(); printf("GLError: %i\n", glError);
		}

		// glBindFramebuffer(GL_FRAMEBUFFER, 0);
		// drawRect(rectCenDim(0,0,100,100), vec4(1,0,0,1));

		swapBuffers(sd);
		glFinish();

		timerStart(&ds->swapTimer);

		if(sd->vsyncTempTurnOff) {
			wglSwapIntervalEXT(1);
			sd->vsyncTempTurnOff = false;
		}

		// if(init) {
		// 	showWindow(windowHandle);
		// 	GLenum glError = glGetError(); printf("GLError: %i\n", glError);
		// }
	}

	debugMain(ds, appMemory, ad, reload, isRunning, init, threadQueue);

	// debugUpdatePlayback(ds, appMemory);

	// Save game.
	if(*isRunning == false)
	{
		char* saveFile = fillString("%s%s", SAVES_FOLDER, SAVE_STATE1);

		DArray<VoxelMesh>* voxels = &ad->voxelData.voxels;

		FILE* file = fopen(saveFile, "wb");
		if(file) {
			fwrite(ad->entityList.e, ad->entityList.size * sizeof(Entity), 1, file);

			fwrite(&voxels->count, sizeof(int), 1, file);

			for(int i = 0; i < voxels->count; i++) {
				fwrite(&voxels->data[i].coord, sizeof(Vec2i), 1, file);
				fwrite(voxels->data[i].voxels, VOXEL_SIZE * sizeof(uchar), 1, file);
				fwrite(voxels->data[i].lighting, VOXEL_SIZE * sizeof(uchar), 1, file);
			}

			fwrite(&startX, sizeof(int), 1, file);
			fwrite(&startY, sizeof(int), 1, file);
		}
	}

	// @AppSessionWrite
	if(*isRunning == false) {
		Rect windowRect = getWindowWindowRect(sd->windowHandle);
		if(ws->fullscreen) windowRect = ws->previousWindowRect;

		AppSessionSettings at = {};

		at.windowRect = windowRect;
		saveAppSettings(at);
	}

	// @AppEnd.
}


#if 0
void debugMain(DebugState* ds, AppMemory* appMemory, AppData* ad, bool reload, bool* isRunning, bool init, ThreadQueue* threadQueue) {
	// @DebugStart.

	globalMemory->debugMode = true;

	timerStart(&ds->tempTimer);

	Input* input = ds->input;
	WindowSettings* ws = &ad->wSettings;

	clearTMemoryDebug();

	ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[1];
	ExtendibleMemoryArray* pMemory = globalMemory->pMemory;

	int clSize = megaBytes(2);
	drawCommandListInit(&ds->commandListDebug, (char*)getTMemoryDebug(clSize), clSize);
	globalCommandList = &ds->commandListDebug;


	ds->gInput = { input->mousePos, input->mouseWheel, input->mouseButtonPressed[0], input->mouseButtonDown[0], 
					input->keysPressed[KEYCODE_ESCAPE], input->keysPressed[KEYCODE_RETURN], input->keysPressed[KEYCODE_SPACE], input->keysPressed[KEYCODE_BACKSPACE], input->keysPressed[KEYCODE_DEL], input->keysPressed[KEYCODE_HOME], input->keysPressed[KEYCODE_END], 
					input->keysPressed[KEYCODE_LEFT], input->keysPressed[KEYCODE_RIGHT], input->keysPressed[KEYCODE_UP], input->keysPressed[KEYCODE_DOWN], 
					input->keysDown[KEYCODE_SHIFT], input->keysDown[KEYCODE_CTRL], input->inputCharacters, input->inputCharacterCount};

	if(input->keysPressed[KEYCODE_F6]) ds->showMenu = !ds->showMenu;
	if(input->keysPressed[KEYCODE_F7]) ds->showStats = !ds->showStats;
	if(input->keysPressed[KEYCODE_F8]) ds->showHud = !ds->showHud;

	// Recording update.
	{
		if(ds->playbackSwapMemory) {
			threadQueueComplete(threadQueue);
			ds->playbackSwapMemory = false;

			pMemory->index = ds->snapShotCount-1;
			pMemory->arrays[pMemory->index].index = ds->snapShotMemoryIndex;

			for(int i = 0; i < ds->snapShotCount; i++) {
				memCpy(pMemory->arrays[i].data, ds->snapShotMemory[i], pMemory->slotSize);
			}
		}
	}

	if(ds->showMenu) {
		int fontSize = ds->fontHeight;

		bool initSections = false;

		Gui* gui = ds->gui;
		gui->start(ds->gInput, getFont(FONT_CALIBRI, fontSize), ws->currentRes);

		static bool sectionGuiRecording = false;
		if(gui->beginSection("Recording", &sectionGuiRecording)) {

			bool noActiveThreads = threadQueueFinished(threadQueue);

			gui->div(vec2(0,0));
			gui->label("Active Threads:");
			gui->label(fillString("%i", !noActiveThreads));

			gui->div(vec2(0,0));
			gui->label("Max Frames:");
			gui->label(fillString("%i", ds->inputCapacity));

			gui->div(vec2(0,0));
			if(gui->switcher("Record", &ds->recordingInput)) {
				if(ds->playbackInput || !noActiveThreads) ds->recordingInput = false;

				if(ds->recordingInput) {
					if(threadQueueFinished(threadQueue)) {

						ds->snapShotCount = pMemory->index+1;
						ds->snapShotMemoryIndex = pMemory->arrays[pMemory->index].index;
						for(int i = 0; i < ds->snapShotCount; i++) {
							if(ds->snapShotMemory[i] == 0) 
								ds->snapShotMemory[i] = (char*)malloc(pMemory->slotSize);

							memCpy(ds->snapShotMemory[i], pMemory->arrays[i].data, pMemory->slotSize);
						}


						ds->recordingInput = true;
						ds->inputIndex = 0;
					}
				}
			}
			gui->label(fillString("%i", ds->inputIndex));


			if(ds->inputIndex > 0 && !ds->recordingInput) {
				char* s = ds->playbackInput ? "Stop Playback" : "Start Playback";

				if(gui->switcher(s, &ds->playbackInput)) {
					if(ds->playbackInput) {
						threadQueueComplete(threadQueue);
						ds->playbackIndex = 0;

						pMemory->index = ds->snapShotCount-1;
						pMemory->arrays[pMemory->index].index = ds->snapShotMemoryIndex;

						for(int i = 0; i < ds->snapShotCount; i++) {
							memCpy(pMemory->arrays[i].data, ds->snapShotMemory[i], pMemory->slotSize);
						}
					} else {
						ds->playbackPause = false;
						ds->playbackBreak = false;
					}
				}

				if(ds->playbackInput) {
					gui->div(vec2(0,0));

					gui->switcher("Pause/Resume", &ds->playbackPause);

					int cap = ds->playbackIndex;
					gui->slider(&ds->playbackIndex, 0, ds->inputIndex - 1);
					ds->playbackIndex = cap;

					gui->div(vec3(0.25f,0.25f,0));
					if(gui->button("Step")) {
						ds->playbackBreak = true;
						ds->playbackPause = false;
						ds->playbackBreakIndex = (ds->playbackIndex + 1)%ds->inputIndex;
					}
					gui->switcher("Break", &ds->playbackBreak);
					gui->slider(&ds->playbackBreakIndex, 0, ds->inputIndex - 1);
				}
			}

		} gui->endSection();

		static bool sectionGuiSettings = initSections;
		if(gui->beginSection("GuiSettings", &sectionGuiSettings)) {
			guiSettings(gui);
		} gui->endSection();

		static bool sectionSettings = initSections;
		if(gui->beginSection("Settings", &sectionSettings)) {
			gui->div(vec2(0,0)); if(gui->button("Compile")) shellExecute("C:\\Projects\\Hmm\\code\\buildWin32.bat");
								 if(gui->button("Up Buffers")) ad->updateFrameBuffers = true;
			gui->div(vec2(0,0)); gui->label("FoV", 0); gui->slider(&ad->fieldOfView, 1, 180);
			gui->div(vec2(0,0)); gui->label("MSAA", 0); gui->slider(&ad->msaaSamples, 1, 8);
			gui->switcher("Native Res", &ad->useNativeRes);
			gui->div(0,0,0); gui->label("FboRes", 0); gui->slider(&ad->fboRes.x, 150, ad->cur3dBufferRes.x); gui->slider(&ad->fboRes.y, 150, ad->cur3dBufferRes.y);
			gui->div(0,0,0); gui->label("NFPlane", 0); gui->slider(&ad->nearPlane, 0.01, 2); gui->slider(&ad->farPlane, 1000, 5000);
		} gui->endSection();

		static bool sectionEntities = true;
		if(gui->beginSection("Entities", &sectionEntities)) { 


			// EntityList* list = &ad->entityList;
			// for(int i = 0; i < list->size; i++) {
			// 	Entity* e = list->e + i;

			// 	if(e->init) {
			// 		gui->label(fillString("Id: %i", e->id), 0);
			// 		gui->startPos.x += 30;
			// 		gui->panelWidth -= 30;

			// 		float tw = 150;

			// 		gui->div(vec2(tw,0)); gui->label("init", 0);           gui->textBoxInt(&e->init);
			// 		gui->div(vec2(tw,0)); gui->label("type", 0);           gui->textBoxInt(&e->type);
			// 		gui->div(vec2(tw,0)); gui->label("id", 0);             gui->textBoxInt(&e->id);
			// 		gui->div(vec2(tw,0)); gui->label("name", 0);           gui->textBoxChar(e->name);

			// 		#define TEDIT_VEC3(name) gui->textBoxFloat(&name.x); gui->textBoxFloat(&name.y); gui->textBoxFloat(&name.z);

			// 		gui->div(tw,0,0,0); gui->label("pos", 0);            TEDIT_VEC3(e->pos);
			// 		gui->div(tw,0,0,0); gui->label("dir", 0);            TEDIT_VEC3(e->dir);
			// 		gui->div(tw,0,0,0); gui->label("rot", 0);            TEDIT_VEC3(e->rot);
			// 		gui->div(vec2(tw,0)); gui->label("rotAngle", 0);     gui->textBoxFloat(&e->rotAngle);
			// 		gui->div(tw,0,0,0); gui->label("dim", 0);            TEDIT_VEC3(e->dim);
			// 		gui->div(tw,0,0,0); gui->label("camOff", 0);         TEDIT_VEC3(e->camOff);
			// 		gui->div(tw,0,0,0); gui->label("vel", 0);            TEDIT_VEC3(e->vel);
			// 		gui->div(tw,0,0,0); gui->label("acc", 0);            TEDIT_VEC3(e->acc);

			// 		gui->div(vec2(tw,0)); gui->label("movementType", 0);   gui->textBoxInt(&e->movementType);
			// 		gui->div(vec2(tw,0)); gui->label("spatial", 0);        gui->textBoxInt(&e->spatial);

			// 		gui->div(vec2(tw,0)); gui->label("deleted", 0);        gui->switcher("deleted", &e->deleted);
			// 		gui->div(vec2(tw,0)); gui->label("isMoving", 0);       gui->switcher("isMoving", &e->isMoving);
			// 		gui->div(vec2(tw,0)); gui->label("isColliding", 0);    gui->switcher("isColliding", &e->isColliding);
			// 		gui->div(vec2(tw,0)); gui->label("exploded", 0);       gui->switcher("exploded", &e->exploded);
			// 		gui->div(vec2(tw,0)); gui->label("playerOnGround", 0); gui->switcher("playerOnGround", &e->playerOnGround);

			// 		gui->startPos.x -= 30;
			// 		gui->panelWidth += 30;
			// 	}
			// }

			// entityStructMemberInfos[];



			EntityList* list = &ad->entityList;
			for(int i = 0; i < list->size; i++) {
				Entity* e = list->e + i;

				if(e->init) {
					guiPrintIntrospection(gui, STRUCTTYPE_ENTITY, (char*)e);
				}
			}


		} gui->endSection();

		addDebugInfo(fillString("%i", ad->entityList.size));

		static bool sectionWorld = initSections;
		if(gui->beginSection("World", &sectionWorld)) { 
			if(gui->button("Reload World") || input->keysPressed[KEYCODE_TAB]) ad->reloadWorld = true;
			
			gui->div(vec2(0,0)); gui->label("RefAlpha", 0); gui->slider(&reflectionAlpha, 0, 1);
			gui->div(vec2(0,0)); gui->label("Light", 0); gui->slider(&globalLumen, 0, 255);
			gui->div(0,0,0); gui->label("MinMax", 0); gui->slider(&WORLD_MIN, 0, 255); gui->slider(&WORLD_MAX, 0, 255);
			gui->div(vec2(0,0)); gui->label("WaterLevel", 0); 
				if(gui->slider(&waterLevelValue, 0, 0.2f)) WATER_LEVEL_HEIGHT = lerp(waterLevelValue, WORLD_MIN, WORLD_MAX);

			gui->div(vec2(0,0)); gui->label("WFreq", 0); gui->slider(&worldFreq, 0.0001f, 0.02f);
			gui->div(vec2(0,0)); gui->label("WDepth", 0); gui->slider(&worldDepth, 1, 10);
			gui->div(vec2(0,0)); gui->label("MFreq", 0); gui->slider(&modFreq, 0.001f, 0.1f);
			gui->div(vec2(0,0)); gui->label("MDepth", 0); gui->slider(&modDepth, 1, 10);
			gui->div(vec2(0,0)); gui->label("MOffset", 0); gui->slider(&modOffset, 0, 1);
			gui->div(vec2(0,0)); gui->label("PowCurve", 0); gui->slider(&worldPowCurve, 1, 6);
			gui->div(0,0,0,0); gui->slider(heightLevels+0,0,1); gui->slider(heightLevels+1,0,1);
								  gui->slider(heightLevels+2,0,1); gui->slider(heightLevels+3,0,1);
		} gui->endSection();

		static bool sectionTest = false;
		if(gui->beginSection("Test", &sectionTest)) {
			static int scrollHeight = 200;
			static int scrollElements = 13;
			static float scrollVal = 0;
			gui->div(vec2(0,0)); gui->slider(&scrollHeight, 0, 2000); gui->slider(&scrollElements, 0, 100);

			gui->beginScroll(scrollHeight, &scrollVal); {
				for(int i = 0; i < scrollElements; i++) {
					gui->button(fillString("Element: %i.", i));
				}
			} gui->endScroll();

			static int textCapacity = 50;
			static char* text = getPArray(char, textCapacity);
			static bool DOIT = true;
			if(DOIT) {
				DOIT = false;
				strClear(text);
				strCpy(text, "This is a really long sentence!");
			}
			gui->div(vec2(0,0)); gui->label("Text Box:", 0); gui->textBoxChar(text, 0, textCapacity);

			static int textNumber = 1234;
			gui->div(vec2(0,0)); gui->label("Int Box:", 0); gui->textBoxInt(&textNumber);

			static float textFloat = 123.456f;
			gui->div(vec2(0,0)); gui->label("Float Box:", 0); gui->textBoxFloat(&textFloat);

		} gui->endSection();

		gui->end();
	}

	ds->timer->timerInfoCount = __COUNTER__;

	int fontHeight = 18;
	Timer* timer = ds->timer;
	int cycleCount = arrayCount(ds->timings);

	bool threadsFinished = threadQueueFinished(threadQueue);

	int bufferIndex = timer->bufferIndex;

	// Save const strings from initialised timerinfos.
	{
		int timerCount = timer->timerInfoCount;
		for(int i = 0; i < timerCount; i++) {
			TimerInfo* info = timer->timerInfos + i;

			// Set colors.
			float ss = i%(timerCount/2) / ((float)timerCount/2);
			float h = i < timerCount/2 ? 0.1f : -0.1f;
			Vec3 color = vec3(0,0,0);
			hslToRgb(color.e, 360*ss, 0.5f, 0.5f+h);

			vSet3(info->color, color.r, color.g, color.b);


			if(!info->initialised || info->stringsSaved) continue;
			char* s;
			
			s = info->file;
			info->file = getPStringDebug(strLen(s) + 1);
			strCpy(info->file, s);

			s = info->function;
			info->function = getPStringDebug(strLen(s) + 1);
			strCpy(info->function, s);

			s = info->name;
			info->name = getPStringDebug(strLen(s) + 1);
			strCpy(info->name, s);

			info->stringsSaved = true;
		}
	}

	if(ds->setPause) {
		ds->lastCycleIndex = ds->cycleIndex;
		ds->cycleIndex = mod(ds->cycleIndex-1, arrayCount(ds->timings));

		ds->timelineCamSize = -1;
		ds->timelineCamPos = -1;

		ds->setPause = false;
	}
	if(ds->setPlay) {
		ds->cycleIndex = ds->lastCycleIndex;
		ds->setPlay = false;
	}

	Timings* timings = ds->timings[ds->cycleIndex];
	Statistic* statistics = ds->statistics[ds->cycleIndex];

	int cycleIndex = ds->cycleIndex;
	int newCycleIndex = (ds->cycleIndex + 1)%cycleCount;

	// Timer update.
	{

		if(!ds->noCollating) {
			zeroMemory(timings, timer->timerInfoCount*sizeof(Timings));
			zeroMemory(statistics, timer->timerInfoCount*sizeof(Statistic));

			ds->cycleIndex = newCycleIndex;

			// Collate timing buffer.

			// for(int threadIndex = 0; threadIndex < threadQueue->threadCount; threadIndex++) 
			{
				// GraphSlot* graphSlots = ds->graphSlots[threadIndex];
				// int index = ds->graphSlotCount[threadIndex];

				for(int i = ds->lastBufferIndex; i < bufferIndex; ++i) {
					TimerSlot* slot = timer->timerBuffer + i;
					
					int threadIndex = threadIdToIndex(threadQueue, slot->threadId);

					if(slot->type == TIMER_TYPE_BEGIN) {
						int index = ds->graphSlotCount[threadIndex];

						GraphSlot graphSlot;
						graphSlot.threadIndex = threadIndex;
						graphSlot.timerIndex = slot->timerIndex;
						graphSlot.stackIndex = index;
						graphSlot.cycles = slot->cycles;
						ds->graphSlots[threadIndex][index] = graphSlot;

						ds->graphSlotCount[threadIndex]++;
					} else {
						ds->graphSlotCount[threadIndex]--;
						int index = ds->graphSlotCount[threadIndex];
						if(index < 0) index = 0; // @Hack, to keep things running.

						ds->graphSlots[threadIndex][index].size = slot->cycles - ds->graphSlots[threadIndex][index].cycles;
						ds->savedBuffer[ds->savedBufferIndex] = ds->graphSlots[threadIndex][index];
						ds->savedBufferIndex = (ds->savedBufferIndex+1)%ds->savedBufferMax;
						ds->savedBufferCount = clampMax(ds->savedBufferCount + 1, ds->savedBufferMax);


						Timings* timing = timings + ds->graphSlots[threadIndex][index].timerIndex;
						timing->cycles += ds->graphSlots[threadIndex][index].size;
						timing->hits++;
					}
				}

				// ds->graphSlotCount[threadIndex] = index;
			}

			// ds->savedBufferCounts[cycleIndex] = savedBufferCount;

			for(int i = 0; i < timer->timerInfoCount; i++) {
				Timings* t = timings + i;
				t->cyclesOverHits = t->hits > 0 ? (u64)(t->cycles/t->hits) : 0; 
			}

			for(int timerIndex = 0; timerIndex < timer->timerInfoCount; timerIndex++) {
				Statistic* stat = statistics + timerIndex;
				beginStatistic(stat);

				for(int i = 0; i < arrayCount(ds->timings); i++) {
					Timings* t = &ds->timings[i][timerIndex];
					if(t->hits == 0) continue;

					updateStatistic(stat, t->cyclesOverHits);
				}

				endStatistic(stat);
				if(stat->count == 0) stat->avg = 0;
			}
		}
	}

	ds->lastBufferIndex = bufferIndex;

	if(threadsFinished) {
		timer->bufferIndex = 0;
		ds->lastBufferIndex = 0;
	}

	assert(timer->bufferIndex < timer->bufferSize);

	if(init) {
		ds->lineGraphCamSize = 700000;
		ds->lineGraphCamPos = 0;
		ds->mode = 0;
		ds->lineGraphHeight = 30;
		ds->lineGraphHighlight = 0;
	}

	//
	// Draw timing info.
	//

	if(ds->showStats) 
	{
		static int highlightedIndex = -1;
		Vec4 highlightColor = vec4(1,1,1,0.05f);

		// float cyclesPerFrame = (float)((3.5f*((float)1/60))*1024*1024*1024);
		float cyclesPerFrame = (float)((3.5f*((float)1/60))*1000*1000*1000);
		int fontSize = ds->fontHeight;
		Vec2 textPos = vec2(550, -fontHeight);
		int infoCount = timer->timerInfoCount;

		Gui* gui = ds->gui2;
		gui->start(ds->gInput, getFont(FONT_CALIBRI, fontHeight), ws->currentRes);

		gui->label("App Statistics", 1, gui->colors.sectionColor, vec4(0,0,0,1));

		float sectionWidth = 120;
		float headerDivs[] = {sectionWidth,sectionWidth,sectionWidth,0,80,80};
		gui->div(headerDivs, arrayCount(headerDivs));
		if(gui->button("Data", (int)(ds->mode == 0) + 1)) ds->mode = 0;
		if(gui->button("Line graph", (int)(ds->mode == 1) + 1)) ds->mode = 1;
		if(gui->button("Timeline", (int)(ds->mode == 2) + 1)) ds->mode = 2;
		gui->empty();
		gui->label(fillString("%fms", ds->debugTime*1000), 1);
		gui->label(fillString("%fms", ds->debugRenderTime*1000), 1);

		gui->div(vec2(0.2f,0));
		if(gui->switcher("Freeze", &ds->noCollating)) {
			if(ds->noCollating) ds->setPause = true;
			else ds->setPlay = true;
		}
		gui->slider(&ds->cycleIndex, 0, cycleCount-1);

		if(ds->mode == 0)
		{
			int barWidth = 1;
			int barCount = arrayCount(ds->timings);
			float sectionWidths[] = {0,0.2f,0,0,0,0,0,0, barWidth*barCount};
			// float sectionWidths[] = {0.1f,0,0.1f,0,0.05f,0,0,0.1f, barWidth*barCount};

			char* headers[] = {"File", "Function", "Description", "Cycles", "Hits", "C/H", "Avg. Cycl.", "Total Time", "Graphs"};
			gui->div(sectionWidths, arrayCount(sectionWidths));

			float textSectionEnd;
			for(int i = 0; i < arrayCount(sectionWidths); i++) {
				// @Hack: Get the end of the text region by looking at last region.
				if(i == arrayCount(sectionWidths)-1) textSectionEnd = gui->getCurrentRegion().max.x;

				Vec4 buttonColor = vec4(gui->colors.regionColor.rgb, 0.5f);
				if(gui->button(headers[i], 0, 1, buttonColor, vec4(0,0,0,1))) {
					if(abs(ds->graphSortingIndex) == i) ds->graphSortingIndex *= -1;
					else ds->graphSortingIndex = i;
				}
			}

			SortPair* sortList = getTArrayDebug(SortPair, infoCount+1);
			{
				for(int i = 0; i < infoCount+1; i++) sortList[i].index = i;

		   			 if(abs(ds->graphSortingIndex) == 3) for(int i = 0; i < infoCount+1; i++) sortList[i].key = timings[i].cycles;
		   		else if(abs(ds->graphSortingIndex) == 4) for(int i = 0; i < infoCount+1; i++) sortList[i].key = timings[i].hits;
		   		else if(abs(ds->graphSortingIndex) == 5) for(int i = 0; i < infoCount+1; i++) sortList[i].key = timings[i].cyclesOverHits;
		   		else if(abs(ds->graphSortingIndex) == 6) for(int i = 0; i < infoCount+1; i++) sortList[i].key = statistics[i].avg;
		   		else if(abs(ds->graphSortingIndex) == 7) for(int i = 0; i < infoCount+1; i++) sortList[i].key = timings[i].cycles/cyclesPerFrame;

		   		bool sortDirection = true;
		   		if(ds->graphSortingIndex < 0) sortDirection = false;

		   		if(valueBetween(abs(ds->graphSortingIndex), 3, 7)) 
					bubbleSort(sortList, infoCount, sortDirection);
			}

			for(int index = 0; index < infoCount; index++) {
				int i = sortList[index].index;

				TimerInfo* tInfo = timer->timerInfos + i;
				Timings* timing = timings + i;

				if(!tInfo->initialised) continue;

				gui->div(sectionWidths, arrayCount(sectionWidths)); 

				// if(highlightedIndex == i) {
				// 	Rect r = gui->getCurrentRegion();
				// 	Rect line = rect(r.min, vec2(textSectionEnd,r.min.y + fontHeight));
				// 	dcRect(line, highlightColor);
				// }

				gui->label(fillString("%s", tInfo->file + 21),0);
				if(gui->button(fillString("%s", tInfo->function),0, 0, vec4(gui->colors.regionColor.rgb, 0.2f))) {
					char* command = fillString("%s %s:%i", editor_executable_path, tInfo->file, tInfo->line);
					shellExecuteNoWindow(command);
				}
				gui->label(fillString("%s", tInfo->name),0);
				gui->label(fillString("%i64.c", timing->cycles),2);
				gui->label(fillString("%i64.", timing->hits),2);
				gui->label(fillString("%i64.c", timing->cyclesOverHits),2);
				gui->label(fillString("%i64.c", (i64)statistics[i].avg),2); // Not a i64 but whatever.
				gui->label(fillString("%.3f%%", ((float)timing->cycles/cyclesPerFrame)*100),2);

				// Bar graphs.
				dcState(STATE_LINEWIDTH, barWidth);

				gui->empty();
				Rect r = gui->getCurrentRegion();
				float rheight = gui->getDefaultHeight();
				float fontBaseOffset = 4;

				float xOffset = 0;
				for(int statIndex = 0; statIndex < barCount; statIndex++) {
					Statistic* stat = statistics + i;
					u64 coh = ds->timings[statIndex][i].cyclesOverHits;

					float height = mapRangeClamp(coh, stat->min, stat->max, 1, rheight);
					Vec2 rmin = r.min + vec2(xOffset, fontBaseOffset);
					float colorOffset = mapRange(coh, stat->min, stat->max, 0, 1);
					// dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,1-colorOffset,0,1));
					dcLine2d(rmin, rmin+vec2(0,height), vec4(colorOffset,1-colorOffset,0,1));

					xOffset += barWidth;
				}
			}
		}

		// Timeline graph.
		if(ds->mode == 2 && ds->noCollating)
		{
			float lineHeightOffset = 1.2;

			gui->empty();
			Rect cyclesRect = gui->getCurrentRegion();
			gui->heightPush(1.5f);
			gui->empty();
			Rect headerRect = gui->getCurrentRegion();
			gui->heightPop();

			float lineHeight = fontHeight * lineHeightOffset;

			gui->heightPush(3*lineHeight +  2*lineHeight*(threadQueue->threadCount-1));
			gui->empty();
			Rect bgRect = gui->getCurrentRegion();
			gui->heightPop();

			float graphWidth = rectDim(bgRect).w;

			int swapTimerIndex = 0;
			for(int i = 0; i < timer->timerInfoCount; i++) {
				if(!timer->timerInfos[i].initialised) continue;

				if(strCompare(timer->timerInfos[i].name, "Swap")) {
					swapTimerIndex = i;
					break;
				}
			}

			int recentIndex = mod(ds->savedBufferIndex-1, ds->savedBufferMax);
			int oldIndex = mod(ds->savedBufferIndex - ds->savedBufferCount, ds->savedBufferMax);
			GraphSlot recentSlot = ds->savedBuffer[recentIndex];
			GraphSlot oldSlot = ds->savedBuffer[oldIndex];
			double cyclesLeft = oldSlot.cycles;
			double cyclesRight = recentSlot.cycles + recentSlot.size;
			double cyclesSize = cyclesRight - cyclesLeft;

			// Setup cam pos and zoom.
			if(ds->timelineCamPos == -1 && ds->timelineCamSize == -1) {
				ds->timelineCamSize = (recentSlot.cycles + recentSlot.size) - oldSlot.cycles;
				ds->timelineCamPos = oldSlot.cycles + ds->timelineCamSize/2;
			}

			if(gui->input.mouseWheel) {
				float wheel = gui->input.mouseWheel;

				float offset = wheel < 0 ? 1.1f : 1/1.1f;
				if(!input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.2f : 1/1.2f;
				if(input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.4f : 1/1.4f;

				double oldZoom = ds->timelineCamSize;
				ds->timelineCamSize *= offset;
				clampDouble(&ds->timelineCamSize, 1000, cyclesSize);
				double diff = ds->timelineCamSize - oldZoom;

				float zoomOffset = mapRange(input->mousePos.x, bgRect.min.x, bgRect.max.x, -0.5f, 0.5f);
				ds->timelineCamPos -= diff*zoomOffset;
			}


			Vec2 dragDelta = vec2(0,0);
			gui->drag(bgRect, &dragDelta, vec4(0,0,0,0));

			ds->timelineCamPos -= dragDelta.x * (ds->timelineCamSize/graphWidth);
			clampDouble(&ds->timelineCamPos, cyclesLeft + ds->timelineCamSize/2, cyclesRight - ds->timelineCamSize/2);


			double camPos = ds->timelineCamPos;
			double zoom = ds->timelineCamSize;
			double orthoLeft = camPos - zoom/2;
			double orthoRight = camPos + zoom/2;


			// Header.
			{
				dcRect(cyclesRect, gui->colors.sectionColor);
				Vec2 cyclesDim = rectDim(cyclesRect);

				dcRect(headerRect, vec4(1,1,1,0.1f));
				Vec2 headerDim = rectDim(headerRect);

				{
					float viewAreaLeft = mapRangeDouble(orthoLeft, cyclesLeft, cyclesRight, cyclesRect.min.x, cyclesRect.max.x);
					float viewAreaRight = mapRangeDouble(orthoRight, cyclesLeft, cyclesRight, cyclesRect.min.x, cyclesRect.max.x);

					float viewSize = viewAreaRight - viewAreaLeft;
					float viewMid = viewAreaRight + viewSize/2;
					float viewMinSize = 2;
					if(viewSize < viewMinSize) {
						viewAreaLeft = viewMid - viewMinSize*0.5;
						viewAreaRight = viewMid + viewMinSize*0.5;
					}

					dcRect(rect(viewAreaLeft, cyclesRect.min.y, viewAreaRight, cyclesRect.max.y), vec4(1,1,1,0.03f));
				}

				float g = 0.7f;
				float heightMod = 0.0f;
				double div = 4;
				double divMod = (1/div) + 0.05f;

				double timelineSection = div;
				while(timelineSection < zoom*divMod*(ws->currentRes.h/(graphWidth))) {
					timelineSection *= div;
					heightMod += 0.1f;
				}

				clampMax(&heightMod, 1);

				dcState(STATE_LINEWIDTH, 3);
				double startPos = roundModDouble(orthoLeft, timelineSection) - timelineSection;
				double pos = startPos;
				while(pos < orthoRight + timelineSection) {
					double p = mapRangeDouble(pos, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);

					// Big line.
					{
						float h = headerDim.h*heightMod;
						dcLine2d(vec2(p,headerRect.min.y), vec2(p,headerRect.min.y + h), vec4(g,g,g,1));
					}

					// Text
					{
						Vec2 textPos = vec2(p,cyclesRect.min.y + cyclesDim.h/2);
						float percent = mapRange(pos, cyclesLeft, cyclesRight, 0, 100);
						int percentInterval = mapRangeDouble(timelineSection, 0, cyclesSize, 0, 100);

						char* s;
						if(percentInterval > 10) s = fillString("%i%%", (int)percent);
						else if(percentInterval > 1) s = fillString("%.1f%%", percent);
						else if(percentInterval > 0.1) s = fillString("%.2f%%", percent);
						else s = fillString("%.3f%%", percent);

						float tw = getTextDim(s, gui->font).w;
						if(valueBetween(bgRect.min.x, textPos.x - tw/2, textPos.x + tw/2)) textPos.x = bgRect.min.x + tw/2;
						if(valueBetween(bgRect.max.x, textPos.x - tw/2, textPos.x + tw/2)) textPos.x = bgRect.max.x - tw/2;

						dcText(s, gui->font, textPos, gui->colors.textColor, vec2i(0,0), 0, 1, gui->colors.shadowColor);
					}

					pos += timelineSection;
				}
				dcState(STATE_LINEWIDTH, 1);

				pos = startPos;
				timelineSection /= div;
				heightMod *= 0.6f;
				int index = 0;
				while(pos < orthoRight + timelineSection) {

					// Small line.
					if((index%(int)div) != 0) {
						double p = mapRangeDouble(pos, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);
						float h = headerDim.h*heightMod;
						dcLine2d(vec2(p,headerRect.min.y), vec2(p,headerRect.min.y + h), vec4(g,g,g,1));
					}

					// Cycle text.
					{
						float pMid = mapRangeDouble(pos - timelineSection/2, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);
						Vec2 textPos = vec2(pMid,headerRect.min.y + headerDim.h/3);

						double cycles = timelineSection;
						char* s;
						if(cycles < 1000) s = fillString("%ic", (int)cycles);
						else if(cycles < 1000000) s = fillString("%.1fkc", cycles/1000);
						else if(cycles < 1000000000) s = fillString("%.1fmc", cycles/1000000);
						else if(cycles < 1000000000000) s = fillString("%.1fbc", cycles/1000000000);
						else s = fillString("INF");

						dcText(s, gui->font, textPos, gui->colors.textColor, vec2i(0,0), 0, gui->settings.textShadow, gui->colors.shadowColor);
					}

					pos += timelineSection;
					index++;

				}
			}

			dcState(STATE_LINEWIDTH, 1);

			bool mouseHighlight = false;
			Rect hRect;
			Vec4 hc;
			char* hText;
			GraphSlot* hSlot;

			Vec2 startPos = rectTL(bgRect);
			startPos -= vec2(0, lineHeight);

			int firstBufferIndex = oldIndex;
			int bufferCount = ds->savedBufferCount;
			for(int threadIndex = 0; threadIndex < threadQueue->threadCount; threadIndex++) {

				// Horizontal lines to distinguish thread bars.
				if(threadIndex > 0) {
					Vec2 p = startPos + vec2(0,lineHeight);
					float g = 0.8f;
					dcLine2d(p, vec2(bgRect.max.x, p.y), vec4(g,g,g,1));
				}

				for(int i = 0; i < bufferCount; ++i) {
					GraphSlot* slot = ds->savedBuffer + ((firstBufferIndex+i)%ds->savedBufferMax);
					if(slot->threadIndex != threadIndex) continue;

					Timings* t = timings + slot->timerIndex;
					TimerInfo* tInfo = timer->timerInfos + slot->timerIndex;

					if(slot->cycles + slot->size < orthoLeft || slot->cycles > orthoRight) continue;


					double barLeft = mapRangeDouble(slot->cycles, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);
					double barRight = mapRangeDouble(slot->cycles + slot->size, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);

					// Draw vertical line at swap boundaries.
					if(slot->timerIndex == swapTimerIndex) {
						float g = 0.8f;
						dcLine2d(vec2(barRight, bgRect.min.y), vec2(barRight, bgRect.max.y), vec4(g,g,g,1));
					}

					// Bar min size is 1.
					if(barRight - barLeft < 1) {
						double mid = barLeft + (barRight - barLeft)/2;
						barLeft = mid - 0.5f;
						barRight = mid + 0.5f;
					}

					float y = startPos.y+slot->stackIndex*-lineHeight;
					Rect r = rect(vec2(barLeft,y), vec2(barRight, y + lineHeight));

					float cOff = slot->timerIndex/(float)timer->timerInfoCount;
					Vec4 c = vec4(tInfo->color[0], tInfo->color[1], tInfo->color[2], 1);

					if(gui->getMouseOver(gui->input.mousePos, r)) {
						mouseHighlight = true;
						hRect = r;
						hc = c;

						hText = fillString("%s %s (%i.c)", tInfo->function, tInfo->name, slot->size);
						hSlot = slot;
					} else {
						float g = 0.1f;
						gui->drawRect(r, vec4(g,g,g,1));

						bool textRectVisible = (barRight - barLeft) > 1;
						if(textRectVisible) {
							if(barLeft < bgRect.min.x) r.min.x = bgRect.min.x;
							Rect textRect = rect(r.min+vec2(1,1), r.max-vec2(1,1));

							gui->drawTextBox(textRect, fillString("%s %s (%i.c)", tInfo->function, tInfo->name, slot->size), c, 0, rectDim(textRect).w);
						}
					}

				}

				if(threadIndex == 0) startPos.y -= lineHeight*3;
				else startPos.y -= lineHeight*2;

			}

			if(mouseHighlight) {
				if(hRect.min.x < bgRect.min.x) hRect.min.x = bgRect.min.x;

				float tw = getTextDim(hText, gui->font).w + 2;
				if(tw > rectDim(hRect).w) hRect.max.x = hRect.min.x + tw;

				float g = 0.8f;
				gui->drawRect(hRect, vec4(g,g,g,1));

				Rect textRect = rect(hRect.min+vec2(1,1), hRect.max-vec2(1,1));
				gui->drawTextBox(textRect, hText, hc);
			}

			gui->div(0.1f, 0); 
			gui->div(0.1f, 0); 

			if(gui->button("Reset")) {
				ds->timelineCamSize = (recentSlot.cycles + recentSlot.size) - oldSlot.cycles;
				ds->timelineCamPos = oldSlot.cycles + ds->timelineCamSize/2;
			}

			gui->label(fillString("Cam: %i64., Zoom: %i64.", (i64)ds->timelineCamPos, (i64)ds->timelineCamSize));
		}
		



		// Line graph.
		if(ds->mode == 1)
		{
			dcState(STATE_LINEWIDTH, 1);

			// Get longest function name string.
			float timerInfoMaxStringSize = 0;
			int cycleCount = arrayCount(ds->timings);
			int timerCount = ds->timer->timerInfoCount;
			for(int timerIndex = 0; timerIndex < timerCount; timerIndex++) {
				TimerInfo* info = &timer->timerInfos[timerIndex];
				if(!info->initialised) continue;

				Statistic* stat = &ds->statistics[cycleIndex][timerIndex];
				if(stat->avg == 0) continue;

				char* text = strLen(info->name) > 0 ? info->name : info->function;
				timerInfoMaxStringSize = max(getTextDim(text, gui->font).w, timerInfoMaxStringSize);
			}

			// gui->div(0.2f, 0);
			gui->slider(&ds->lineGraphHeight, 1, 60);
			// gui->empty();

			gui->heightPush(gui->getDefaultHeight() * ds->lineGraphHeight);
			gui->div(vec3(timerInfoMaxStringSize + 2, 0, 120));
			gui->empty(); Rect rectNames = gui->getCurrentRegion();
			gui->empty(); Rect rectLines = gui->getCurrentRegion();
			gui->empty(); Rect rectNumbers = gui->getCurrentRegion();
			gui->heightPop();

			float rTop = rectLines.max.y;
			float rBottom = rectLines.min.y;

			Vec2 dragDelta = vec2(0,0);
			gui->drag(rectLines, &dragDelta, vec4(0,0,0,0.2f));

			float wheel = gui->input.mouseWheel;
			if(wheel) {
				float offset = wheel < 0 ? 1.1f : 1/1.1f;
				if(!input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.2f : 1/1.2f;
				if(input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.4f : 1/1.4f;

				float heightDiff = ds->lineGraphCamSize;
				ds->lineGraphCamSize *= offset;
				ds->lineGraphCamSize = clampMin(ds->lineGraphCamSize, 0.00001f);
				heightDiff -= ds->lineGraphCamSize;

				float mouseOffset = mapRange(input->mousePosNegative.y, rBottom, rTop, -0.5f, 0.5f);
				ds->lineGraphCamPos += heightDiff * mouseOffset;
			}

			ds->lineGraphCamPos -= dragDelta.y * ((ds->lineGraphCamSize)/(rTop - rBottom));
			clampMin(&ds->lineGraphCamPos, ds->lineGraphCamSize/2.05f);

			float orthoTop = ds->lineGraphCamPos + ds->lineGraphCamSize/2;
			float orthoBottom = ds->lineGraphCamPos - ds->lineGraphCamSize/2;

			// Draw numbers.
			{
				gui->scissorPush(rectNumbers);

				float y = 0;
				float length = 10;

				float div = 10;
				float timelineSection = div;
				float splitMod = (1/div)*0.2f;
				while(timelineSection < ds->lineGraphCamSize*splitMod*(ws->currentRes.h/(rTop-rBottom))) timelineSection *= div;

				float start = roundMod(orthoBottom, timelineSection) - timelineSection;

				float p = start;
				while(p < orthoTop) {
					p += timelineSection;
					y = mapRange(p, orthoBottom, orthoTop, rBottom, rTop);

					dcLine2d(vec2(rectNumbers.min.x, y), vec2(rectNumbers.min.x + length, y), vec4(1,1,1,1)); 
					dcText(fillString("%i64.c",(i64)p), gui->font, vec2(rectNumbers.min.x + length + 4, y), vec4(1,1,1,1), vec2i(-1,0));
				}

				gui->scissorPop();
			}

			for(int timerIndex = 0; timerIndex < timerCount; timerIndex++) {
				TimerInfo* info = &timer->timerInfos[timerIndex];
				if(!info->initialised) continue;

				Statistic* stat = &ds->statistics[cycleIndex][timerIndex];
				if(stat->avg == 0) continue;

				float statMin = mapRange(stat->min, orthoBottom, orthoTop, rBottom, rTop);
				float statMax = mapRange(stat->max, orthoBottom, orthoTop, rBottom, rTop);
				if(statMax < rBottom || statMin > rTop) continue;

				Vec4 color = vec4(info->color[0], info->color[1], info->color[2], 1);

				float yAvg = mapRange(stat->avg, orthoBottom, orthoTop, rBottom, rTop);
				char* text = strLen(info->name) > 0 ? info->name : info->function;
				float textWidth = getTextDim(text, gui->font, vec2(rectNames.max.x - 2, yAvg)).w;

				gui->scissorPush(rectNames);
				Rect tr = getTextLineRect(text, gui->font, vec2(rectNames.max.x - 2, yAvg), vec2i(1,-1));
				if(gui->buttonUndocked(text, tr, 2, gui->colors.panelColor)) ds->lineGraphHighlight = timerIndex;
				gui->scissorPop();

				Rect rectNamesAndLines = rect(rectNames.min, rectLines.max);
				gui->scissorPush(rectNamesAndLines);
				dcLine2d(vec2(rectLines.min.x - textWidth - 2, yAvg), vec2(rectLines.max.x, yAvg), color);
				gui->scissorPop();

				gui->scissorPush(rectLines);

				if(timerIndex == ds->lineGraphHighlight) dcState(STATE_LINEWIDTH, 3);
				else dcState(STATE_LINEWIDTH, 1);

				bool firstEmpty = ds->timings[0][timerIndex].cyclesOverHits == 0;
				Vec2 p = vec2(rectLines.min.x, 0);
				if(firstEmpty) p.y = yAvg;
				else p.y = mapRange(ds->timings[0][timerIndex].cyclesOverHits, orthoBottom, orthoTop, rBottom, rTop);
				for(int i = 1; i < cycleCount; i++) {
					Timings* t = &ds->timings[i][timerIndex];

					bool lastElementEmpty = false;
					if(t->cyclesOverHits == 0) {
						if(i != cycleCount-1) continue;
						else lastElementEmpty = true;
					}

					float y = mapRange(t->cyclesOverHits, orthoBottom, orthoTop, rBottom, rTop);
					float xOff = rectDim(rectLines).w/(cycleCount-1);
					Vec2 np = vec2(rectLines.min.x + xOff*i, y);

					if(lastElementEmpty) np.y = yAvg;

					dcLine2d(p, np, color);
					p = np;
				}

				dcState(STATE_LINEWIDTH, 1);

				gui->scissorPop();
			}

			gui->empty();
			Rect r = gui->getCurrentRegion();
			Vec2 rc = rectCen(r);
			float rw = rectDim(r).w;

			// Draw color rectangles.
			float width = (rw/timerCount)*0.75f;
			float height = fontHeight*0.8f;
			float sw = (rw-(timerCount*width))/(timerCount+1);

			for(int i = 0; i < timerCount; i++) {
				TimerInfo* info = &timer->timerInfos[i];

				Vec4 color = vec4(info->color[0], info->color[1], info->color[2], 1);
				Vec2 pos = vec2(r.min.x + sw+width/2 + i*(width+sw), rc.y);
				dcRect(rectCenDim(pos, vec2(width, height)), color);
			}

		}

		gui->end();

	}

	//
	// Dropdown Console.
	//

	{
		Console* con = &ds->console;

		if(init) {
			con->init(ws->currentRes.y);
		}

		bool smallExtension = input->keysPressed[KEYCODE_F5] && !input->keysDown[KEYCODE_CTRL];
		bool bigExtension = input->keysPressed[KEYCODE_F5] && input->keysDown[KEYCODE_CTRL];

		con->update(ds->input, vec2(ws->currentRes), ad->dt, smallExtension, bigExtension);

		// Execute commands.

		if(con->commandAvailable) {
			con->commandAvailable = false;

			char* comName = con->comName;
			char** args = con->comArgs;
			char* resultString = "";
			bool pushResult = true;

			if(strCompare(comName, "add")) {
				int a = strToInt(args[0]);
				int b = strToInt(args[1]);

				resultString = fillString("%i + %i = %i.", a, b, a+b);

			} else if(strCompare(comName, "addFloat")) {
				float a = strToFloat(args[0]);
				float b = strToFloat(args[1]);

				resultString = fillString("%f + %f = %f.", a, b, a+b);

			} else if(strCompare(comName, "print")) {
				resultString = fillString("\"%s\"", args[0]);

			} else if(strCompare(comName, "cls")) {
				con->clearMainBuffer();
				pushResult = false;

			} else if(strCompare(comName, "doNothing")) {

			} else if(strCompare(comName, "setGuiAlpha")) {
				ds->guiAlpha = strToFloat(args[0]);

			} else if(strCompare(comName, "exit")) {
				*isRunning = false;

			}
			if(pushResult) con->pushToMainBuffer(resultString);
		}

		con->updateBody();

	}

	// Notifications.
	{
		// Update notes.
		int deletionCount = 0;
		for(int i = 0; i < ds->notificationCount; i++) {
			ds->notificationTimes[i] -= ds->dt;
			if(ds->notificationTimes[i] <= 0) {
				deletionCount++;
			}
		}

		// Delete expired notes.
		if(deletionCount > 0) {
			for(int i = 0; i < ds->notificationCount-deletionCount; i++) {
				ds->notificationStack[i] = ds->notificationStack[i+deletionCount];
				ds->notificationTimes[i] = ds->notificationTimes[i+deletionCount];
			}
			ds->notificationCount -= deletionCount;
		}

		// Draw notes.
		int fontSize = ds->fontHeight;
		Font* font = getFont(FONT_CALIBRI, fontSize);
		Vec4 color = vec4(1,0.5f,0,1);

		float y = -fontSize/2;
		for(int i = 0; i < ds->notificationCount; i++) {
			char* note = ds->notificationStack[i];
			dcText(note, font, vec2(ws->currentRes.w/2, y), color, vec2i(0,0), 0, 2);
			y -= fontSize;
		}
	}

	if(ds->showHud) {
		int fontSize = ds->fontHeight*1.1f;
		int pi = 0;
		// Vec4 c = vec4(1.0f,0.2f,0.0f,1);
		Vec4 c = vec4(1.0f,0.4f,0.0f,1);
		Vec4 c2 = vec4(0,0,0,1);
		Font* font = getFont(FONT_CONSOLAS, fontSize);
		int sh = 1;
		Vec2 offset = vec2(6,6);
		Vec2i ali = vec2i(1,1);

		Vec2 tp = vec2(ad->wSettings.currentRes.x, 0) - offset;

		static f64 timer = 0;
		static int fpsCounter = 0;
		static int fps = 0;
		timer += ds->dt;
		fpsCounter++;
		if(timer >= 1.0f) {
			fps = fpsCounter;
			fpsCounter = 0;
			timer = 0;
		}

		dcText(fillString("Fps  : %i", fps), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Pos  : (%f,%f,%f)", PVEC3(ad->activeCam.pos)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Pos  : (%f,%f,%f)", PVEC3(ad->selectedBlock)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Look : (%f,%f,%f)", PVEC3(ad->activeCam.look)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Up   : (%f,%f,%f)", PVEC3(ad->activeCam.up)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Right: (%f,%f,%f)", PVEC3(ad->activeCam.right)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Rot  : (%f,%f)",    PVEC2(ad->player->rot)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Vec  : (%f,%f,%f)", PVEC3(ad->player->vel)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Acc  : (%f,%f,%f)", PVEC3(ad->player->acc)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Draws: (%i)", 	   ad->voxelDrawCount), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Quads: (%i)", 	   ad->voxelTriangleCount), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("BufferIndex: %i",    ds->timer->bufferIndex), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("LastBufferIndex: %i",ds->lastBufferIndex), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;


		for(int i = 0; i < ds->infoStackCount; i++) {
			dcText(fillString("%s", ds->infoStack[i]), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		}
		ds->infoStackCount = 0;
	}


	if(*isRunning == false) {
		guiSave(ds->gui, 2, 0);
		if(globalDebugState->gui2) guiSave(globalDebugState->gui2, 2, 3);
	}

	// Update debugTime every second.
	static f64 tempTime = 0;
	tempTime += ds->dt;
	if(tempTime >= 1) {
		ds->debugTime = timerStop(&ds->tempTimer);
		tempTime = 0;
	}
}

#pragma optimize( "", on ) 

#endif




void debugMain(DebugState* ds, AppMemory* appMemory, AppData* ad, bool reload, bool* isRunning, bool init, ThreadQueue* threadQueue) {
	// @DebugStart.

	globalMemory->debugMode = true;

	timerStart(&ds->tempTimer);

	Input* input = ds->input;
	WindowSettings* ws = &ad->wSettings;

	clearTMemoryDebug();

	ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[1];
	ExtendibleMemoryArray* pMemory = globalMemory->pMemory;

	int clSize = megaBytes(2);
	drawCommandListInit(&ds->commandListDebug, (char*)getTMemoryDebug(clSize), clSize);
	globalCommandList = &ds->commandListDebug;


	ds->gInput = { input->mousePos, input->mouseWheel, input->mouseButtonPressed[0], input->mouseButtonDown[0], 
					input->keysPressed[KEYCODE_ESCAPE], input->keysPressed[KEYCODE_RETURN], input->keysPressed[KEYCODE_SPACE], input->keysPressed[KEYCODE_BACKSPACE], input->keysPressed[KEYCODE_DEL], input->keysPressed[KEYCODE_HOME], input->keysPressed[KEYCODE_END], 
					input->keysPressed[KEYCODE_LEFT], input->keysPressed[KEYCODE_RIGHT], input->keysPressed[KEYCODE_UP], input->keysPressed[KEYCODE_DOWN], 
					input->keysDown[KEYCODE_SHIFT], input->keysDown[KEYCODE_CTRL], input->inputCharacters, input->inputCharacterCount};

	if(input->keysPressed[KEYCODE_F6]) ds->showMenu = !ds->showMenu;
	if(input->keysPressed[KEYCODE_F7]) ds->showStats = !ds->showStats;
	if(input->keysPressed[KEYCODE_F8]) ds->showHud = !ds->showHud;

	// Recording update.
	{
		if(ds->playbackSwapMemory) {
			threadQueueComplete(threadQueue);
			ds->playbackSwapMemory = false;

			pMemory->index = ds->snapShotCount-1;
			pMemory->arrays[pMemory->index].index = ds->snapShotMemoryIndex;

			for(int i = 0; i < ds->snapShotCount; i++) {
				memCpy(pMemory->arrays[i].data, ds->snapShotMemory[i], pMemory->slotSize);
			}
		}
	}

	if(ds->showMenu) {
		int fontSize = ds->fontHeight;

		bool initSections = false;

		Gui* gui = ds->gui;
		gui->start(ds->gInput, getFont(FONT_CALIBRI, fontSize), ws->currentRes);

		static bool sectionGuiRecording = false;
		if(gui->beginSection("Recording", &sectionGuiRecording)) {

			bool noActiveThreads = threadQueueFinished(threadQueue);

			gui->div(vec2(0,0));
			gui->label("Active Threads:");
			gui->label(fillString("%i", !noActiveThreads));

			gui->div(vec2(0,0));
			gui->label("Max Frames:");
			gui->label(fillString("%i", ds->inputCapacity));

			gui->div(vec2(0,0));
			if(gui->switcher("Record", &ds->recordingInput)) {
				if(ds->playbackInput || !noActiveThreads) ds->recordingInput = false;

				if(ds->recordingInput) {
					if(threadQueueFinished(threadQueue)) {

						ds->snapShotCount = pMemory->index+1;
						ds->snapShotMemoryIndex = pMemory->arrays[pMemory->index].index;
						for(int i = 0; i < ds->snapShotCount; i++) {
							if(ds->snapShotMemory[i] == 0) 
								ds->snapShotMemory[i] = (char*)malloc(pMemory->slotSize);

							memCpy(ds->snapShotMemory[i], pMemory->arrays[i].data, pMemory->slotSize);
						}


						ds->recordingInput = true;
						ds->inputIndex = 0;
					}
				}
			}
			gui->label(fillString("%i", ds->inputIndex));


			if(ds->inputIndex > 0 && !ds->recordingInput) {
				char* s = ds->playbackInput ? "Stop Playback" : "Start Playback";

				if(gui->switcher(s, &ds->playbackInput)) {
					if(ds->playbackInput) {
						threadQueueComplete(threadQueue);
						ds->playbackIndex = 0;

						pMemory->index = ds->snapShotCount-1;
						pMemory->arrays[pMemory->index].index = ds->snapShotMemoryIndex;

						for(int i = 0; i < ds->snapShotCount; i++) {
							memCpy(pMemory->arrays[i].data, ds->snapShotMemory[i], pMemory->slotSize);
						}
					} else {
						ds->playbackPause = false;
						ds->playbackBreak = false;
					}
				}

				if(ds->playbackInput) {
					gui->div(vec2(0,0));

					gui->switcher("Pause/Resume", &ds->playbackPause);

					int cap = ds->playbackIndex;
					gui->slider(&ds->playbackIndex, 0, ds->inputIndex - 1);
					ds->playbackIndex = cap;

					gui->div(vec3(0.25f,0.25f,0));
					if(gui->button("Step")) {
						ds->playbackBreak = true;
						ds->playbackPause = false;
						ds->playbackBreakIndex = (ds->playbackIndex + 1)%ds->inputIndex;
					}
					gui->switcher("Break", &ds->playbackBreak);
					gui->slider(&ds->playbackBreakIndex, 0, ds->inputIndex - 1);
				}
			}

		} gui->endSection();

		static bool sectionGuiSettings = initSections;
		if(gui->beginSection("GuiSettings", &sectionGuiSettings)) {
			guiSettings(gui);
		} gui->endSection();

		static bool sectionSettings = initSections;
		if(gui->beginSection("Settings", &sectionSettings)) {
			gui->div(vec2(0,0)); if(gui->button("Compile")) shellExecute("C:\\Projects\\Hmm\\code\\buildWin32.bat");
								 if(gui->button("Up Buffers")) ad->updateFrameBuffers = true;
			gui->div(vec2(0,0)); gui->label("FoV", 0); gui->slider(&ad->fieldOfView, 1, 180);
			gui->div(vec2(0,0)); gui->label("MSAA", 0); gui->slider(&ad->msaaSamples, 1, 8);
			gui->switcher("Native Res", &ad->useNativeRes);
			gui->div(0,0,0); gui->label("FboRes", 0); gui->slider(&ad->fboRes.x, 150, ad->cur3dBufferRes.x); gui->slider(&ad->fboRes.y, 150, ad->cur3dBufferRes.y);
			gui->div(0,0,0); gui->label("NFPlane", 0); gui->slider(&ad->nearPlane, 0.01, 2); gui->slider(&ad->farPlane, 1000, 5000);
		} gui->endSection();

		static bool sectionEntities = true;
		if(gui->beginSection("Entities", &sectionEntities)) { 

			EntityList* list = &ad->entityList;
			for(int i = 0; i < list->size; i++) {
				Entity* e = list->e + i;

				if(e->init) {
					guiPrintIntrospection(gui, STRUCTTYPE_ENTITY, (char*)e);
				}
			}

		} gui->endSection();

		addDebugInfo(fillString("%i", ad->entityList.size));

		static bool sectionWorld = initSections;
		if(gui->beginSection("World", &sectionWorld)) { 
			if(gui->button("Reload World") || input->keysPressed[KEYCODE_TAB]) ad->reloadWorld = true;
			
			gui->div(vec2(0,0)); gui->label("RefAlpha", 0); gui->slider(&reflectionAlpha, 0, 1);
			gui->div(vec2(0,0)); gui->label("Light", 0); gui->slider(&globalLumen, 0, 255);
			gui->div(0,0,0); gui->label("MinMax", 0); gui->slider(&WORLD_MIN, 0, 255); gui->slider(&WORLD_MAX, 0, 255);
			gui->div(vec2(0,0)); gui->label("WaterLevel", 0); 
				if(gui->slider(&waterLevelValue, 0, 0.2f)) WATER_LEVEL_HEIGHT = lerp(waterLevelValue, WORLD_MIN, WORLD_MAX);

			gui->div(vec2(0,0)); gui->label("WFreq", 0); gui->slider(&worldFreq, 0.0001f, 0.02f);
			gui->div(vec2(0,0)); gui->label("WDepth", 0); gui->slider(&worldDepth, 1, 10);
			gui->div(vec2(0,0)); gui->label("MFreq", 0); gui->slider(&modFreq, 0.001f, 0.1f);
			gui->div(vec2(0,0)); gui->label("MDepth", 0); gui->slider(&modDepth, 1, 10);
			gui->div(vec2(0,0)); gui->label("MOffset", 0); gui->slider(&modOffset, 0, 1);
			gui->div(vec2(0,0)); gui->label("PowCurve", 0); gui->slider(&worldPowCurve, 1, 6);
			gui->div(0,0,0,0); gui->slider(heightLevels+0,0,1); gui->slider(heightLevels+1,0,1);
								  gui->slider(heightLevels+2,0,1); gui->slider(heightLevels+3,0,1);
		} gui->endSection();

		gui->end();
	}

	ds->timer->timerInfoCount = __COUNTER__;

	int fontHeight = 18;
	Timer* timer = ds->timer;
	int cycleCount = arrayCount(ds->timings);

	bool threadsFinished = threadQueueFinished(threadQueue);

	int bufferIndex = timer->bufferIndex;

	// Save const strings from initialised timerinfos.
	{
		int timerCount = timer->timerInfoCount;
		for(int i = 0; i < timerCount; i++) {
			TimerInfo* info = timer->timerInfos + i;

			// Set colors.
			float ss = i%(timerCount/2) / ((float)timerCount/2);
			float h = i < timerCount/2 ? 0.1f : -0.1f;
			Vec3 color = vec3(0,0,0);
			hslToRgb(color.e, 360*ss, 0.5f, 0.5f+h);

			vSet3(info->color, color.r, color.g, color.b);


			if(!info->initialised || info->stringsSaved) continue;
			char* s;
			
			s = info->file;
			info->file = getPStringDebug(strLen(s) + 1);
			strCpy(info->file, s);

			s = info->function;
			info->function = getPStringDebug(strLen(s) + 1);
			strCpy(info->function, s);

			s = info->name;
			info->name = getPStringDebug(strLen(s) + 1);
			strCpy(info->name, s);

			info->stringsSaved = true;
		}
	}

	if(ds->setPause) {
		ds->lastCycleIndex = ds->cycleIndex;
		ds->cycleIndex = mod(ds->cycleIndex-1, arrayCount(ds->timings));

		ds->timelineCamSize = -1;
		ds->timelineCamPos = -1;

		ds->setPause = false;
	}
	if(ds->setPlay) {
		ds->cycleIndex = ds->lastCycleIndex;
		ds->setPlay = false;
	}

	Timings* timings = ds->timings[ds->cycleIndex];
	Statistic* statistics = ds->statistics[ds->cycleIndex];

	int cycleIndex = ds->cycleIndex;
	int newCycleIndex = (ds->cycleIndex + 1)%cycleCount;

	// Timer update.
	{

		if(!ds->noCollating) {
			zeroMemory(timings, timer->timerInfoCount*sizeof(Timings));
			zeroMemory(statistics, timer->timerInfoCount*sizeof(Statistic));

			ds->cycleIndex = newCycleIndex;

			// Collate timing buffer.

			// for(int threadIndex = 0; threadIndex < threadQueue->threadCount; threadIndex++) 
			{
				// GraphSlot* graphSlots = ds->graphSlots[threadIndex];
				// int index = ds->graphSlotCount[threadIndex];

				for(int i = ds->lastBufferIndex; i < bufferIndex; ++i) {
					TimerSlot* slot = timer->timerBuffer + i;
					
					int threadIndex = threadIdToIndex(threadQueue, slot->threadId);

					if(slot->type == TIMER_TYPE_BEGIN) {
						int index = ds->graphSlotCount[threadIndex];

						GraphSlot graphSlot;
						graphSlot.threadIndex = threadIndex;
						graphSlot.timerIndex = slot->timerIndex;
						graphSlot.stackIndex = index;
						graphSlot.cycles = slot->cycles;
						ds->graphSlots[threadIndex][index] = graphSlot;

						ds->graphSlotCount[threadIndex]++;
					} else {
						ds->graphSlotCount[threadIndex]--;
						int index = ds->graphSlotCount[threadIndex];
						if(index < 0) index = 0; // @Hack, to keep things running.

						ds->graphSlots[threadIndex][index].size = slot->cycles - ds->graphSlots[threadIndex][index].cycles;
						ds->savedBuffer[ds->savedBufferIndex] = ds->graphSlots[threadIndex][index];
						ds->savedBufferIndex = (ds->savedBufferIndex+1)%ds->savedBufferMax;
						ds->savedBufferCount = clampMax(ds->savedBufferCount + 1, ds->savedBufferMax);


						Timings* timing = timings + ds->graphSlots[threadIndex][index].timerIndex;
						timing->cycles += ds->graphSlots[threadIndex][index].size;
						timing->hits++;
					}
				}

				// ds->graphSlotCount[threadIndex] = index;
			}

			// ds->savedBufferCounts[cycleIndex] = savedBufferCount;

			for(int i = 0; i < timer->timerInfoCount; i++) {
				Timings* t = timings + i;
				t->cyclesOverHits = t->hits > 0 ? (u64)(t->cycles/t->hits) : 0; 
			}

			for(int timerIndex = 0; timerIndex < timer->timerInfoCount; timerIndex++) {
				Statistic* stat = statistics + timerIndex;
				beginStatistic(stat);

				for(int i = 0; i < arrayCount(ds->timings); i++) {
					Timings* t = &ds->timings[i][timerIndex];
					if(t->hits == 0) continue;

					updateStatistic(stat, t->cyclesOverHits);
				}

				endStatistic(stat);
				if(stat->count == 0) stat->avg = 0;
			}
		}
	}

	ds->lastBufferIndex = bufferIndex;

	if(threadsFinished) {
		timer->bufferIndex = 0;
		ds->lastBufferIndex = 0;
	}

	assert(timer->bufferIndex < timer->bufferSize);

	if(init) {
		ds->lineGraphCamSize = 700000;
		ds->lineGraphCamPos = 0;
		ds->mode = 0;
		ds->lineGraphHeight = 30;
		ds->lineGraphHighlight = 0;
	}

	//
	// Draw timing info.
	//

	if(ds->showStats) 
	{
		static int highlightedIndex = -1;
		Vec4 highlightColor = vec4(1,1,1,0.05f);

		// float cyclesPerFrame = (float)((3.5f*((float)1/60))*1024*1024*1024);
		float cyclesPerFrame = (float)((3.5f*((float)1/60))*1000*1000*1000);
		int fontSize = ds->fontHeight;
		Vec2 textPos = vec2(550, -fontHeight);
		int infoCount = timer->timerInfoCount;

		Gui* gui = ds->gui2;
		gui->start(ds->gInput, getFont(FONT_CALIBRI, fontHeight), ws->currentRes);

		gui->label("App Statistics", 1, gui->colors.sectionColor, vec4(0,0,0,1));

		float sectionWidth = 120;
		float headerDivs[] = {sectionWidth,sectionWidth,sectionWidth,0,80,80};
		gui->div(headerDivs, arrayCount(headerDivs));
		if(gui->button("Data", (int)(ds->mode == 0) + 1)) ds->mode = 0;
		if(gui->button("Line graph", (int)(ds->mode == 1) + 1)) ds->mode = 1;
		if(gui->button("Timeline", (int)(ds->mode == 2) + 1)) ds->mode = 2;
		gui->empty();
		gui->label(fillString("%fms", ds->debugTime*1000), 1);
		gui->label(fillString("%fms", ds->debugRenderTime*1000), 1);

		gui->div(vec2(0.2f,0));
		if(gui->switcher("Freeze", &ds->noCollating)) {
			if(ds->noCollating) ds->setPause = true;
			else ds->setPlay = true;
		}
		gui->slider(&ds->cycleIndex, 0, cycleCount-1);

		if(ds->mode == 0)
		{
			int barWidth = 1;
			int barCount = arrayCount(ds->timings);
			float sectionWidths[] = {0,0.2f,0,0,0,0,0,0, barWidth*barCount};
			// float sectionWidths[] = {0.1f,0,0.1f,0,0.05f,0,0,0.1f, barWidth*barCount};

			char* headers[] = {"File", "Function", "Description", "Cycles", "Hits", "C/H", "Avg. Cycl.", "Total Time", "Graphs"};
			gui->div(sectionWidths, arrayCount(sectionWidths));

			float textSectionEnd;
			for(int i = 0; i < arrayCount(sectionWidths); i++) {
				// @Hack: Get the end of the text region by looking at last region.
				if(i == arrayCount(sectionWidths)-1) textSectionEnd = gui->getCurrentRegion().max.x;

				Vec4 buttonColor = vec4(gui->colors.regionColor.rgb, 0.5f);
				if(gui->button(headers[i], 0, 1, buttonColor, vec4(0,0,0,1))) {
					if(abs(ds->graphSortingIndex) == i) ds->graphSortingIndex *= -1;
					else ds->graphSortingIndex = i;
				}
			}

			SortPair* sortList = getTArrayDebug(SortPair, infoCount+1);
			{
				for(int i = 0; i < infoCount+1; i++) sortList[i].index = i;

		   			 if(abs(ds->graphSortingIndex) == 3) for(int i = 0; i < infoCount+1; i++) sortList[i].key = timings[i].cycles;
		   		else if(abs(ds->graphSortingIndex) == 4) for(int i = 0; i < infoCount+1; i++) sortList[i].key = timings[i].hits;
		   		else if(abs(ds->graphSortingIndex) == 5) for(int i = 0; i < infoCount+1; i++) sortList[i].key = timings[i].cyclesOverHits;
		   		else if(abs(ds->graphSortingIndex) == 6) for(int i = 0; i < infoCount+1; i++) sortList[i].key = statistics[i].avg;
		   		else if(abs(ds->graphSortingIndex) == 7) for(int i = 0; i < infoCount+1; i++) sortList[i].key = timings[i].cycles/cyclesPerFrame;

		   		bool sortDirection = true;
		   		if(ds->graphSortingIndex < 0) sortDirection = false;

		   		if(valueBetween(abs(ds->graphSortingIndex), 3, 7)) 
					bubbleSort(sortList, infoCount, sortDirection);
			}

			for(int index = 0; index < infoCount; index++) {
				int i = sortList[index].index;

				TimerInfo* tInfo = timer->timerInfos + i;
				Timings* timing = timings + i;

				if(!tInfo->initialised) continue;

				gui->div(sectionWidths, arrayCount(sectionWidths)); 

				// if(highlightedIndex == i) {
				// 	Rect r = gui->getCurrentRegion();
				// 	Rect line = rect(r.min, vec2(textSectionEnd,r.min.y + fontHeight));
				// 	dcRect(line, highlightColor);
				// }

				gui->label(fillString("%s", tInfo->file + 21),0);
				if(gui->button(fillString("%s", tInfo->function),0, 0, vec4(gui->colors.regionColor.rgb, 0.2f))) {
					char* command = fillString("%s %s:%i", editor_executable_path, tInfo->file, tInfo->line);
					shellExecuteNoWindow(command);
				}
				gui->label(fillString("%s", tInfo->name),0);
				gui->label(fillString("%i64.c", timing->cycles),2);
				gui->label(fillString("%i64.", timing->hits),2);
				gui->label(fillString("%i64.c", timing->cyclesOverHits),2);
				gui->label(fillString("%i64.c", (i64)statistics[i].avg),2); // Not a i64 but whatever.
				gui->label(fillString("%.3f%%", ((float)timing->cycles/cyclesPerFrame)*100),2);

				// Bar graphs.
				dcState(STATE_LINEWIDTH, barWidth);

				gui->empty();
				Rect r = gui->getCurrentRegion();
				float rheight = gui->getDefaultHeight();
				float fontBaseOffset = 4;

				float xOffset = 0;
				for(int statIndex = 0; statIndex < barCount; statIndex++) {
					Statistic* stat = statistics + i;
					u64 coh = ds->timings[statIndex][i].cyclesOverHits;

					float height = mapRangeClamp(coh, stat->min, stat->max, 1, rheight);
					Vec2 rmin = r.min + vec2(xOffset, fontBaseOffset);
					float colorOffset = mapRange(coh, stat->min, stat->max, 0, 1);
					// dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,1-colorOffset,0,1));
					dcLine2d(rmin, rmin+vec2(0,height), vec4(colorOffset,1-colorOffset,0,1));

					xOffset += barWidth;
				}
			}
		}

		// Timeline graph.
		if(ds->mode == 2 && ds->noCollating)
		{
			float lineHeightOffset = 1.2;

			gui->empty();
			Rect cyclesRect = gui->getCurrentRegion();
			gui->heightPush(1.5f);
			gui->empty();
			Rect headerRect = gui->getCurrentRegion();
			gui->heightPop();

			float lineHeight = fontHeight * lineHeightOffset;

			gui->heightPush(3*lineHeight +  2*lineHeight*(threadQueue->threadCount-1));
			gui->empty();
			Rect bgRect = gui->getCurrentRegion();
			gui->heightPop();

			float graphWidth = rectDim(bgRect).w;

			int swapTimerIndex = 0;
			for(int i = 0; i < timer->timerInfoCount; i++) {
				if(!timer->timerInfos[i].initialised) continue;

				if(strCompare(timer->timerInfos[i].name, "Swap")) {
					swapTimerIndex = i;
					break;
				}
			}

			int recentIndex = mod(ds->savedBufferIndex-1, ds->savedBufferMax);
			int oldIndex = mod(ds->savedBufferIndex - ds->savedBufferCount, ds->savedBufferMax);
			GraphSlot recentSlot = ds->savedBuffer[recentIndex];
			GraphSlot oldSlot = ds->savedBuffer[oldIndex];
			double cyclesLeft = oldSlot.cycles;
			double cyclesRight = recentSlot.cycles + recentSlot.size;
			double cyclesSize = cyclesRight - cyclesLeft;

			// Setup cam pos and zoom.
			if(ds->timelineCamPos == -1 && ds->timelineCamSize == -1) {
				ds->timelineCamSize = (recentSlot.cycles + recentSlot.size) - oldSlot.cycles;
				ds->timelineCamPos = oldSlot.cycles + ds->timelineCamSize/2;
			}

			if(gui->input.mouseWheel) {
				float wheel = gui->input.mouseWheel;

				float offset = wheel < 0 ? 1.1f : 1/1.1f;
				if(!input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.2f : 1/1.2f;
				if(input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.4f : 1/1.4f;

				double oldZoom = ds->timelineCamSize;
				ds->timelineCamSize *= offset;
				clampDouble(&ds->timelineCamSize, 1000, cyclesSize);
				double diff = ds->timelineCamSize - oldZoom;

				float zoomOffset = mapRange(input->mousePos.x, bgRect.min.x, bgRect.max.x, -0.5f, 0.5f);
				ds->timelineCamPos -= diff*zoomOffset;
			}


			Vec2 dragDelta = vec2(0,0);
			gui->drag(bgRect, &dragDelta, vec4(0,0,0,0));

			ds->timelineCamPos -= dragDelta.x * (ds->timelineCamSize/graphWidth);
			clampDouble(&ds->timelineCamPos, cyclesLeft + ds->timelineCamSize/2, cyclesRight - ds->timelineCamSize/2);


			double camPos = ds->timelineCamPos;
			double zoom = ds->timelineCamSize;
			double orthoLeft = camPos - zoom/2;
			double orthoRight = camPos + zoom/2;


			// Header.
			{
				dcRect(cyclesRect, gui->colors.sectionColor);
				Vec2 cyclesDim = rectDim(cyclesRect);

				dcRect(headerRect, vec4(1,1,1,0.1f));
				Vec2 headerDim = rectDim(headerRect);

				{
					float viewAreaLeft = mapRangeDouble(orthoLeft, cyclesLeft, cyclesRight, cyclesRect.min.x, cyclesRect.max.x);
					float viewAreaRight = mapRangeDouble(orthoRight, cyclesLeft, cyclesRight, cyclesRect.min.x, cyclesRect.max.x);

					float viewSize = viewAreaRight - viewAreaLeft;
					float viewMid = viewAreaRight + viewSize/2;
					float viewMinSize = 2;
					if(viewSize < viewMinSize) {
						viewAreaLeft = viewMid - viewMinSize*0.5;
						viewAreaRight = viewMid + viewMinSize*0.5;
					}

					dcRect(rect(viewAreaLeft, cyclesRect.min.y, viewAreaRight, cyclesRect.max.y), vec4(1,1,1,0.03f));
				}

				float g = 0.7f;
				float heightMod = 0.0f;
				double div = 4;
				double divMod = (1/div) + 0.05f;

				double timelineSection = div;
				while(timelineSection < zoom*divMod*(ws->currentRes.h/(graphWidth))) {
					timelineSection *= div;
					heightMod += 0.1f;
				}

				clampMax(&heightMod, 1);

				dcState(STATE_LINEWIDTH, 3);
				double startPos = roundModDouble(orthoLeft, timelineSection) - timelineSection;
				double pos = startPos;
				while(pos < orthoRight + timelineSection) {
					double p = mapRangeDouble(pos, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);

					// Big line.
					{
						float h = headerDim.h*heightMod;
						dcLine2d(vec2(p,headerRect.min.y), vec2(p,headerRect.min.y + h), vec4(g,g,g,1));
					}

					// Text
					{
						Vec2 textPos = vec2(p,cyclesRect.min.y + cyclesDim.h/2);
						float percent = mapRange(pos, cyclesLeft, cyclesRight, 0, 100);
						int percentInterval = mapRangeDouble(timelineSection, 0, cyclesSize, 0, 100);

						char* s;
						if(percentInterval > 10) s = fillString("%i%%", (int)percent);
						else if(percentInterval > 1) s = fillString("%.1f%%", percent);
						else if(percentInterval > 0.1) s = fillString("%.2f%%", percent);
						else s = fillString("%.3f%%", percent);

						float tw = getTextDim(s, gui->font).w;
						if(valueBetween(bgRect.min.x, textPos.x - tw/2, textPos.x + tw/2)) textPos.x = bgRect.min.x + tw/2;
						if(valueBetween(bgRect.max.x, textPos.x - tw/2, textPos.x + tw/2)) textPos.x = bgRect.max.x - tw/2;

						dcText(s, gui->font, textPos, gui->colors.textColor, vec2i(0,0), 0, 1, gui->colors.shadowColor);
					}

					pos += timelineSection;
				}
				dcState(STATE_LINEWIDTH, 1);

				pos = startPos;
				timelineSection /= div;
				heightMod *= 0.6f;
				int index = 0;
				while(pos < orthoRight + timelineSection) {

					// Small line.
					if((index%(int)div) != 0) {
						double p = mapRangeDouble(pos, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);
						float h = headerDim.h*heightMod;
						dcLine2d(vec2(p,headerRect.min.y), vec2(p,headerRect.min.y + h), vec4(g,g,g,1));
					}

					// Cycle text.
					{
						float pMid = mapRangeDouble(pos - timelineSection/2, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);
						Vec2 textPos = vec2(pMid,headerRect.min.y + headerDim.h/3);

						double cycles = timelineSection;
						char* s;
						if(cycles < 1000) s = fillString("%ic", (int)cycles);
						else if(cycles < 1000000) s = fillString("%.1fkc", cycles/1000);
						else if(cycles < 1000000000) s = fillString("%.1fmc", cycles/1000000);
						else if(cycles < 1000000000000) s = fillString("%.1fbc", cycles/1000000000);
						else s = fillString("INF");

						dcText(s, gui->font, textPos, gui->colors.textColor, vec2i(0,0), 0, gui->settings.textShadow, gui->colors.shadowColor);
					}

					pos += timelineSection;
					index++;

				}
			}

			dcState(STATE_LINEWIDTH, 1);

			bool mouseHighlight = false;
			Rect hRect;
			Vec4 hc;
			char* hText;
			GraphSlot* hSlot;

			Vec2 startPos = rectTL(bgRect);
			startPos -= vec2(0, lineHeight);

			int firstBufferIndex = oldIndex;
			int bufferCount = ds->savedBufferCount;
			for(int threadIndex = 0; threadIndex < threadQueue->threadCount; threadIndex++) {

				// Horizontal lines to distinguish thread bars.
				if(threadIndex > 0) {
					Vec2 p = startPos + vec2(0,lineHeight);
					float g = 0.8f;
					dcLine2d(p, vec2(bgRect.max.x, p.y), vec4(g,g,g,1));
				}

				for(int i = 0; i < bufferCount; ++i) {
					GraphSlot* slot = ds->savedBuffer + ((firstBufferIndex+i)%ds->savedBufferMax);
					if(slot->threadIndex != threadIndex) continue;

					Timings* t = timings + slot->timerIndex;
					TimerInfo* tInfo = timer->timerInfos + slot->timerIndex;

					if(slot->cycles + slot->size < orthoLeft || slot->cycles > orthoRight) continue;


					double barLeft = mapRangeDouble(slot->cycles, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);
					double barRight = mapRangeDouble(slot->cycles + slot->size, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);

					// Draw vertical line at swap boundaries.
					if(slot->timerIndex == swapTimerIndex) {
						float g = 0.8f;
						dcLine2d(vec2(barRight, bgRect.min.y), vec2(barRight, bgRect.max.y), vec4(g,g,g,1));
					}

					// Bar min size is 1.
					if(barRight - barLeft < 1) {
						double mid = barLeft + (barRight - barLeft)/2;
						barLeft = mid - 0.5f;
						barRight = mid + 0.5f;
					}

					float y = startPos.y+slot->stackIndex*-lineHeight;
					Rect r = rect(vec2(barLeft,y), vec2(barRight, y + lineHeight));

					float cOff = slot->timerIndex/(float)timer->timerInfoCount;
					Vec4 c = vec4(tInfo->color[0], tInfo->color[1], tInfo->color[2], 1);

					if(gui->getMouseOver(gui->input.mousePos, r)) {
						mouseHighlight = true;
						hRect = r;
						hc = c;

						hText = fillString("%s %s (%i.c)", tInfo->function, tInfo->name, slot->size);
						hSlot = slot;
					} else {
						float g = 0.1f;
						gui->drawRect(r, vec4(g,g,g,1));

						bool textRectVisible = (barRight - barLeft) > 1;
						if(textRectVisible) {
							if(barLeft < bgRect.min.x) r.min.x = bgRect.min.x;
							Rect textRect = rect(r.min+vec2(1,1), r.max-vec2(1,1));

							gui->drawTextBox(textRect, fillString("%s %s (%i.c)", tInfo->function, tInfo->name, slot->size), c, 0, rectDim(textRect).w);
						}
					}

				}

				if(threadIndex == 0) startPos.y -= lineHeight*3;
				else startPos.y -= lineHeight*2;

			}

			if(mouseHighlight) {
				if(hRect.min.x < bgRect.min.x) hRect.min.x = bgRect.min.x;

				float tw = getTextDim(hText, gui->font).w + 2;
				if(tw > rectDim(hRect).w) hRect.max.x = hRect.min.x + tw;

				float g = 0.8f;
				gui->drawRect(hRect, vec4(g,g,g,1));

				Rect textRect = rect(hRect.min+vec2(1,1), hRect.max-vec2(1,1));
				gui->drawTextBox(textRect, hText, hc);
			}

			gui->div(0.1f, 0); 
			gui->div(0.1f, 0); 

			if(gui->button("Reset")) {
				ds->timelineCamSize = (recentSlot.cycles + recentSlot.size) - oldSlot.cycles;
				ds->timelineCamPos = oldSlot.cycles + ds->timelineCamSize/2;
			}

			gui->label(fillString("Cam: %i64., Zoom: %i64.", (i64)ds->timelineCamPos, (i64)ds->timelineCamSize));
		}
		



		// Line graph.
		if(ds->mode == 1)
		{
			dcState(STATE_LINEWIDTH, 1);

			// Get longest function name string.
			float timerInfoMaxStringSize = 0;
			int cycleCount = arrayCount(ds->timings);
			int timerCount = ds->timer->timerInfoCount;
			for(int timerIndex = 0; timerIndex < timerCount; timerIndex++) {
				TimerInfo* info = &timer->timerInfos[timerIndex];
				if(!info->initialised) continue;

				Statistic* stat = &ds->statistics[cycleIndex][timerIndex];
				if(stat->avg == 0) continue;

				char* text = strLen(info->name) > 0 ? info->name : info->function;
				timerInfoMaxStringSize = max(getTextDim(text, gui->font).w, timerInfoMaxStringSize);
			}

			// gui->div(0.2f, 0);
			gui->slider(&ds->lineGraphHeight, 1, 60);
			// gui->empty();

			gui->heightPush(gui->getDefaultHeight() * ds->lineGraphHeight);
			gui->div(vec3(timerInfoMaxStringSize + 2, 0, 120));
			gui->empty(); Rect rectNames = gui->getCurrentRegion();
			gui->empty(); Rect rectLines = gui->getCurrentRegion();
			gui->empty(); Rect rectNumbers = gui->getCurrentRegion();
			gui->heightPop();

			float rTop = rectLines.max.y;
			float rBottom = rectLines.min.y;

			Vec2 dragDelta = vec2(0,0);
			gui->drag(rectLines, &dragDelta, vec4(0,0,0,0.2f));

			float wheel = gui->input.mouseWheel;
			if(wheel) {
				float offset = wheel < 0 ? 1.1f : 1/1.1f;
				if(!input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.2f : 1/1.2f;
				if(input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.4f : 1/1.4f;

				float heightDiff = ds->lineGraphCamSize;
				ds->lineGraphCamSize *= offset;
				ds->lineGraphCamSize = clampMin(ds->lineGraphCamSize, 0.00001f);
				heightDiff -= ds->lineGraphCamSize;

				float mouseOffset = mapRange(input->mousePosNegative.y, rBottom, rTop, -0.5f, 0.5f);
				ds->lineGraphCamPos += heightDiff * mouseOffset;
			}

			ds->lineGraphCamPos -= dragDelta.y * ((ds->lineGraphCamSize)/(rTop - rBottom));
			clampMin(&ds->lineGraphCamPos, ds->lineGraphCamSize/2.05f);

			float orthoTop = ds->lineGraphCamPos + ds->lineGraphCamSize/2;
			float orthoBottom = ds->lineGraphCamPos - ds->lineGraphCamSize/2;

			// Draw numbers.
			{
				gui->scissorPush(rectNumbers);

				float y = 0;
				float length = 10;

				float div = 10;
				float timelineSection = div;
				float splitMod = (1/div)*0.2f;
				while(timelineSection < ds->lineGraphCamSize*splitMod*(ws->currentRes.h/(rTop-rBottom))) timelineSection *= div;

				float start = roundMod(orthoBottom, timelineSection) - timelineSection;

				float p = start;
				while(p < orthoTop) {
					p += timelineSection;
					y = mapRange(p, orthoBottom, orthoTop, rBottom, rTop);

					dcLine2d(vec2(rectNumbers.min.x, y), vec2(rectNumbers.min.x + length, y), vec4(1,1,1,1)); 
					dcText(fillString("%i64.c",(i64)p), gui->font, vec2(rectNumbers.min.x + length + 4, y), vec4(1,1,1,1), vec2i(-1,0));
				}

				gui->scissorPop();
			}

			for(int timerIndex = 0; timerIndex < timerCount; timerIndex++) {
				TimerInfo* info = &timer->timerInfos[timerIndex];
				if(!info->initialised) continue;

				Statistic* stat = &ds->statistics[cycleIndex][timerIndex];
				if(stat->avg == 0) continue;

				float statMin = mapRange(stat->min, orthoBottom, orthoTop, rBottom, rTop);
				float statMax = mapRange(stat->max, orthoBottom, orthoTop, rBottom, rTop);
				if(statMax < rBottom || statMin > rTop) continue;

				Vec4 color = vec4(info->color[0], info->color[1], info->color[2], 1);

				float yAvg = mapRange(stat->avg, orthoBottom, orthoTop, rBottom, rTop);
				char* text = strLen(info->name) > 0 ? info->name : info->function;
				float textWidth = getTextDim(text, gui->font, vec2(rectNames.max.x - 2, yAvg)).w;

				gui->scissorPush(rectNames);
				Rect tr = getTextLineRect(text, gui->font, vec2(rectNames.max.x - 2, yAvg), vec2i(1,-1));
				if(gui->buttonUndocked(text, tr, 2, gui->colors.panelColor)) ds->lineGraphHighlight = timerIndex;
				gui->scissorPop();

				Rect rectNamesAndLines = rect(rectNames.min, rectLines.max);
				gui->scissorPush(rectNamesAndLines);
				dcLine2d(vec2(rectLines.min.x - textWidth - 2, yAvg), vec2(rectLines.max.x, yAvg), color);
				gui->scissorPop();

				gui->scissorPush(rectLines);

				if(timerIndex == ds->lineGraphHighlight) dcState(STATE_LINEWIDTH, 3);
				else dcState(STATE_LINEWIDTH, 1);

				bool firstEmpty = ds->timings[0][timerIndex].cyclesOverHits == 0;
				Vec2 p = vec2(rectLines.min.x, 0);
				if(firstEmpty) p.y = yAvg;
				else p.y = mapRange(ds->timings[0][timerIndex].cyclesOverHits, orthoBottom, orthoTop, rBottom, rTop);
				for(int i = 1; i < cycleCount; i++) {
					Timings* t = &ds->timings[i][timerIndex];

					bool lastElementEmpty = false;
					if(t->cyclesOverHits == 0) {
						if(i != cycleCount-1) continue;
						else lastElementEmpty = true;
					}

					float y = mapRange(t->cyclesOverHits, orthoBottom, orthoTop, rBottom, rTop);
					float xOff = rectDim(rectLines).w/(cycleCount-1);
					Vec2 np = vec2(rectLines.min.x + xOff*i, y);

					if(lastElementEmpty) np.y = yAvg;

					dcLine2d(p, np, color);
					p = np;
				}

				dcState(STATE_LINEWIDTH, 1);

				gui->scissorPop();
			}

			gui->empty();
			Rect r = gui->getCurrentRegion();
			Vec2 rc = rectCen(r);
			float rw = rectDim(r).w;

			// Draw color rectangles.
			float width = (rw/timerCount)*0.75f;
			float height = fontHeight*0.8f;
			float sw = (rw-(timerCount*width))/(timerCount+1);

			for(int i = 0; i < timerCount; i++) {
				TimerInfo* info = &timer->timerInfos[i];

				Vec4 color = vec4(info->color[0], info->color[1], info->color[2], 1);
				Vec2 pos = vec2(r.min.x + sw+width/2 + i*(width+sw), rc.y);
				dcRect(rectCenDim(pos, vec2(width, height)), color);
			}

		}

		gui->end();

	}

	//
	// Dropdown Console.
	//

	{
		Console* con = &ds->console;

		if(init) {
			con->init(ws->currentRes.y);
		}

		bool smallExtension = input->keysPressed[KEYCODE_F5] && !input->keysDown[KEYCODE_CTRL];
		bool bigExtension = input->keysPressed[KEYCODE_F5] && input->keysDown[KEYCODE_CTRL];

		con->update(ds->input, vec2(ws->currentRes), ad->dt, smallExtension, bigExtension);

		// Execute commands.

		if(con->commandAvailable) {
			con->commandAvailable = false;

			char* comName = con->comName;
			char** args = con->comArgs;
			char* resultString = "";
			bool pushResult = true;

			if(strCompare(comName, "add")) {
				int a = strToInt(args[0]);
				int b = strToInt(args[1]);

				resultString = fillString("%i + %i = %i.", a, b, a+b);

			} else if(strCompare(comName, "addFloat")) {
				float a = strToFloat(args[0]);
				float b = strToFloat(args[1]);

				resultString = fillString("%f + %f = %f.", a, b, a+b);

			} else if(strCompare(comName, "print")) {
				resultString = fillString("\"%s\"", args[0]);

			} else if(strCompare(comName, "cls")) {
				con->clearMainBuffer();
				pushResult = false;

			} else if(strCompare(comName, "doNothing")) {

			} else if(strCompare(comName, "setGuiAlpha")) {
				ds->guiAlpha = strToFloat(args[0]);

			} else if(strCompare(comName, "exit")) {
				*isRunning = false;

			}
			if(pushResult) con->pushToMainBuffer(resultString);
		}

		con->updateBody();

	}

	// Notifications.
	{
		// Update notes.
		int deletionCount = 0;
		for(int i = 0; i < ds->notificationCount; i++) {
			ds->notificationTimes[i] -= ds->dt;
			if(ds->notificationTimes[i] <= 0) {
				deletionCount++;
			}
		}

		// Delete expired notes.
		if(deletionCount > 0) {
			for(int i = 0; i < ds->notificationCount-deletionCount; i++) {
				ds->notificationStack[i] = ds->notificationStack[i+deletionCount];
				ds->notificationTimes[i] = ds->notificationTimes[i+deletionCount];
			}
			ds->notificationCount -= deletionCount;
		}

		// Draw notes.
		int fontSize = ds->fontHeight;
		Font* font = getFont(FONT_CALIBRI, fontSize);
		Vec4 color = vec4(1,0.5f,0,1);

		float y = -fontSize/2;
		for(int i = 0; i < ds->notificationCount; i++) {
			char* note = ds->notificationStack[i];
			dcText(note, font, vec2(ws->currentRes.w/2, y), color, vec2i(0,0), 0, 2);
			y -= fontSize;
		}
	}

	if(ds->showHud) {
		int fontSize = ds->fontHeight*1.1f;
		int pi = 0;
		// Vec4 c = vec4(1.0f,0.2f,0.0f,1);
		Vec4 c = vec4(1.0f,0.4f,0.0f,1);
		Vec4 c2 = vec4(0,0,0,1);
		Font* font = getFont(FONT_CONSOLAS, fontSize);
		int sh = 1;
		Vec2 offset = vec2(6,6);
		Vec2i ali = vec2i(1,1);

		Vec2 tp = vec2(ad->wSettings.currentRes.x, 0) - offset;

		static f64 timer = 0;
		static int fpsCounter = 0;
		static int fps = 0;
		timer += ds->dt;
		fpsCounter++;
		if(timer >= 1.0f) {
			fps = fpsCounter;
			fpsCounter = 0;
			timer = 0;
		}

		dcText(fillString("Fps  : %i", fps), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Pos  : (%f,%f,%f)", PVEC3(ad->activeCam.pos)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Pos  : (%f,%f,%f)", PVEC3(ad->selectedBlock)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Look : (%f,%f,%f)", PVEC3(ad->activeCam.look)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Up   : (%f,%f,%f)", PVEC3(ad->activeCam.up)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Right: (%f,%f,%f)", PVEC3(ad->activeCam.right)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Rot  : (%f,%f)",    PVEC2(ad->player->rot)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Vec  : (%f,%f,%f)", PVEC3(ad->player->vel)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Acc  : (%f,%f,%f)", PVEC3(ad->player->acc)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Draws: (%i)", 	   ad->voxelDrawCount), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Quads: (%i)", 	   ad->voxelTriangleCount), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("BufferIndex: %i",    ds->timer->bufferIndex), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("LastBufferIndex: %i",ds->lastBufferIndex), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;


		for(int i = 0; i < ds->infoStackCount; i++) {
			dcText(fillString("%s", ds->infoStack[i]), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		}
		ds->infoStackCount = 0;
	}


	if(*isRunning == false) {
		guiSave(ds->gui, 2, 0);
		if(globalDebugState->gui2) guiSave(globalDebugState->gui2, 2, 3);
	}

	// Update debugTime every second.
	static f64 tempTime = 0;
	tempTime += ds->dt;
	if(tempTime >= 1) {
		ds->debugTime = timerStop(&ds->tempTimer);
		tempTime = 0;
	}
}

#pragma optimize( "", on ) 
