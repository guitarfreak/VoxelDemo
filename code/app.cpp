/*
//-----------------------------------------
//				WHAT TO DO
//-----------------------------------------
- Joysticks, Xinput-DirectInput
- Data Package - Streaming
- create simpler windows.h
- remove c runtime library
- ballistic motion on jumping
- round collision
- screen space reflections

- implement sun and clouds that block beams of light
- glowstone emmitting light
- stb_voxel push alpha test in voxel vertex shader
- rocket launcher
- antialiased pixel graphics with neighbour sampling 
- level of detail for world gen back row, project to cubemap.
  
- put in spaces for fillString
- simplex noise instead of perlin noise

- simd voxel generation
- simd on vectors

- 3d animation system. (Search Opengl vertex skinning.)
- Sound perturbation.
- When switching between text editor and debugger, synchronize open files.
- Entity introspection in gui.
- Shadow mapping, start with cloud texture projection.

- atomicAdd sections are not thread proof. 
  We could have multiple equal jobs in thread queue.
- Simulate earth curvature in shader.
- Make horizon work. The cubemap solution is not good.
- DirectX.
- Fix selection algorithm.
- Cubemap not seemless.
- Sound starts to glitch when under 30 hz because audio buffer is 2*framrate.
- Remove command lists.
- Creative mode.
- Fix texture reloading.

//-------------------------------------
//               BUGS
//-------------------------------------
- Release build for appMain takes forever.

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

#if USE_DIRECT3D

#include <D3D11.h>
#include <D3DCommon.h>
// #include <D3Dcompiler.h>
// #include <D3DX11async.h>
typedef ID3D10Blob ID3DBlob;

#endif USE_DIRECT3D

// Stb.

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "external\stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #define STBI_ONLY_PNG
// #include "external\stb_image_write.h"

#define STB_VOXEL_RENDER_IMPLEMENTATION
// #define STBVOX_CONFIG_LIGHTING_SIMPLE
#define STBVOX_CONFIG_FOG_SMOOTHSTEP
// #define STBVOX_CONFIG_MODE 0
#define STBVOX_CONFIG_MODE 1
#include "external\stb_voxel_render.h"

#define STB_VORBIS_NO_PUSHDATA_API
#include "external\stb_vorbis.c"

//

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_PARAMETER_TAGS_H
#include FT_MODULE_H

//

struct ThreadQueue;
struct GraphicsState;
struct AudioState;
struct DrawCommandList;
struct MemoryBlock;
struct DebugState;
struct Timer;
ThreadQueue*     theThreadQueue;
GraphicsState*   theGraphicsState;
AudioState*      theAudioState;
DrawCommandList* theCommandList;
MemoryBlock*     theMemory;
DebugState*      theDebugState;
Timer*           theTimer;

// Internal.

#include "rt_types.cpp"
#include "rt_timer.cpp"
#include "rt_misc.cpp"
#include "rt_math.cpp"
#include "rt_memory.cpp"
#include "rt_appMemory.cpp"
#include "rt_misc2.cpp"
#include "rt_hotload.cpp"
#include "rt_misc_win32.cpp"
#include "rt_platformWin32.cpp"

#include "rendering.h"
#include "font.h"

#include "openglDefines.cpp"
#include "userSettings.cpp"

#include "audio.cpp"
#include "rendering.cpp"
#include "font.cpp"
#include "drawCommandList.cpp"
#include "gui.cpp"

#include "entity.cpp"
#include "voxel.cpp"
#include "menu.cpp"
#include "inventory.cpp"




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
	float mouseSensitivity;

	// 

	bool debugMouse;
	bool captureMouseKeepCenter;
	bool captureMouse;
	bool fpsMode;
	bool fpsModeFixed;
	bool lostFocusWhileCaptured;

	Camera activeCam;

	Vec2i cur3dBufferRes;
	int msaaSamples;
	float resolutionScale;

	float aspectRatio;
	int fieldOfView;
	float nearPlane;
	float farPlane;

	// Game.

	int gameMode;
	MainMenu menu;

	bool loading;
	bool newGame;
	bool saveGame;
	float startFade;

	GameSettings settings;

	float volumeFootsteps;
	float volumeGeneral;
	float volumeMenu;

	//

	EntityList entityList;
	Entity* player;
	Entity* cameraEntity;

	//

	Particle* particleLists[50];
	bool particleListUsage[50];
	int particleListsSize;

	//

	bool playerMode;
	bool pickMode;
	int selectionRadius;

	Inventory inventory;

	//

	float bombFireInterval;
	bool bombButtonDown;
	float bombSpawnTimer;

	bool blockSelected;
	Vec3 selectedBlock;
	Vec3 selectedBlockFaceDir;

	bool firstWalk;
	float footstepSoundValue;
	int lastFootstepSoundId;

	Vec3i breakingBlock;
	float breakingBlockTime;
	float hardnessModifier;
	float breakingBlockTimeSound;
	
	//

	VoxelWorldSettings voxelSettings;
	VoxelData voxelData;
	Vec2i chunkOffset;
	Vec3i voxelOffset;

	// int skyBoxId;
	char* skybox;

	bool reloadWorld;

	int voxelDrawCount;
	int voxelTriangleCount;

	// Particles.

	// GLuint testBufferId;
	// char* testBuffer;
	// int testBufferSize;
};

#include "debug.cpp"



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
	theMemory = &gMemory;

	DebugState* ds = (DebugState*)getBaseMemoryArray(gMemory.pDebugMemory);
	AppData* ad = (AppData*)getBaseExtendibleMemoryArray(gMemory.pMemory);
	GraphicsState* gs = &ad->graphicsState;

	Input* input = &ad->input;
	SystemData* sd = &ad->systemData;
	HWND windowHandle = sd->windowHandle;
	WindowSettings* ws = &ad->wSettings;

	theThreadQueue = threadQueue;
	theGraphicsState = &ad->graphicsState;
	theAudioState = &ad->audioState;
	theDebugState = ds;
	theTimer = ds->timer;

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
		theTimer = ds->timer;
		int timerSlots = 50000;
		ds->timer->bufferSize = timerSlots;
		ds->timer->timerBuffer = (TimerSlot*)getPMemoryDebug(sizeof(TimerSlot) * timerSlots);

		ds->savedBufferMax = 20000;
		ds->savedBufferIndex = 0;
		ds->savedBufferCount = 0;
		ds->savedBuffer = (GraphSlot*)getPMemoryDebug(sizeof(GraphSlot) * ds->savedBufferMax);


		ds->gui = getPStructDebug(Gui);
		ds->gui->init(rectCenDim(vec2(1300,1), vec2(300, ws->currentRes.h)), 0);

		ds->gui2 = getPStructDebug(Gui);
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
		
		int windowStyle = WS_OVERLAPPEDWINDOW;
		// int windowStyle = WS_OVERLAPPEDWINDOW & ~WS_SYSMENU;
		initSystem(sd, ws, windowsData, vec2i(1920*0.85f, 1080*0.85f), windowStyle, 1);

		windowHandle = sd->windowHandle;
		SetWindowText(windowHandle, APP_NAME);

		loadFunctions();

		#if 1
			wglSwapIntervalEXT(1);
			ws->vsync = true;
			ws->frameRate = ws->refreshRate;
		#else 
			wglSwapIntervalEXT(0);
			ws->vsync = false;
			ws->frameRate = 200;
		#endif

		initInput(&ad->input);
		sd->input = &ad->input;

		#ifndef SHIPPING_MODE
		if(!IsDebuggerPresent()) {
			makeWindowTopmost(sd);
		}
		#endif

		ws->lastMousePosition = {0,0};

		//
		// Init Folder Handles.
		//

		initWatchFolders(sd->folderHandles, ds->assets, &ds->assetCount);

		//
		// Setup Textures.
		//

		gs->texturesCount = 0;
		gs->texturesCountMax = 100;
		gs->textures = getPArray(Texture, gs->texturesCountMax);

		{
			RecursiveFolderSearchData fd;
			recursiveFolderSearchStart(&fd, App_Texture_Folder);
			while(recursiveFolderSearchNext(&fd)) {
				Texture tex;
				tex.name = getPStringCpy(fd.fileName);
				tex.file = getPStringCpy(fd.filePath);

				if(strFind(fd.fileName, "skyboxes\\") != -1) {
					loadCubeMapFromFile(&tex, fd.filePath, 5, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE);
				} else {
					loadTextureFromFile(&tex, fd.filePath, -1, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE);
				}

				gs->textures[gs->texturesCount++] = tex;
			}

			{
				Texture tex;
				tex.name = getPStringCpy("voxelTextures");
				Texture tex2;
				tex2.name = getPStringCpy("voxelTextures2");

				char* voxelTexturesFolder = "minecraft\\";

				char* voxelTexturesPath = fillString("%s%s", App_Texture_Folder, voxelTexturesFolder);
				loadVoxelTextures(&tex, &tex2, voxelTexturesPath, ad->voxelSettings.waterAlpha, INTERNAL_TEXTURE_FORMAT);

				gs->textures[gs->texturesCount++] = tex;
				gs->textures[gs->texturesCount++] = tex2;
			}

			gs->textureWhite = getTexture("misc\\white.png");
		}

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

		// ad->testBufferSize = megaBytes(10);
		// ad->testBuffer = getPArray(char, ad->testBufferSize);
		// glCreateBuffers(1, &ad->testBufferId);
		// glNamedBufferData(ad->testBufferId, ad->testBufferSize, ad->testBuffer, GL_STREAM_DRAW);

		// 
		// Samplers.
		//

		gs->samplers[SAMPLER_NORMAL] = createSampler(16.0f, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);

		gs->samplers[SAMPLER_VOXEL_1] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR);
		gs->samplers[SAMPLER_VOXEL_2] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR);
		gs->samplers[SAMPLER_VOXEL_3] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);

		//
		//
		//

		ad->fieldOfView = 60;
		ad->msaaSamples = 4;
		ad->resolutionScale = 1;
		ad->nearPlane = 0.1f;
		ad->farPlane = 3000;
		ad->dt = 1/(float)ws->frameRate;

		//
		// FrameBuffers.
		//

		{
			gs->frameBufferCountMax = 10;
			gs->frameBufferCount = 0;
			gs->frameBuffers = getPArray(FrameBuffer, gs->frameBufferCountMax);

			FrameBuffer* fb;

			fb = addFrameBuffer("3dMsaa");
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0, ad->msaaSamples);
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_DEPTH_STENCIL, GL_DEPTH_STENCIL, 0, 0, ad->msaaSamples);

			fb = addFrameBuffer("3dNoMsaa");
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, 0, 0);

			fb = addFrameBuffer("Reflection");
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, 0, 0);

			fb = addFrameBuffer("2d");
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);

			fb = addFrameBuffer("DebugMsaa");
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0, ad->msaaSamples);

			fb = addFrameBuffer("DebugNoMsaa");
			attachToFrameBuffer(fb, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);

			ad->updateFrameBuffers = true;
		}

		//
		// @AudioInit.
		//

		AudioState* as = &ad->audioState;
		(*as) = {};
		as->masterVolume = 1.0f;

		audioDeviceInit(as, ws->frameRate);

		as->fileCountMax = 100;
		as->files = getPArray(Audio, as->fileCountMax);

		{
			RecursiveFolderSearchData fd;
			recursiveFolderSearchStart(&fd, App_Audio_Folder);
			while(recursiveFolderSearchNext(&fd)) {
				addAudio(as, fd.filePath, fd.fileName);
			}
		}

		//
		//
		//

		theGraphicsState->fontFolders[theGraphicsState->fontFolderCount++] = getPStringCpy(App_Font_Folder);
		char* windowsFontFolder = fillString("%s%s", getenv(Windows_Font_Path_Variable), Windows_Font_Folder);
		theGraphicsState->fontFolders[theGraphicsState->fontFolderCount++] = getPStringCpy(windowsFontFolder);

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

				updateResolution(windowHandle, ws);

				appSessionSettings = at;
			}
		}

		pcg32_srandom(0, __rdtsc());

		//
		// @AppInit.
		//

		// Vec2 asdf = vec2(1,1);
		// pushUniformX(0, 0, "asdf", &asdf);
		// int sd = 234;;
		// pushUniformX(0, 0, "asdf", &sd);
		// test(2);

		timerInit(&ds->swapTimer);
		timerInit(&ds->frameTimer);
		timerInit(&ds->tempTimer);

		as->masterVolume = 0.4f;

		ad->gameMode = GAME_MODE_LOAD;
		// ad->gameMode = GAME_MODE_TEST;
		// ad->newGame = false;
		ad->menu.activeId = 0;

		#if SHIPPING_MODE
		ad->captureMouse = true;
		#else 
		ad->captureMouse = false;
		ad->debugMouse = true;
		#endif

		folderExistsCreate(SAVES_FOLDER);	

		ad->volumeFootsteps = 0.2f;
		ad->volumeGeneral = 0.5f;
		ad->volumeMenu = 0.7f;

		ad->mouseSensitivity = 0.2f;

		//

		ad->particleListsSize = 100;
		for(int i = 0; i < arrayCount(ad->particleLists); i++) {
			ad->particleLists[i] = getPArray(Particle, ad->particleListsSize);
			ad->particleListUsage[i] = false;
		}

		// Entity.

		ad->playerMode = true;
		ad->pickMode = true;

		ad->selectionRadius = 4;

		ad->entityList.size = 1000;
		ad->entityList.e = (Entity*)getPMemory(sizeof(Entity)*ad->entityList.size);
		for(int i = 0; i < ad->entityList.size; i++) ad->entityList.e[i].init = false;

		// Voxel.

		ad->chunkOffset = vec2i(0,0);

		// ad->skyBoxId = CUBEMAP_5;
		ad->skybox = "skyboxes\\skybox1.png";
		ad->bombFireInterval = 0.1f;
		ad->bombButtonDown = false;

		ad->hardnessModifier = 0.2f;

		// Hashes and thread data.
		{
			VoxelData* vd = &ad->voxelData;
			vd->size = 10000;
			vd->voxelHashSize = vd->size * 10;
			vd->voxelHash = getPArray(VoxelArray, vd->voxelHashSize);
			vd->voxels.reserve(vd->size);

			float hashSize = (vd->voxelHashSize * sizeof(VoxelArray)) / (float)(1024*1024);
			float voxelsSize = (vd->size * sizeof(VoxelMesh)) / (float)(1024*1024);
			float totalSize = hashSize + voxelsSize;

			voxelWorldSettingsInit(&ad->voxelSettings);
		}

		// Trees.
		{
			int treeRadius = 10;
			ad->voxelSettings.treeNoise = (bool*)getPMemory(VOXEL_X*VOXEL_Y);
			zeroMemory(ad->voxelSettings.treeNoise, VOXEL_X*VOXEL_Y);

			Rect bounds = rect(0, 0, 64, 64);
			Vec2* noiseSamples;
			int noiseSamplesSize = blueNoise(bounds, treeRadius, &noiseSamples);
			for(int i = 0; i < noiseSamplesSize; i++) {
				Vec2 s = noiseSamples[i];
				Vec2i index = vec2i(s);
				ad->voxelSettings.treeNoise[index.y*VOXEL_X + index.x] = 1;
			}
			free(noiseSamples);
		}

		// Load game settings.
		{
			// Init default.
			if(!fileExists(Game_Settings_File)) {
				GameSettings settings = {};

				settings.fullscreen = true;
				settings.vsync = true;
				settings.resolutionScale = 1;
				settings.volume = 0.5f;
				settings.mouseSensitivity = 0.2f;
				settings.fieldOfView = 60;
				settings.viewDistance = 5;

				writeDataToFile((char*)&settings, sizeof(GameSettings), Game_Settings_File);
			}

			// Load Gamesettings.
			{
				GameSettings settings = {};

				readDataFile((char*)&settings, Game_Settings_File);

				if(settings.fullscreen) setWindowMode(windowHandle, ws, WINDOW_MODE_FULLBORDERLESS);
				else setWindowMode(windowHandle, ws, WINDOW_MODE_WINDOWED);

				ws->vsync = settings.vsync;
				ad->resolutionScale = settings.resolutionScale;
				ad->audioState.masterVolume = settings.volume;
				ad->mouseSensitivity = settings.mouseSensitivity;
				ad->fieldOfView = settings.fieldOfView;
				ad->voxelSettings.viewDistance = settings.viewDistance;
			}
		}

		#if USE_DIRECT3D
		{
			//Set up DX swap chain
			//--------------------------------------------------------------
			 
			DXGI_SWAP_CHAIN_DESC swapChainDesc;
			ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));
			 
			//set buffer dimensions and format
			swapChainDesc.BufferCount = 2;
			swapChainDesc.BufferDesc.Width = 1000;
			swapChainDesc.BufferDesc.Height = 1000;
			swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
			swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;;
			swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
			swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
			swapChainDesc.SampleDesc.Quality = 0;
			swapChainDesc.SampleDesc.Count = 1;
			swapChainDesc.OutputWindow = windowHandle;
			swapChainDesc.Windowed = true;
			 
			//Create the D3D device
			//--------------------------------------------------------------
			 
			D3D_FEATURE_LEVEL featureLevels[] = {D3D_FEATURE_LEVEL_11_0};

			IDXGISwapChain* swapChain;
			ID3D11Device* d3dDevice;
			if ( FAILED( D3D11CreateDeviceAndSwapChain( NULL,
			                                            D3D_DRIVER_TYPE_HARDWARE,
			                                            NULL,
			                                            D3D11_CREATE_DEVICE_DEBUG, // D3D11_CREATE_DEVICE_DEBUG,
			                                            featureLevels,
			                                            arrayCount(featureLevels),
			                                            D3D11_SDK_VERSION,
			                                            &swapChainDesc,
			                                            &swapChain,
			                                            &d3dDevice,
			                                            0,
			                                            0) ) ) 
				printf("D3D device creation failed");


			//Create render target view
			//--------------------------------------------------------------
			 
			//try to get the back buffer
			ID3D11Texture2D* backBuffer;
			 
			if ( FAILED( swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)(&backBuffer)) ) ) 
				printf("Could not get back buffer");
			 
			ID3D11RenderTargetView* renderTargetView;

			//try to create render target view
			if ( FAILED( d3dDevice->CreateRenderTargetView(backBuffer, NULL, &renderTargetView) ) ) 
				printf("Could not create render target view");
			 
			//release the back buffer
			backBuffer->Release();
			 
			ID3D11DeviceContext* d3dDeviceContext;
			d3dDevice->GetImmediateContext(&d3dDeviceContext);

			//set the render target
			d3dDeviceContext->OMSetRenderTargets(1, &renderTargetView, NULL);

			//create viewport structure
			
			D3D11_VIEWPORT viewPort;
			viewPort.Width = 1000;
			viewPort.Height = 1000;
			viewPort.MinDepth = 0.0f;
			viewPort.MaxDepth = 1.0f;
			viewPort.TopLeftX = 0;
			viewPort.TopLeftY = 0;
			 
			//set the viewport

			d3dDeviceContext->RSSetViewports(1, &viewPort);


			sd->swapChain = swapChain;
			sd->d3dDevice = d3dDevice;
			sd->d3dDeviceContext = d3dDeviceContext;
			sd->renderTargetView = renderTargetView;

			// 

		    ID3DBlob* blobError;

		    ID3DBlob* vertexShaderBlob;
		    uint result = D3D10CompileShader( d3dShader, strlen( d3dShader ), NULL, NULL, NULL, "vertexShader", "vs_4_0", 0, &vertexShaderBlob, &blobError );
		    {
		    	if (blobError != nullptr)
		    		printf((char*)blobError->GetBufferPointer());
		    	if (blobError) blobError->Release();
		    }

		    ID3D11VertexShader* vertexShader;
		    d3dDevice->CreateVertexShader(vertexShaderBlob->GetBufferPointer(), vertexShaderBlob->GetBufferSize(), 0, &vertexShader);
		    d3dDeviceContext->VSSetShader(vertexShader, 0, 0);


		    ID3DBlob* pixelShaderBlob;
		    result = D3D10CompileShader( d3dShader, strlen( d3dShader ), NULL, NULL, NULL, "pixelShader", "ps_4_0", 0, &pixelShaderBlob, &blobError );
		    {
		    	if (blobError != nullptr)
		    		printf((char*)blobError->GetBufferPointer());
		    	if (blobError) blobError->Release();
		    }

		    ID3D11PixelShader* pixelShader;
		    d3dDevice->CreatePixelShader(pixelShaderBlob->GetBufferPointer(), pixelShaderBlob->GetBufferSize(), 0, &pixelShader);
		    d3dDeviceContext->PSSetShader(pixelShader, 0, 0);

			//

			D3D11_INPUT_ELEMENT_DESC layout[] = {
			    { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			    { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT,0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			    { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,0, 28, D3D11_INPUT_PER_VERTEX_DATA, 0 }
			};

			ID3D11InputLayout* vertexLayout;
			if ( FAILED( d3dDevice->CreateInputLayout( layout,
			                                            arrayCount(layout),
			                                            vertexShaderBlob->GetBufferPointer(),
			                                            vertexShaderBlob->GetBufferSize(),
			                                            &vertexLayout ) ) ) 
				printf("Could not create Input Layout!");
			 
			// Set the input layout
			d3dDeviceContext->IASetInputLayout( vertexLayout );

			//

			//create vertex buffer (space for 100 vertices)
			//---------------------------------------------
			 
			UINT numVertices = 100;
			 
			D3D11_BUFFER_DESC bd;
			bd.Usage = D3D11_USAGE_DYNAMIC;
			bd.ByteWidth = sizeof( DVertex ) * numVertices; //total size of buffer in bytes
			bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bd.MiscFlags = 0;

			ID3D11Buffer* vertexBuffer;
			if ( FAILED( d3dDevice->CreateBuffer( &bd, NULL, &vertexBuffer ) ) ) 
				printf("Could not create vertex buffer!");
			 
			// Set vertex buffer
			UINT stride = sizeof( DVertex );
			UINT offset = 0;
			d3dDeviceContext->IASetVertexBuffers( 0, 1, &vertexBuffer, &stride, &offset );

			//

			D3D11_RASTERIZER_DESC rasterizerState;
			rasterizerState.CullMode = D3D11_CULL_NONE;
			rasterizerState.FillMode = D3D11_FILL_SOLID;
			rasterizerState.FrontCounterClockwise = true;
			rasterizerState.DepthBias = false;
			rasterizerState.DepthBiasClamp = 0;
			rasterizerState.SlopeScaledDepthBias = 0;
			rasterizerState.DepthClipEnable = true;
			rasterizerState.ScissorEnable = false;
			rasterizerState.MultisampleEnable = false;
			rasterizerState.AntialiasedLineEnable = true;
			 
			ID3D11RasterizerState* pRS;
			d3dDevice->CreateRasterizerState( &rasterizerState, &pRS);
			d3dDeviceContext->RSSetState(pRS);

			//

			//fill vertex buffer with vertices
			numVertices = 3;
			DVertex* v = NULL;
			 
			//lock vertex buffer for CPU use
			d3dDeviceContext->Map(vertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, (D3D11_MAPPED_SUBRESOURCE*) &v);

			v[0] = { vec3(-0.5f,-0.5f,0), vec4(1,0,0,1), vec2(0,0)};
			v[1] = { vec3(-0.5f, 0.5f,0), vec4(0,1,0,1), vec2(1,0)};
			v[2] = { vec3( 0.5f,-0.5f,0), vec4(1,1,1,1), vec2(0,1)};
			v[3] = { vec3( 0.5f, 0.5f,0), vec4(0,0,1,1), vec2(1,1)};

			d3dDeviceContext->Unmap(vertexBuffer, 0);

			//

			// Set primitive topology
			d3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

			//

			ID3D11Texture2D* texture;
			ID3D11ShaderResourceView* textureView;
			// if(false)
			{
				DXGI_SAMPLE_DESC sampleDesc;
				sampleDesc.Count = 1;
				sampleDesc.Quality = 0;

				Vec2i texDim = vec2i(2,2);

				D3D11_TEXTURE2D_DESC texDesc;
				texDesc.Width = texDim.w;
				texDesc.Height = texDim.h;
				texDesc.MipLevels = 1;
				texDesc.ArraySize = 1;
				// texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
				texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
				// texDesc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
				// texDesc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
				texDesc.SampleDesc = sampleDesc;
				texDesc.Usage = D3D11_USAGE_DEFAULT;
				texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
				texDesc.CPUAccessFlags = 0;
				texDesc.MiscFlags = 0;

				// uint texData[] = {255,0,0,255, 0,255,0,255, 
				//                   0,0,255,255, 0,0,255,255,};

				uchar texData[] = {255,0,0,255, 0,255,0,255, 
				                  0,0,255,255, 255,255,255,255,};

				// uchar texData[] = {255,255,255,255, 255,255,255,255, 
				//                    255,255,255,255, 255,255,255,255,};

				// Vec4 texData[] = {vec4(1,0,0,1),vec4(0,1,0,1),
				// 				  vec4(0,0,1,1),vec4(1,1,1,1)};

                int typeSize = sizeof(uint);

				D3D11_SUBRESOURCE_DATA texResourceData;
				texResourceData.pSysMem = texData;
				texResourceData.SysMemPitch = texDim.w*typeSize;
				texResourceData.SysMemSlicePitch = texDim.w*texDim.h*typeSize;

				uint result = d3dDevice->CreateTexture2D(&texDesc, &texResourceData, &texture);

				D3D11_TEX2D_SRV srv;
				srv.MostDetailedMip = 0;
				srv.MipLevels = -1;
				D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
				// viewDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
				// viewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
				// viewDesc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
				viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
				viewDesc.Texture2D = srv;

				d3dDevice->CreateShaderResourceView(texture, &viewDesc, &textureView);
			}

			ID3D11SamplerState* sampler;
			{
				D3D11_SAMPLER_DESC samplerDesc;
				samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
				// samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
				samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
				samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
				samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
				samplerDesc.MipLODBias = 0;
				samplerDesc.MaxAnisotropy = 1;
				samplerDesc.ComparisonFunc = D3D11_COMPARISON_LESS_EQUAL;
				samplerDesc.BorderColor[0] = 0;
				samplerDesc.BorderColor[1] = 0;
				samplerDesc.BorderColor[2] = 0;
				samplerDesc.BorderColor[3] = 0;
				samplerDesc.MinLOD = 0;
				samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

				d3dDevice->CreateSamplerState(&samplerDesc, &sampler);
			}

			d3dDeviceContext->PSSetShaderResources(0, 1, &textureView);
			d3dDeviceContext->PSSetSamplers(0, 1, &sampler);

			// //create world matrix
			// static float r;
			// D3DXMATRIX w;
			// D3DXMatrixIdentity(&w);
			// D3DXMatrixRotationY(&w, r);
			// r += 0.001f;

			// //set effect matrices
			// pWorldMatrixEffectVariable->SetMatrix(w);
			// pViewMatrixEffectVariable->SetMatrix(viewMatrix);
			// pProjectionMatrixEffectVariable->SetMatrix(projectionMatrix);

			//

				//get technique desc
				// D3D11_TECHNIQUE_DESC techDesc;
				// pBasicTechnique->GetDesc( &techDesc );
				 
				// pWorldMatrixEffectVariable->SetMatrix(w);
				// pViewMatrixEffectVariable->SetMatrix(viewMatrix);
				// pProjectionMatrixEffectVariable->SetMatrix(projectionMatrix);
				 
				// for( UINT p = 0; p < techDesc.Passes; ++p ) {
				//       //apply technique
				//       pBasicTechnique->GetPassByIndex( p )->Apply( 0 );
				 
				//       //draw
				//       d3dDevice->Draw( numVertices, 0 );
				// }
		}
		#endif
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
		for(int i = 0; i < arrayCount(theGraphicsState->fonts); i++) {
			for(int j = 0; j < arrayCount(theGraphicsState->fonts[0]); j++) {
				Font* font = &theGraphicsState->fonts[i][j];
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
		}
	}

	clearTMemory();

	// Allocate drawCommandlist.

	int clSize = kiloBytes(1000);
	drawCommandListInit(&ad->commandList3d, (char*)getTMemory(clSize), clSize);
	drawCommandListInit(&ad->commandList2d, (char*)getTMemory(clSize), clSize);
	theCommandList = &ad->commandList3d;

	// Hotload changed files.

	reloadChangedFiles(sd->folderHandles, ds->assets, ds->assetCount);

	// Update input.
	{
		TIMER_BLOCK_NAMED("Input");

		if(ws->vsync) wglSwapIntervalEXT(1);
		else wglSwapIntervalEXT(0);

		inputPrepare(input);
		SwitchToFiber(sd->messageFiber);

		if(ad->input.closeWindow) *isRunning = false;

		// ad->input = *ds->input;
		*ds->input = ad->input;

		if(ds->console.isActive) {
			memSet(ad->input.keysPressed, 0, sizeof(ad->input.keysPressed));
			memSet(ad->input.keysDown, 0, sizeof(ad->input.keysDown));
		}

		ad->dt = ds->dt;
		ad->time = ds->time;

		ad->frameCount++;

		sd->fontHeight = getSystemFontHeight(sd->windowHandle);
		ds->fontHeight = roundInt(ds->fontScale*sd->fontHeight);
		ds->fontHeightScaled = roundInt(ds->fontHeight * ws->windowScale);
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
		
		ad->cur3dBufferRes = ws->currentRes;
		if(ad->resolutionScale < 1.0f) {
			ad->cur3dBufferRes = ad->cur3dBufferRes * ad->resolutionScale;
		}

		Vec2i s = ad->cur3dBufferRes;
		Vec2 reflectionRes = vec2(s);

		setDimForFrameBufferAttachmentsAndUpdate("3dMsaa", s.w, s.h);
		setDimForFrameBufferAttachmentsAndUpdate("3dNoMsaa", s.w, s.h);
		setDimForFrameBufferAttachmentsAndUpdate("Reflection", reflectionRes.w, reflectionRes.h);
		setDimForFrameBufferAttachmentsAndUpdate("2d", ws->currentRes.w, ws->currentRes.h);
		setDimForFrameBufferAttachmentsAndUpdate("DebugMsaa", ws->currentRes.w, ws->currentRes.h);
		setDimForFrameBufferAttachmentsAndUpdate("DebugNoMsaa", ws->currentRes.w, ws->currentRes.h);
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
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClearColor(0,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);

		int bits = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;

		clearFrameBuffer("3dMsaa",      vec4(1), bits);
		clearFrameBuffer("3dNoMsaa",    vec4(1), bits);
		clearFrameBuffer("Reflection",  vec4(1), bits);
		clearFrameBuffer("2d",          vec4(0), bits);
		clearFrameBuffer("DebugMsaa",   vec4(0), bits);
		clearFrameBuffer("DebugNoMsaa", vec4(0), bits);
	}

	// Setup opengl.
	{
		// glDepthRange(-1.0,1.0);
		glFrontFace(GL_CW);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glDisable(GL_SCISSOR_TEST);

		glEnable(GL_TEXTURE_2D);
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

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

		if(input->keysPressed[KEYCODE_F3]) {
			ad->debugMouse = !ad->debugMouse;
		}

		bool showMouse = ad->debugMouse || (ad->inventory.show && ad->gameMode == GAME_MODE_MAIN);

		if(!ad->captureMouse) {
			if(!showMouse) {
				input->mouseButtonPressed[1] = false;
				ad->captureMouse = true;

				GetCursorPos(&ws->lastMousePosition);
			}

		} else {
			if(showMouse) {
				ad->captureMouse = false;

				if(ws->lastMousePosition.x == 0 && ws->lastMousePosition.y == 0) {
					int w,h;
					Vec2i wPos;
					getWindowProperties(windowHandle, &w, &h, 0, 0, &wPos.x, &wPos.y);
					ws->lastMousePosition.x = wPos.x + w/2;
					ws->lastMousePosition.y = wPos.y + h/2;
				}

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

	if(ad->gameMode == GAME_MODE_MENU) {
		theCommandList = &ad->commandList2d;

		Rect sr = getScreenRect(ws);
		Vec2 top = rectT(sr);
		float rHeight = rectH(sr);
		float rWidth = rectW(sr);

		int titleFontHeight = ds->fontHeightScaled * 6.0f;
		int optionFontHeight = titleFontHeight * 0.45f;
		Font* titleFont = getFont("Merriweather-Regular.ttf", titleFontHeight);
		Font* font = getFont("LiberationSans-Regular.ttf", optionFontHeight);

		Vec4 cBackground = vec4(hslToRgbf(0.63f,0.3f,0.13f),1);
		Vec4 cTitle = vec4(1,1);
		Vec4 cTitleShadow = vec4(0,0,0,1);
		Vec4 cOption = vec4(0.5f,1);
		Vec4 cOptionActive = vec4(0.9f,1);
		Vec4 cOptionShadow1 = vec4(0,1);
		Vec4 cOptionShadow2 = vec4(hslToRgbf(0.0f,0.5f,0.5f), 1);
		Vec4 cOptionShadow = vec4(0,1);

		float titleShadowSize = titleFontHeight * 0.07f;
		float optionShadowSize = optionFontHeight * 0.07f;

		float buttonAnimSpeed = 4;

		float optionOffset = optionFontHeight*1.2f;
		float settingsOffset = 0.15f;


		MainMenu* menu = &ad->menu;
		menuSetInput(menu, input);
		menu->volume = ad->volumeMenu;

		bool selectionChange = false;

		if(input->keysPressed[KEYCODE_DOWN]) {
			addTrack("ui\\select.ogg", menu->volume, true);
			menu->activeId++;
			selectionChange = true;
		}
		if(input->keysPressed[KEYCODE_UP]) {
			addTrack("ui\\select.ogg", menu->volume, true);
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
		}

		dcRect(sr, cBackground);

		if(menu->screen == MENU_SCREEN_MAIN) {

			Vec2 p = top - vec2(0, rHeight*0.2f);
			dcText("Voxel Demo", titleFont, p, cTitle, vec2i(0,0), 0, titleShadowSize, cTitleShadow);

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
				addTrack("ui\\start.ogg");
				ad->gameMode = GAME_MODE_LOAD;
				ad->newGame = true;
			}

			p.y -= optionOffset;
			if(menuOption(menu, "Settings", p, vec2i(0,0))) {
				addTrack("ui\\menuPush.ogg", menu->volume, true);

				menu->screen = MENU_SCREEN_SETTINGS;
				menu->activeId = 0;
			}

			p.y -= optionOffset;
			if(menuOption(menu, "Exit", p, vec2i(0,0))) {
				*isRunning = false;
			}

		} else if(menu->screen == MENU_SCREEN_SETTINGS) {

			Vec2 p = top - vec2(0, rHeight*0.2f);
			Vec2 pos;
			float leftX = rWidth * settingsOffset;
			float rightX = rWidth * (1-settingsOffset);

			dcText("Settings", titleFont, p, cTitle, vec2i(0,0), 0, titleShadowSize, cTitleShadow);

			int optionCount = 7;
			p.y = rectCen(sr).y + ((optionCount-1)*optionOffset)/2;


			// List settings.

			GameSettings* settings = &ad->settings;
			VoxelWorldSettings* voxelSettings = &ad->voxelSettings;



			p.x = leftX;
			menuOption(menu, "Fullscreen", p, vec2i(-1,0));
			
			bool tempBool = ws->fullscreen;
			if(menuOptionBool(menu, vec2(rightX, p.y), &tempBool)) {
				if(ws->fullscreen) setWindowMode(windowHandle, ws, WINDOW_MODE_WINDOWED);
				else setWindowMode(windowHandle, ws, WINDOW_MODE_FULLBORDERLESS);
			}

			//

			p.y -= optionOffset; p.x = leftX;
			menuOption(menu, "VSync", p, vec2i(-1,0));

			menuOptionBool(menu, vec2(rightX, p.y), &ws->vsync);

			//

			p.y -= optionOffset; p.x = leftX;
			menuOption(menu, "Resolution Scale", p, vec2i(-1,0));

			if(menuOptionSliderFloat(menu, vec2(rightX, p.y), &ad->resolutionScale, 0.2f, 1, 0.1f, 1)) {
				ad->updateFrameBuffers = true;
			}

			//

			p.y -= optionOffset; p.x = leftX;
			menuOption(menu, "Volume", p, vec2i(-1,0));

			menuOptionSliderFloat(menu, vec2(rightX, p.y), &ad->audioState.masterVolume, 0.0f, 1, 0.1f, 1);

			//

			p.y -= optionOffset; p.x = leftX;
			menuOption(menu, "Mouse Sensitivity", p, vec2i(-1,0));

			menuOptionSliderFloat(menu, vec2(rightX, p.y), &ad->mouseSensitivity, 0.01f, 2, 0.01f, 2);

			//

			p.y -= optionOffset; p.x = leftX;
			menuOption(menu, "Field of View", p, vec2i(-1,0));

			menuOptionSliderInt(menu, vec2(rightX, p.y), &ad->fieldOfView, 20, 90, 1);

			//

			p.y -= optionOffset;

			// Game specific.

			p.y -= optionOffset; p.x = leftX;
			menuOption(menu, "View Distance", p, vec2i(-1,0));

			menuOptionSliderInt(menu, vec2(rightX, p.y), &voxelSettings->viewDistance, 2, 32, 1);

			//

			p.y -= optionOffset;

			p.x = rWidth * 0.5f;
			p.y -= optionOffset;
			if(menuOption(menu, "Back", p, vec2i(0,0)) || 
			      input->keysPressed[KEYCODE_ESCAPE] ||
			      input->keysPressed[KEYCODE_BACKSPACE]) {
				addTrack("ui\\menuPop.ogg", menu->volume, true);

				menu->screen = MENU_SCREEN_MAIN;
				menu->activeId = 0;
			}

		}
	}

	if(ad->gameMode == GAME_MODE_LOAD) {

		theCommandList = &ad->commandList2d;

		int titleFontHeight = ds->fontHeightScaled * 8.0f;
		Font* titleFont = getFont("Merriweather-Regular.ttf", titleFontHeight);

		Rect sr = getScreenRect(ws);

		dcRect(sr, vec4(0,1));
		dcText("Loading", titleFont, rectCen(sr), vec4(1,1), vec2i(0,-1));

		// @InitNewGame.

		if(!ad->loading && threadQueueFinished(theThreadQueue)) {
			ad->loading = true;

			bool hasSaveState;
			char* saveFile = fillString("%s%s", SAVES_FOLDER, SAVE_STATE1);
			if(fileExists(saveFile)) hasSaveState = true;

			// Pre work.
			{
				resetVoxelHashAndMeshes(&ad->voxelData);
			}

			// Load game.

			if(!ad->newGame && hasSaveState) {
				DArray<VoxelMesh>* voxels = &ad->voxelData.voxels;
				voxels->clear();

				VoxelWorldSettings* vs = &ad->voxelSettings;

				FILE* file = fopen(saveFile, "rb");
				if(file) {
					fread(&ad->inventory, sizeof(Inventory), 1, file);

					fread(ad->entityList.e, ad->entityList.size * sizeof(Entity), 1, file);

					fread(&vs->startX, sizeof(int), 1, file);
					fread(&vs->startY, sizeof(int), 1, file);
					fread(&vs->startXMod, sizeof(int), 1, file);
					fread(&vs->startYMod, sizeof(int), 1, file);

					int count = 0;
					fread(&count, sizeof(int), 1, file);

					voxels->reserve(count);
					voxels->count = count;

					for(int i = 0; i < voxels->count; i++) {
						VoxelMesh* mesh = voxels->data + i;
						initVoxelMesh(mesh, vec2i(0,0));

						fread(&mesh->coord, sizeof(Vec2i), 1, file);

						fread(&mesh->compressedVoxelsSize, sizeof(int), 1, file);
						fread(&mesh->compressedLightingSize, sizeof(int), 1, file);

						int compressedDataCountTotal = mesh->compressedVoxelsSize + mesh->compressedLightingSize;
						mesh->compressedData = mallocArray(uchar, compressedDataCountTotal * (sizeof(uchar)*2));

						fread(mesh->compressedData, compressedDataCountTotal * (sizeof(uchar)*2), 1, file);
					}

				}

				for(int i = 0; i < voxels->count; i++) {
					VoxelMesh* m = voxels->data + i;
					m->stored = true;
					m->generated = true;
					m->upToDate = false;
					m->uploaded = false;

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

				player->vel = vec3(0,0,0);

				ad->player = player;
				ad->cameraEntity = camera;

				if(ad->playerMode) ad->chunkOffset = player->chunk;
				else ad->chunkOffset = camera->chunk;

			} else {

				// New Game.

				for(int i = 0; i < ad->entityList.size; i++) ad->entityList.e[i].init = false;

				// Init player.
				{
					float v = randomFloat(0,M_2PI);
					Vec3 startRot = vec3(v,0,0);

					Entity player = {};
					Vec3 playerDim = vec3(0.8f, 0.8f, 1.8f);
					float camOff = playerDim.z*0.5f - playerDim.x*0.25f;
					// initEntity(&player, ET_Player, vec3(0.5f,0.5f,40), playerDim);
					initEntity(&player, ET_Player, vec3(20,20,40), playerDim, vec2i(0,0));
					player.camOff = vec3(0,0,camOff);
					player.rot = startRot;
					player.onGround = false;
					strCpy(player.name, "Player");
					ad->footstepSoundValue = 0;
					
					ad->player = addEntity(&ad->entityList, &player);
				}

				// @InventoryInit.
				{
					Inventory* inv = &ad->inventory;

					inv->show = false;

					inv->maxStackCount = 20; // Should change for resource type in the future.
					inv->slotCount = 40;
					inv->quickSlotCount = 10;

					for(int i = 0; i < 5; i++) inventoryAdd(inv, BT_Snow);
					for(int i = 0; i < 5; i++) inventoryAdd(inv, BT_TreeLog);
					for(int i = 0; i < 5; i++) inventoryAdd(inv, BT_Glass);
					for(int i = 0; i < 5; i++) inventoryAdd(inv, BT_GlowStone);
					for(int i = 0; i < 5; i++) inventoryAdd(inv, BT_Pumpkin);

					inv->slots[inv->slotCount + 0] = {BT_Sand, 10};
					inv->slots[inv->slotCount + 1] = {BT_Ground, 10};
					inv->slots[inv->slotCount + 2] = {BT_Stone, 10};

					inv->quickSlotSelected = 0;
				}

				// Debug cam.
				{
					Entity freeCam = {};
					initEntity(&freeCam, ET_Camera, vec3(35,35,32), vec3(0,0,0), vec2i(0,0));
					strCpy(freeCam.name, "Camera");
					ad->cameraEntity = addEntity(&ad->entityList, &freeCam);
				}

				ad->voxelSettings.startX = randomInt(0,1000000);
				ad->voxelSettings.startY = randomInt(0,1000000);
				ad->voxelSettings.startXMod = randomInt(0,1000000);
				ad->voxelSettings.startYMod = randomInt(0,1000000);
			}

			// Load voxel meshes around the player at startup.
			{
				Vec2i pPos = coordToMesh(ad->player->pos);
				for(int y = -1; y < 2; y++) {
					for(int x = -1; x < 2; x++) {
						Vec2i coord = pPos - vec2i(x,y);

						VoxelMesh* m = getVoxelMesh(&ad->voxelData, coord);
						makeMesh(m, &ad->voxelData, &ad->voxelSettings);
					}
				}
			}
		}

		if(threadQueueFinished(theThreadQueue)) {

			// Push the player up until he is right above the ground.

			Entity* player = ad->player;
			while(collisionVoxelBox(&ad->voxelData, player->pos, player->dim, player->chunk)) {
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
			camera->chunk = player->chunk;
		}
		ad->playerMode = !ad->playerMode;
	}

	#if 0
	if(input->mouseWheel) {
		ad->blockMenuSelected += -input->mouseWheel;
		ad->blockMenuSelected = mod(ad->blockMenuSelected, arrayCount(ad->blockMenu));
	}

	if(input->keysPressed[KEYCODE_0]) ad->blockMenuSelected = 9;
	for(int i = 0; i < 9; i++) {
		if(input->keysPressed[KEYCODE_0 + i+1]) ad->blockMenuSelected = i;
	}
	#endif

	{
		if(input->mouseWheel) {
			ad->inventory.quickSlotSelected += -input->mouseWheel;
			ad->inventory.quickSlotSelected = mod(ad->inventory.quickSlotSelected, ad->inventory.quickSlotCount);
		}

		if(!input->keysDown[KEYCODE_CTRL]) {
			if(input->keysPressed[KEYCODE_0]) ad->inventory.quickSlotSelected = 9;
			for(int i = 0; i < 9; i++) {
				if(input->keysPressed[KEYCODE_0 + i+1]) ad->inventory.quickSlotSelected = i;
			}
		}

		if(ad->playerMode && (input->keysPressed[KEYCODE_TAB] || 
		   					  input->keysPressed[KEYCODE_I])) {
			ad->inventory.show = !ad->inventory.show;
		}
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
		player->chunk = camera->chunk;
		ad->playerMode = true;
		input->keysPressed[KEYCODE_SPACE] = false;
		input->keysDown[KEYCODE_SPACE] = false;
	}

	#if 0
	// spawn bomb
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

		if(e->pos.x >= VOXEL_X) { e->pos.x -= VOXEL_X; e->chunk.x++; }
		else if(e->pos.x <  0)  { e->pos.x += VOXEL_X; e->chunk.x--; }
		if(e->pos.y >= VOXEL_Y) { e->pos.y -= VOXEL_Y; e->chunk.y++; }
		else if(e->pos.y < 0)   { e->pos.y += VOXEL_Y; e->chunk.y--; }

		switch(e->type) {

			case ET_Player: {
				if(!ad->playerMode) continue;

				ad->chunkOffset = ad->player->chunk;
				ad->voxelOffset = chunkOffsetToVoxelOffset(ad->chunkOffset);

				if((input->mouseButtonDown[1] && ad->debugMouse) || 
				   ad->fpsMode)
					entityMouseLook(ad->player, input, ad->mouseSensitivity);

				e->acc = vec3(0,0,0);
				entityKeyboardAcceleration(ad->player, input, 30, 1.5f, false);
				e->acc.z = 0;

				bool onGroundStart = player->onGround;

				if(input->keysDown[KEYCODE_SPACE]) {
					if(player->onGround) {
						player->vel += up*7.0f;
						player->onGround = false;
					}
				}

				float gravity = 20.0f;
				if(!e->onGround) e->acc += -up*gravity;
				e->vel = e->vel + e->acc*dt;
				float friction = 0.01f;
				e->vel.x *= pow(friction,dt);
				e->vel.y *= pow(friction,dt);
				// e->vel *= 0.9f;

				if(e->onGround) e->vel.z = 0;

				Vec3 collisionNormal = vec3(0,0,0);
				Vec2 positionOffset = vec2(0,0);
				uchar groundCollisionBlockType = 0;

				{
					if(e->vel.xy != vec2(0,0)) ad->firstWalk = true;

					Vec3 nPos = e->pos + -0.5f*e->acc*dt*dt + e->vel*dt;

					bool result = collisionVoxelBoxDistance(&ad->voxelData, nPos, e->dim, e->chunk, &nPos, &collisionNormal);

					if(!result) {
						nPos.z += 5;
						e->vel = vec3(0,0,0);
						// assert(false);
					}

					float stillnessThreshold = 0.0001f;
					if(between(e->vel.z, -stillnessThreshold, stillnessThreshold)) {
						e->vel.z = 0;
					}

					if(collisionNormal.z < 0) {
						e->vel.z = 0;
					}

					if(collisionNormal.xy != vec2(0,0)) {
						// float sideFriction = 0.001f;
						float sideFriction = 0.01f;
						e->vel.x *= pow(sideFriction,dt);
						e->vel.y *= pow(sideFriction,dt);
					}

					positionOffset = nPos.xy - e->pos.xy;
					e->pos = nPos;


					// raycast for touching ground

					if(collisionNormal.z > 0 && e->vel.z <= 0) {
						e->onGround = true;
						e->vel.z = 0;
					}

					if(e->onGround) {

						bool groundCollision = raycastGroundVoxelBox(&ad->voxelData, e->pos, e->dim, e->chunk, &groundCollisionBlockType);

						if(groundCollision) {
							if(e->vel.z <= 0) {
								e->onGround = true;
							}
						} else {
							e->onGround = false;
						}

						if(ad->firstWalk) {
							// Hit the ground.
							if(!onGroundStart && player->onGround) ad->footstepSoundValue = 100;

							if(!e->onGround) ad->footstepSoundValue = 0;
						}
					}
				}

				// Footstep sound.
				if(e->vel != vec3(0,0,0) && groundCollisionBlockType) {
					float moveLength = len(positionOffset);
					if(player->onGround) ad->footstepSoundValue += moveLength;

					float stepDistance = 2.3f;
					if(ad->footstepSoundValue > stepDistance) {

						int footstepType = blockTypeFootsteps[groundCollisionBlockType];
						FootstepArray fa = footstepFiles[footstepType];
						char** files = fa.files;
						int filesCount = fa.count;

						int soundId = randomInt(0,filesCount-1);
						if(soundId == ad->lastFootstepSoundId) 
							soundId = (soundId+1)%(filesCount-1);

						addTrack(files[soundId], ad->volumeFootsteps, true, 2.0f);
						ad->footstepSoundValue = 0;
						
						ad->lastFootstepSoundId = soundId;
					}
				}
			} break;

			case ET_Camera: {
				if(ad->playerMode) continue;

				ad->chunkOffset = ad->cameraEntity->chunk;
				ad->voxelOffset = chunkOffsetToVoxelOffset(ad->chunkOffset);

				if((!ad->fpsMode && input->mouseButtonDown[1]) || ad->fpsMode)
					entityMouseLook(ad->cameraEntity, input, ad->mouseSensitivity);

				e->acc = vec3(0,0,0);
				float speed = !input->keysDown[KEYCODE_T] ? 150 : 1000;
				entityKeyboardAcceleration(ad->cameraEntity, input, speed, 2.0f, true);

				e->vel = e->vel + e->acc*dt;
				float friction = 0.01f;
				e->vel = e->vel * pow(friction,dt);

				if(e->vel != vec3(0,0,0)) {
					e->pos = e->pos - 0.5f*e->acc*dt*dt + e->vel*dt;
				}

			} break;

			case ET_ParticleEmitter: {

				ParticleEmitter* emitter = &e->emitter;
				emitter->pos = e->pos;
				emitter->time += ad->dt;

				float timeToLive = 10;
				float startAlphaFade = 0.75f;

				// Don't spawn new particles if we're done.
				if(emitter->liveTimeSpawns < emitter->liveTimeSpawnCount) {
					emitter->spawnTime += ad->dt;
					while(emitter->spawnTime >= emitter->spawnRate) {
						emitter->spawnTime -= emitter->spawnRate;

						for(int i = 0; i < emitter->spawnCount; i++) {
							if(emitter->particleListCount >= emitter->particleListSize) break;

							Particle p = {};
							float velSpeed = randomFloat(0.5f,1.0f);
							p.vel = randomUnitSphereDirection() * velSpeed;

							p.pos = emitter->pos + p.vel*0.3f;

							p.acc = vec3(0,0,-1) * 2.0f;

							p.size = vec3(0.05f) + randomOffset(0.01f);
							p.timeToLive = timeToLive;

							p.color = emitter->color;

							float co = 0.05f;
							Vec3 hsl = rgbToHslf(p.color.rgb);
							hsl.y = clamp01(hsl.y + randomOffset(co));
							hsl.z = clamp01(hsl.z + randomOffset(co));
							p.color.rgb = hslToRgbf(hsl);

							emitter->particleList[emitter->particleListCount++] = p;
						}

						emitter->liveTimeSpawns++;
					}
				}

				float animSpeed = 2;
				float dt = ad->dt*animSpeed;

				particleEmitterUpdate(emitter, dt, ad->dt);
				particleEmitterFinish(emitter);

				// Particle world collision.
				{
					for(int i = 0; i < emitter->particleListCount; i++) {
						Particle* p = emitter->particleList + i;

						entityWorldCollision(&p->pos, p->size, &p->vel, e->chunk, &ad->voxelData, dt);
					}
				}

				Vec3i voxelOffset = chunkOffsetToVoxelOffset(e->chunk);

				for(int i = 0; i < emitter->particleListCount; i++) {
					Particle* p = emitter->particleList + i;

					float alpha = p->color.a;
					if(p->time > (startAlphaFade*timeToLive)) {
						alpha = mapRange((float)p->time, startAlphaFade*timeToLive, timeToLive, p->color.a, 0.0f);
					}

					Vec3i voxel = coordToVoxel(p->pos);
					uchar* lighting = getLightingFromVoxel(&ad->voxelData, voxel + voxelOffset);
					float fLighting = lighting ? (*lighting) / 255.0f : 1;

					Vec3 pos = p->pos;
					Vec2i offset = e->chunk - ad->chunkOffset;
					pos.xy += vec2(offset * vec2i(VOXEL_X, VOXEL_Y));

					dcCube(pos, p->size, vec4(p->color.rgb * fLighting, alpha));
				}

				if(emitter->liveTimeSpawns == emitter->liveTimeSpawnCount && 
				   emitter->particleListCount == 0) {
					e->init = false;
					ad->particleListUsage[emitter->particleListIndex] = false;
				}

			} break;

			case ET_BlockResource: {

				float gravity = 10.0f;
				e->acc = vec3(0,0,-1)*gravity;

				e->vel = e->vel + e->acc*dt;
				e->pos = e->pos - 0.5f*e->acc*dt*dt + e->vel*dt;

				entityWorldCollision(e, &ad->voxelData, ad->dt);

				Vec3i voxelOffset = chunkOffsetToVoxelOffset(e->chunk);
				Vec3 pos = e->pos + vec3(voxelOffset - ad->voxelOffset);

				// Collecting.
				{
					float distanceMod = 2.0f;
					bool intersection = boxIntersection(ad->player->pos, ad->player->dim, pos, e->dim*distanceMod);

					if(intersection) {

						bool result = inventoryAdd(&ad->inventory, e->blockType);
						if(result) {
							addTrack("general\\remove.ogg", ad->volumeGeneral, true, 1);

							e->init = false;
							break;
						}
					}
				}

				Vec3i voxel = coordToVoxel(e->pos);
				uchar* lighting = getLightingFromVoxel(&ad->voxelData, voxel + voxelOffset);
				float fLighting = lighting ? (*lighting) / 255.0f : 1;

				drawVoxelCube(pos, e->dim.x, vec4(fLighting,1), e->blockType);

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
							Vec3 dir = rotateVec3(norm(vec3(0,1,0)), off, vec3(1,0,0));
							float off2 = sin(off/(float)2)*sRad;

							float rad = (dir*sRad).y;
							for(int i = 0; i < 2; i++) {
								Vec3 pos;
								if(i == 0) pos = startPos + vec3(0,off2,0);
								else pos = startPos + vec3(0,-off2,0);

								int itCount = rad*resolution;
								for(int it = 0; it < itCount+1; it++) {
									float off = degreeToRadian(it * (360/(float)itCount));
									Vec3 dir = rotateVec3(norm(vec3(1,0,0)), off, vec3(0,-1,0));
									Vec3 p = pos + dir*rad;

									float cubeSize = 1.0f;

									// dcCube({coordToVoxelCoord(pos + dir*rad), vec3(cubeSize), vec4(1,0.5f,0,1), 0, vec3(0,0,0)});
									// dcCube({coordToVoxelCoord(pos - dir*rad), vec3(cubeSize), vec4(1,0.5f,0,1), 0, vec3(0,0,0)});

									int globalLumen = ad->voxelSettings.globalLumen;
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
							m->uploaded = false;
							m->modified = true;
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

	// @Selecting blocks and modifying them.
	if(ad->playerMode) 
	{
		ad->blockSelected = false;

		bool intersection = false;
		Vec3 intersectionBox;
		Vec3 intersectionNormal;

		{
			Vec3 dir = ad->activeCam.look;
			Vec3 startPos = player->pos + player->camOff;
			Vec3 pos = startPos;

			int smallerAxis[2];
			int biggestAxis = getBiggestAxis(dir, smallerAxis);

			for(int i = 0; i < ad->selectionRadius; i++) {
				pos = pos + norm(dir);

				Vec3 coords[9];
				int coordsSize = 0;

				Vec3 blockCoords = voxelToVoxelCoord(coordToVoxel(pos));

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
				float minDistance = FLT_MAX;

				for(int i = 0; i < coordsSize; i++) {
					Vec3 block = coords[i];

					Vec3i voxel = coordToVoxel(block);
					uchar* blockType = getBlockFromVoxel(&ad->voxelData, voxel + ad->voxelOffset);

					if(blockType && *blockType > 0) {
						Vec3 iBox = voxelToVoxelCoord(voxel);
						int face;
						Vec3 intersectionPoint;
						Vec3 normal;
						float distance = boxRaycast(startPos, dir, iBox, vec3(1,1,1), &intersectionPoint, &normal);

						if(distance != -1 && distance < minDistance) {
							minDistance = distance;
							intersectionBox = iBox;
							intersectionNormal = normal;

							intersection = true;
						}
					}
				}

				if(intersection) break;
			}
		}

		bool activeBreaking = false;

		if(intersection) {
			ad->selectedBlock = intersectionBox;
			ad->blockSelected = true;

			Vec3 faceDir = intersectionNormal;
			ad->selectedBlockFaceDir = faceDir;

			if(ad->playerMode && ad->fpsMode) {
				Vec3i voxel = coordToVoxel(intersectionBox) + ad->voxelOffset;

				VoxelMesh* vm = getVoxelMesh(&ad->voxelData, voxelToMesh(voxel));

				uchar* block = getBlockFromVoxel(&ad->voxelData, voxel);
				uchar* lighting = getLightingFromVoxel(&ad->voxelData, voxel);

				bool mouse1 = input->mouseButtonPressed[0];
				bool mouse2 = input->mouseButtonDown[1];
				bool placeBlock = (!ad->fpsMode && ad->pickMode && mouse1) || (ad->fpsMode && mouse1);
				bool removeBlock = (!ad->fpsMode && !ad->pickMode && mouse1) || (ad->fpsMode && mouse2);

				bool breakedBlock = false;
				if(removeBlock) {
					activeBreaking = true;

					if(voxel != ad->breakingBlock) {
						ad->breakingBlockTime = 0;
						ad->breakingBlock = voxel;
					}

					ad->breakingBlockTime += ad->dt;
					ad->breakingBlockTimeSound += ad->dt;

					float breakingSoundTimeMax = 0.25f;
					if(ad->breakingBlockTimeSound >= breakingSoundTimeMax) {
						ad->breakingBlockTimeSound = 0;
						addTrack("general\\dig.ogg", ad->volumeGeneral*0.5f, true, 1);
					}

					if(ad->breakingBlockTime >= blockTypeHardness[*block]*ad->hardnessModifier) {
						ad->breakingBlockTime = 0;

						breakedBlock = true;
					}
				} else {
					ad->breakingBlockTimeSound = 100;
				}

				if(placeBlock || breakedBlock) {
					vm->upToDate = false;
					vm->uploaded = false;
					vm->modified = true;

					// if block at edge of mesh, we have to update the mesh on the other side too
					Vec2i currentCoord = voxelToMesh(voxel);
					Vec3i offsets[] = {vec3i(1,0,0), vec3i(-1,0,0), vec3i(0,1,0), vec3i(0,-1,0), };
					for(int i = 0; i < 4; i++) {
						Vec3i offset = offsets[i];

						Vec2i mc = voxelToMesh(voxel + offset);
						if(mc != currentCoord) {
							VoxelMesh* edgeMesh = getVoxelMesh(&ad->voxelData, mc);
							edgeMesh->upToDate = false;
							edgeMesh->uploaded = false;
							edgeMesh->modified = true;
						}
					}
				}

				if(placeBlock) {
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
						voxelSideBlock += ad->voxelOffset;
						uchar* sideBlockType = getBlockFromVoxel(&ad->voxelData, voxelSideBlock);
						uchar* sideBlockLighting = getLightingFromVoxel(&ad->voxelData, voxelSideBlock);

						int resourceType = inventoryRemoveQuick(&ad->inventory);

						if(resourceType) {
							*sideBlockType = resourceType;
							*sideBlockLighting = 0;

							addTrack("general\\place.ogg", ad->volumeGeneral, true, 1);
						}
					}
				} else if(breakedBlock) {
					if(*block > 0) {

						uchar blockType = *block;

						*block = 0;
						*lighting = ad->voxelSettings.globalLumen;

						// @SpawnParticles.
						{
							ParticleEmitter emitter = {};

							emitter.particleListSize = ad->particleListsSize;

							emitter.particleListIndex = -1;
							for(int i = 0; i < arrayCount(ad->particleLists); i++) {
								if(!ad->particleListUsage[i]) {
									ad->particleListUsage[i] = true;

									emitter.particleList = ad->particleLists[i];
									emitter.particleListIndex = i;
									break;
								}
							}
							if(emitter.particleListIndex != -1) {

								emitter.spawnRate = 10000;
								emitter.spawnCount = 20;
								emitter.spawnTime = emitter.spawnRate;
								emitter.liveTimeSpawnCount = 1;
								emitter.color = blockTypeParticleColor[blockType];

								Entity e;
								initEntity(&e, ET_ParticleEmitter, intersectionBox, vec3(0,0,0), ad->chunkOffset);

								e.emitter = emitter;

								addEntity(&ad->entityList, &e);
							}
						}

						// @SpawnBlockResource
						{
							Entity e;
							initEntity(&e, ET_BlockResource, intersectionBox, vec3(1), ad->chunkOffset);
							inventoryInitResource(&e, blockType);

							float velSpeed = randomFloat(0.5f,1.0f);
							e.vel = randomUnitHalfSphereDirection(vec3(0,0,1)) * velSpeed;
							e.pos += e.vel*0.3f;

							addEntity(&ad->entityList, &e);
						}

						addTrack("general\\remove.ogg", ad->volumeGeneral, true, 1, 0.75f);
					}
				}
			}
		}

		if(!activeBreaking) {
			ad->breakingBlockTime = 0;
			// ad->breakingBlockTime -= ad->dt*2;
			// clampMin(&ad->breakingBlockTime, 0);
		}
	}

	// Main view proj setup.
	Mat4 view, proj;
	{
		ad->farPlane = ad->voxelSettings.viewDistance * VOXEL_X;

		viewMatrix(&view, ad->activeCam.pos, -ad->activeCam.look, ad->activeCam.up, ad->activeCam.right);
		projMatrix(&proj, degreeToRadian(ad->fieldOfView), ad->aspectRatio, ad->nearPlane, ad->farPlane);

		bindFrameBuffer("3dMsaa");

		// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VIEW, view);
		// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_PROJ, proj);

		pushUniform(SHADER_CUBE, 0, "view", &view);
		pushUniform(SHADER_CUBE, 0, "proj", &proj);
	}	

	// Draw cubemap.
	{
		drawCubeMap(ad->skybox, ad->player, ad->cameraEntity, ad->playerMode, ad->fieldOfView, ad->aspectRatio, &ad->voxelSettings, false);
	}

	if(ad->reloadWorld) {
		ad->reloadWorld = false;

		threadQueueComplete(threadQueue);

		resetVoxelHashAndMeshes(&ad->voxelData);

		for(int i = 0; i < ad->entityList.size; i++) ad->entityList.e[i].init = false;
	}

	TIMER_BLOCK_BEGIN_NAMED(world, "Upd World");

	Vec2i* coordList = (Vec2i*)getTMemory(sizeof(Vec2i)*2000);
	int coordListSize = 0;

	VoxelWorldSettings* vs = &ad->voxelSettings;

	// Collect voxel meshes to draw.
	{
		int meshGenerationCount = 0;
		int radCounter = 0;

		ad->voxelTriangleCount = 0;
		ad->voxelDrawCount = 0;

		Vec2i pPos = ad->chunkOffset;
		int storeDistance = (vs->viewDistance + vs->storeDistanceOffset);
		int radius = storeDistance + vs->storeSize;

		// generate the meshes around the player in a spiral by drawing lines and rotating
		// the directing every time we reach a corner
		for(int r = 0; r < radius; r++) {
			int lineLength = r == 0 ? 1 : 8*r;
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

				// Throw away coordinates outside store circle.
				float distance = len(vec2(coord - pPos));
				if((distance >= storeDistance) && distance <= (storeDistance + vs->storeSize)) {

					VoxelMesh* mesh = getVoxelMesh(&ad->voxelData, coord, false);
					if(mesh && !mesh->stored && mesh->generated && !mesh->activeMaking) {
						if(!mesh->compressionStep) {
							if(!mesh->activeStoring && !threadQueueFull(theThreadQueue)) {
								atomicAdd(&mesh->activeStoring);
								threadQueueAdd(theThreadQueue, storeMeshThreaded, mesh);
							}
						} else {
							freeVoxelMesh(mesh);
							mesh->compressionStep = false;
							mesh->stored = true;
						}
					}

					continue;
				}

				// Throw away coordinates outside view circle.
				if(distance >= vs->viewDistance+1) continue;

				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coord);

				if(!m->uploaded) {
					makeMesh(m, &ad->voxelData, &ad->voxelSettings);
					meshGenerationCount++;

					if(!m->modified) continue;
				}

				// Throw away coordinates outside view circle.
				if(distance > vs->viewDistance) continue;

				// frustum culling
				Vec3 cp = ad->activeCam.pos;
				Vec3 cl = ad->activeCam.look;
				Vec3 cu = ad->activeCam.up;
				Vec3 cr = ad->activeCam.right;

				float aspect = ad->aspectRatio;
				float fovV = degreeToRadian(ad->fieldOfView);
				float fovH = degreeToRadian(ad->fieldOfView) * aspect;

				Vec3 left =   rotateVec3(cl,  fovH * 0.5f, cu);
				Vec3 right =  rotateVec3(cl, -fovH * 0.5f, cu);
				Vec3 top =    rotateVec3(cl,  fovV * 0.5f, cr);
				Vec3 bottom = rotateVec3(cl, -fovV * 0.5f, cr);

				Vec3 normalLeftPlane = cross(cu, left);
				Vec3 normalRightPlane = cross(right, cu);
				Vec3 normalTopPlane = cross(cr, top);
				Vec3 normalBottomPlane = cross(bottom, cr);

				coord -= ad->chunkOffset;
				Vec3 boxPos = vec3(coord.x*VOXEL_X+VOXEL_X*0.5f, coord.y*VOXEL_Y+VOXEL_Y*0.5f, VOXEL_Z*0.5f);
				Vec3 boxSize = vec3(VOXEL_X, VOXEL_Y, VOXEL_Z);

				Vec3 testNormals[] = {normalLeftPlane, normalRightPlane, normalTopPlane, normalBottomPlane};

				Vec3 offsets[] = {
					vec3( 0.5f,  0.5f, -0.5f),
					vec3(-0.5f,  0.5f, -0.5f),
					vec3( 0.5f, -0.5f, -0.5f),
					vec3(-0.5f, -0.5f, -0.5f),
					vec3( 0.5f,  0.5f,  0.5f),
					vec3(-0.5f,  0.5f,  0.5f),
					vec3( 0.5f, -0.5f,  0.5f),
					vec3(-0.5f, -0.5f,  0.5f),
				};

				bool isIntersecting = true;	
				for(int test = 0; test < 4; test++) {

					Vec3 testNormal = testNormals[test];

					bool inside = false;
					for(int i = 0; i < 8; i++) {
						Vec3 boxPoint = boxPos + boxSize*offsets[i];
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
					coordList[coordListSize++] = m->coord;

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
			float distanceToCamera = len(ad->activeCam.pos.xy - c);
			sortList[sortListSize++] = {distanceToCamera, i};
		}

		radixSortPair(sortList, sortListSize);

		// for(int i = 0; i < sortListSize-1; i++) {
		// 	assert(sortList[i].key <= sortList[i+1].key);
		// }
	}

	#if 0
	// Store chunks that are in store region.
	{
		Vec2i playerPos = coordToMesh(ad->activeCam.pos);

		for(int i = 0; i < 3; i++) {
			int dist;
			if(i == 0) dist = TEXTURE_CACHE_DISTANCE;
			else if(i == 1) dist = STORE_DISTANCE;
			else if(i == 2) dist = DATA_CACHE_DISTANCE;

			Vec2i startPos = vec2i(dist, dist);
			Vec2i dir = vec2i(0,-1);
			Vec2i pos = startPos;
			do {
				VoxelMesh* mesh = getVoxelMesh(&ad->voxelData, playerPos + pos, false);
				if(mesh) {
					if(i == 0) {
						freeVoxelGPUData(mesh);
						mesh->upToDate = false;
						mesh->uploaded = false;
					} else if(i == 1 && !mesh->stored && mesh->generated && !mesh->activeGeneration
					          && !mesh->activeMaking && mesh->uploaded) {
						if(!threadQueueFull(theThreadQueue)) {
							threadQueueAdd(theThreadQueue, storeMesh, mesh);
						}
					} else if(i == 2 && mesh->stored && mesh->generated && !mesh->activeGeneration
					          && !mesh->activeMaking && mesh->uploaded) {
						if(!threadQueueFull(theThreadQueue)) {
							threadQueueAdd(theThreadQueue, restoreMesh, mesh);
						}
					}
				}

				pos += dir;
				if(abs(pos.x) == dist && abs(pos.y) == dist) dir = vec2i(dir.y, -dir.x);
			} while(pos != startPos);
		}
	}
	#endif

	TIMER_BLOCK_END(world);

	TIMER_BLOCK_BEGIN_NAMED(worldDraw, "Draw World");

	// Draw voxel world and reflection.
	{
		VoxelWorldSettings* vs = &ad->voxelSettings;
		int waterLevelHeight = vs->waterLevelHeight;

		setupVoxelUniforms(ad->activeCam.pos, view, proj, vs->fogColor.rgb, vs->viewDistance);

		// TIMER_BLOCK_NAMED("D World");
		// draw world without water
		{
			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, ad->chunkOffset, 2);
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

			pushUniform(SHADER_VOXEL, 0, "clipPlane", true);
			pushUniform(SHADER_VOXEL, 0, "cPlane", 0,0,1,-waterLevelHeight);
			pushUniform(SHADER_VOXEL, 0, "cPlane2", 0,0,-1,waterLevelHeight);
			pushUniform(SHADER_VOXEL, 1, "alphaTest", 0.5f);

			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, ad->chunkOffset, 1);
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

			bindFrameBuffer("Reflection");
			glClearColor(0,0,0,0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

			Vec2i reflectionRes = ad->cur3dBufferRes;
			blitFrameBuffers("3dMsaa", "Reflection", ad->cur3dBufferRes, GL_STENCIL_BUFFER_BIT, GL_NEAREST);

			glEnable(GL_CLIP_DISTANCE0);
			// glEnable(GL_CLIP_DISTANCE1);
			glEnable(GL_DEPTH_TEST);

			drawCubeMap(ad->skybox, ad->player, ad->cameraEntity, ad->playerMode, ad->fieldOfView, ad->aspectRatio, &ad->voxelSettings, true);

			glFrontFace(GL_CCW);

			setupVoxelUniforms(ad->activeCam.pos, view, proj, vs->fogColor.rgb, vs->viewDistance, vec3(0,0,waterLevelHeight*2 + 0.01f), vec3(1,1,-1));
			pushUniform(SHADER_VOXEL, 0, "clipPlane", true);
			pushUniform(SHADER_VOXEL, 0, "cPlane", 0,0,-1,waterLevelHeight);

			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, ad->chunkOffset, 2);
			}
			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, ad->chunkOffset, 1);
			}

			glFrontFace(GL_CW);
			glDisable(GL_CLIP_DISTANCE0);
			// glDisable(GL_CLIP_DISTANCE1);
			glDisable(GL_STENCIL_TEST);
		}

		// draw reflection texture	
		{ 
			bindFrameBuffer("3dMsaa");
			glDisable(GL_DEPTH_TEST);

			bindShader(SHADER_QUAD);
			drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), vec4(1,1,1,ad->voxelSettings.reflectionAlpha), rect(0,1,1,0), 
			         getFrameBuffer("Reflection")->colorSlot[0]->id);

			glEnable(GL_DEPTH_TEST);
		}

		// draw water
		{
			setupVoxelUniforms(ad->activeCam.pos, view, proj, vs->fogColor.rgb, vs->viewDistance);
			pushUniform(SHADER_VOXEL, 1, "alphaTest", 0.5f);

			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(&ad->voxelData, coordList[sortList[i].index]);
				drawVoxelMesh(m, ad->chunkOffset, 1);
			}
		}
	}

	TIMER_BLOCK_END(worldDraw);

	// Draw player and selected block.
	{
		if(!ad->playerMode) {
			Vec3i voxelOffset = chunkOffsetToVoxelOffset(player->chunk) - ad->voxelOffset;
			Vec3 pos = player->pos + vec3(voxelOffset);

			Camera cam = getCamData(pos, player->rot);
			Vec3 pCamPos = pos + player->camOff;
			float lineLength = 0.5f;

			dcState(STATE_LINEWIDTH, 3);

			dcLine(pCamPos, pCamPos + cam.look*lineLength, vec4(1,0,0,1));
			dcLine(pCamPos, pCamPos + cam.up*lineLength, vec4(0,1,0,1));
			dcLine(pCamPos, pCamPos + cam.right*lineLength, vec4(0,0,1,1));

			dcState(STATE_LINEWIDTH, 1);

			dcState(STATE_POLYGONMODE, POLYGON_MODE_LINE);
			dcCube(pos, player->dim, vec4(1,1,1,1), 0, vec3(0,0,0));
			dcState(STATE_POLYGONMODE, POLYGON_MODE_FILL);
		} else {

			// @DrawBreakingBlock.
			if(ad->breakingBlockTime > 0)
			{
				float breakingOverlayAlpha = 0.5f;
				uchar* blockType = getBlockFromVoxel(&ad->voxelData, ad->breakingBlock);
				float blockTime = blockTypeHardness[*blockType]*ad->hardnessModifier;

				dcEnable(STATE_POLYGON_OFFSET);

				dcPolygonOffset(-1.0f,-1.0f);

				Vec3i breakingBlock = ad->breakingBlock;
				Vec3 block = voxelToVoxelCoord(breakingBlock - ad->voxelOffset);
				Vec3 fds[3] = {};
				getVoxelShowingVoxelFaceDirs(ad->activeCam.pos - ad->selectedBlock, fds);

				Texture* tex = getTexture("minecraft\\destroyStages.png");
				for(int i = 0; i < 3; i++) {

					Vec3 vs[4];
					getVoxelQuadFromFaceDir(block, fds[i], vs, 1);

					int breakStage = (ad->breakingBlockTime/blockTime) * 10;

					float texSize = (float)tex->dim.h/(float)tex->dim.w;
					float texPos = texSize * breakStage;
					Rect uv = rect(texPos + texSize, 0, texPos, 1);

					dcQuad(vs[0], vs[1], vs[2], vs[3], vec4(1,breakingOverlayAlpha), tex->id, uv);
				}

				dcDisable(STATE_POLYGON_OFFSET);
			}

			// @SelectedBlock.
			if(ad->blockSelected) 
			{

				dcBlend(GL_DST_COLOR, GL_ONE, GL_ZERO, GL_ONE, GL_FUNC_ADD, GL_FUNC_ADD);
				dcEnable(STATE_POLYGON_OFFSET);

				dcPolygonOffset(-2.0f,-2.0f);

				float highlight1 = 0.75f;
				float highlight2 = 1.0f;

				Vec3 fds[3] = {};
				getVoxelShowingVoxelFaceDirs(ad->activeCam.pos - ad->selectedBlock, fds);

				Vec3 vs[4];

				for(int i = 0; i < 3; i++) {
					Vec3 c = fds[i] == ad->selectedBlockFaceDir ? vec3(highlight2) : vec3(highlight1); 
					getVoxelQuadFromFaceDir(ad->selectedBlock, fds[i], vs, 1);

					dcQuad(vs[0], vs[1], vs[2], vs[3], vec4(c,0));
				}

				dcDisable(STATE_POLYGON_OFFSET);
				dcBlend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_FUNC_ADD);
			}
		}
	}

	// @Inventory.

	if(ad->playerMode) {
		theCommandList = &ad->commandList2d;

		Inventory* inv = &ad->inventory;

		Vec4 cBackground = vec4(0.22f,1);
		Vec4 cCells = vec4(0.12f,1);
		Vec4 cFont = vec4(1,1);
		Vec4 cFontShadow = vec4(0,1);

		Font* fTitle = getFont("Merriweather-Regular.ttf", ds->fontHeightScaled * 2.0f);
		Font* fQuantity = getFont("LiberationSans-Regular.ttf", ds->fontHeightScaled * 1.0f);
		Font* fQuick = getFont("LiberationSans-Regular.ttf", ds->fontHeightScaled * 1.0f);

		Vec2 res = vec2(ws->currentRes);
		Rect sr = getScreenRect(ws);

		float cellSize = roundf(0.06f * res.h);
		float cellMargin = roundf(0.005f * res.h);
		float quickBarOffset = 0.04 * res.h;

		float rounding = cellSize*0.1f;

		for(int inventoryStage = 0; inventoryStage < 2; inventoryStage++) {

			if(!inventoryStage && !ad->inventory.show) continue;

			int rowWidth, rowCount;
			Rect r;
			Vec2 topLeft;
			int slotStart, slotEnd;

			if(!inventoryStage) {
				int slotCount = inv->slotCount;
				slotStart = 0;
				slotEnd = slotCount;

				rowWidth = 10;
				rowCount = slotCount / rowWidth;
				if(slotCount % rowWidth != 0) rowCount++;

				r = rectCenDim(rectCen(sr), vec2(cellSize * rowWidth, cellSize * rowCount));

				topLeft = rectTL(r);
				rectExpand(&r, vec2(cellMargin));
				r = round(r);

				// 

				dcRoundedRect(r, cBackground, cellSize*0.1f);

				// Header.
				{
					char* text = "Inventory";
					Vec2 td = getTextDim(text, fTitle) + vec2(fTitle->height*0.5f,fTitle->height*0.05f);

					Vec2 p = rectT(r) - vec2(0,td.h*0.1f);
					dcRect(round(rectBDim(p, td * vec2(1,0.5f))), cBackground);
					dcRoundedRect(round(rectBDim(p, td)), cBackground, rounding);
					dcText(text, fTitle, p, cFont, vec2i(0,-1), 0, 1, cFontShadow);
				}

			} else {
				int slotCount = inv->quickSlotCount;
				slotStart = inv->slotCount;
				slotEnd = slotStart + inv->quickSlotCount;

				rowWidth = slotCount;
				rowCount = 1;

				r = rectBDim(rectB(sr) + vec2(0,quickBarOffset), vec2(cellSize * rowWidth, cellSize * rowCount));

				topLeft = rectTL(r);
				rectExpand(&r, vec2(cellMargin));

				// 
					
				dcRect(round(r), cBackground);
			}

			for(int slotIndex = slotStart; slotIndex < slotEnd; slotIndex++) {
				int x = (slotIndex-slotStart) % rowWidth;
				int y = (slotIndex-slotStart) / rowWidth;

				Vec2 off = vec2(x*cellSize, -y*cellSize);
				Rect cellRect = rectTLDim(topLeft + off, vec2(cellSize));
				Rect cellRectSmall = rectExpand(cellRect, -vec2(cellMargin));

				bool mouseOver = false;
				if(!ad->fpsMode) {

					Vec2 mouseOffset = inv->activeSlotDrag ? inv->dragMouseOffset : vec2(0,0);
					if(pointInRect(input->mousePosNegative + mouseOffset, cellRect)) {
						mouseOver = true;

						if(input->keysDown[KEYCODE_CTRL]) {
							int quickSlotIndex = -1;

							if(input->keysPressed[KEYCODE_0]) quickSlotIndex = 9;
							for(int i = 0; i < 9; i++) {
								if(input->keysPressed[KEYCODE_1 + i]) quickSlotIndex = i;
							}

							if(quickSlotIndex != -1) {
								inventorySwapQuick(inv, slotIndex, quickSlotIndex);
							}
						}

						if(input->mouseButtonPressed[1]) {
							if(inv->slots[slotIndex].count > 0) {
								InventorySlot slot = inv->slots[slotIndex];
								slot.count = 1;
								inv->slots[slotIndex].count--;

								inventoryThrowAway(slot, &ad->entityList, ad->player, &ad->activeCam);
							}
						}

						if(input->mouseButtonPressed[0]) {
							if(inv->slots[slotIndex].count > 0) {
								inv->activeSlotDrag = true;

								inv->dragSlotIndex = slotIndex;

								// Split stack.
								if(input->keysDown[KEYCODE_CTRL]) {
									int countBefore = inv->slots[slotIndex].count;
									int splitCount = ceil(inv->slots[slotIndex].count/2.0f);

									inv->dragSlot = inv->slots[slotIndex];
									inv->dragSlot.count = splitCount;

									inv->slots[slotIndex].count = countBefore - splitCount;
								} else {
									inv->dragSlot = inv->slots[slotIndex];
									inv->slots[slotIndex].count = 0;
								}

								inv->dragMouseOffset = rectCen(cellRect) - input->mousePosNegative;
							}
						}

						if(input->mouseButtonReleased[0] && inv->activeSlotDrag) {
							inv->activeSlotDrag = false;

							if(inv->slots[slotIndex].type == inv->dragSlot.type) {
								if(inv->slots[slotIndex].count + inv->dragSlot.count > inv->maxStackCount) {
									int rest = inv->maxStackCount - inv->slots[slotIndex].count;
									inv->slots[slotIndex].count += rest;
									inv->dragSlot.count -= rest;
									inv->slots[inv->dragSlotIndex].count += inv->dragSlot.count;

								} else {
									inv->slots[slotIndex].count += inv->dragSlot.count;
								}

							} else {
								// Abort stack split.
								if(inv->slots[slotIndex].count > 0 && inv->slots[inv->dragSlotIndex].count > 0) {
									inv->slots[inv->dragSlotIndex].count += inv->dragSlot.count;

								} else {
									InventorySlot temp = inv->slots[slotIndex];
									inv->slots[slotIndex] = inv->dragSlot;

									if(slotIndex != inv->dragSlotIndex && 
									   temp.count > 0) {
										inv->slots[inv->dragSlotIndex] = temp;
									}
								}
							}

						}
					}
				}

				bool selected = inv->quickSlotSelected + inv->slotCount == slotIndex;

				Vec4 c = cCells;
				if(mouseOver) c.rgb += vec3(0.03f);
				if(selected) c.rgb += vec3(0,0.1f,0.2f)*0.8f;

				dcRect(cellRectSmall, c);

				if(slotIndex >= inv->slotCount) {

					Vec2 p = rectT(cellRectSmall) + vec2(0,cellMargin * 0.5f);
					char* t = fillString("%i", slotIndex - inv->slotCount);

					dcRect(rectCenDim(p+vec2(0,fQuick->height*0.05f),vec2(fQuick->height*1.2f)), rect(0,0,1,1), cBackground, "misc\\circle.png");
					dcText(t, fQuick, p, cFont, vec2i(0,0), 0, 1, cFontShadow);
				}

				inventoryDrawIcon(inv->slots[slotIndex], cellRectSmall, fQuantity, cFont, cFontShadow);
			}

		}

		// Draw resource while dragging.
		if(inv->activeSlotDrag) {

			Vec2 p = input->mousePosNegative + inv->dragMouseOffset;

			Rect cellRectSmall = rectCenDim(p, vec2(cellSize - cellMargin));
			inventoryDrawIcon(inv->dragSlot, cellRectSmall, fQuantity, cFont, cFontShadow);
		}

		// Drag throw away.
		if(input->mouseButtonReleased[0] && inv->activeSlotDrag) {
			inv->activeSlotDrag = false;
			inventoryThrowAway(inv->dragSlot, &ad->entityList, ad->player, &ad->activeCam);
		}

		theCommandList = &ad->commandList3d;
	}

	// Start fading.

	if(ad->startFade < 1.0f) {
		float v = ad->startFade;

		ad->startFade += ad->dt / 3.0f;

		float r[] = {0, 0.3f, 1};
		int stage = 0;
		for(int i = 0; i < arrayCount(r); i++) {
			if(v >= r[i] && v < r[i+1]) { stage = i; break; }
		}

		float a = 0;
		if(stage == 0) {
			a = 1;
		} else if(stage == 1) {
			a = mapRange(v, r[stage], r[stage+1], 1.0f, 0.0f);
		}

		Vec4 c = vec4(0,a);
		dcRect(getScreenRect(ws), c, &ad->commandList2d);
	}


	#if 0
	// Particle test.

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
					Vec3 dir = norm(vec3(randomFloat(-1,1,0.01f), randomFloat(-1,1,0.01f), randomFloat(-1,1,0.01f)));
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

		// 	Vec3 normal = norm(p->vel);

		// 	float size = 0.1f;
		// 	Vec4 color = p->color;
		// 	Vec3 base = p->pos;

		// 	Vec3 dir1 = norm(cross(normal, vec3(1,0,0)));
		// 	rotateVec3(&dir1, p->rot2, normal);
		// 	rotateVec3(&normal, p->rot, dir1);
		// 	Vec3 dir2 = norm(cross(normal, dir1));

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

		// 	Vec3 normal = norm(p->vel);

		// 	float size = 0.1f;
		// 	Vec4 color = p->color;
		// 	Vec3 base = p->pos;

		// 	Vec3 dir1 = norm(cross(normal, vec3(1,0,0)));
		// 	rotateVec3(&dir1, p->rot2, normal);
		// 	rotateVec3(&normal, p->rot, dir1);
		// 	Vec3 dir2 = norm(cross(normal, dir1));

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
					Vec3 normal = norm(p->vel);

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
	#endif

	}

	#if 0
	{
		theCommandList = &ad->commandList2d;

		static Vec3* colorArray = 0;
		static Vec2i dim;

		if(init || reload)
		{
			int n;

			unsigned char* stbData = stbi_load("C:\\Projects\\VoxelGame\\text.png", &dim.w, &dim.h, &n, 0);

			colorArray = getPArray(Vec3, dim.w*dim.h);
			for(int i = 0; i < dim.w*dim.h; i++) {
				colorArray[i].r = stbData[i*n + 0] / 255.0f;
				colorArray[i].g = stbData[i*n + 1] / 255.0f;
				colorArray[i].b = stbData[i*n + 2] / 255.0f;
			}

		}

		Vec3 fontColor = vec3(0.0f);
		Vec3 backgroundColor = vec3(1.0f);

		dcRect(rectTLDim(vec2(0,0),vec2(10000,100000)), vec4(backgroundColor,1));

		// float fi[] = {1/9.0f, 2/9.0f, 3/9.0f, 2/9.0f, 1/9.0f, };
		float fi[] = { 0x08/256.0f, 0x4D/256.0f, 0x56/256.0f, 0x4D/256.0f, 0x08/256.0f, };

		Vec2 pos = vec2(10,-10);
		Vec2 pos2 = vec2(100,-300);
		float cellDim = 10;
		float gap = 0;
		for(int y = 0; y < dim.h; y++) {
			for(int x = 0; x < dim.w; x++) {
				Vec3 cm = colorArray[y*dim.w + x];

				Vec3 cl = x-1<0?vec3(0,0,0) : colorArray[y*dim.w + x-1];
				Vec3 cr = x+1>=dim.w?vec3(0,0,0) : colorArray[y*dim.w + x+1];

				Vec3 finalColor;
				finalColor.r = cl.g*fi[0] + cl.b*fi[1] + cm.r*fi[2] + cm.g*fi[3] + cm.b*fi[4];
				finalColor.g = cl.b*fi[0] + cm.r*fi[1] + cm.g*fi[2] + cm.b*fi[3] + cr.r*fi[4];
				finalColor.b = cm.r*fi[0] + cm.g*fi[1] + cm.b*fi[2] + cr.r*fi[3] + cr.g*fi[4];

				finalColor = backgroundColor * (vec3(1)-finalColor) + fontColor * finalColor;
				// finalColor = srgbToLinear(vec4(finalColor,1)).rgb;
				// finalColor = linearToGamma(vec4(finalColor,1)).rgb;



				Vec2 p = pos + vec2(x,-y)*cellDim;

				Rect r = rectCenDim(p, vec2(cellDim));
				rectExpand(&r, vec2(-gap*2));

				dcRect(r, vec4(finalColor,1));



				Vec2 p2 = pos2 + vec2(x,-y)*1;
				dcRect(rectTLDim(p2, vec2(1,1)), vec4(finalColor,1));
			}
		}

		// dcRect(rectCenDim(100,-100,100,100), vec4(1,0,0,1));

		theCommandList = &ad->commandList3d;
	}
	#endif

	#if 0
	{
		theCommandList = &ad->commandList2d;

		static Font font;
		static Font font2;
		if(init || reload) {
			font = {};
			font2 = {};
			char* file = "Merriweather-Regular.ttf";
			int size = 15;
			fontInitSubpixel(&font,  file, size);
			fontInit(&font2, file, size+1);
		}

		Vec2 pos = vec2(10,-30);
		Vec4 c  = vec4(1,1);
		Vec4 cb = vec4(0,1);

		dcRect(rectTLDim(0,0,100000,100000), cb);

		char* text = "The Quick Brown Fox Jumps Over The Lazy Dog. 123";
		dcText(text, &font2, pos, c, vec2i(-1,0));
		dcText(text, &font,  pos - vec2(0,30), c, vec2i(-1,0));

		theCommandList = &ad->commandList3d;
	}
	#endif

	#if 0
	// Visualize chunk storing/restoring.

	{
		VoxelData* vd = &ad->voxelData;

		printf("Count: %i.\n", vd->voxels.count);
		// for(int i = 0; i < vd->voxels.count; i++) {
		// 	VoxelMesh* mesh = vd->voxels.data + i;
		// 	if(!mesh->stored)
		// 		printf(" %i, %i\n", PVEC2(mesh->coord));
		// }
		// for(int i = 0; i < vd->voxels.count; i++) {
		// 	VoxelMesh* mesh = vd->voxels.data + i;
		// 	if(mesh->stored)
		// 		printf(" S: %i, %i\n", PVEC2(mesh->coord));
		// }

		// printf("\n");
	}

	{
		dcState(STATE_POLYGONMODE, POLYGON_MODE_LINE);

		VoxelData* vd = &ad->voxelData;
		for(int i = 0; i < vd->voxels.count; i++) {
			VoxelMesh* mesh = vd->voxels.data + i;

			Vec3 pos = meshToMeshCoord(mesh->coord);
			pos.z = ad->player->pos.z-1-0.3;
			Vec4 c = vec4(0.5f,1);
			if(mesh->generated) c.r += 0.5f;
			if(mesh->upToDate) c.g += 0.5f;
			if(mesh->uploaded) c.b += 0.5f;

			// if(mesh->activeGeneration) c.r += 0.5f;
			// if(mesh->activeMaking) c.g += 0.5f;
			if(mesh->stored) c.rgb -= vec3(0.5f);

			dcCube(pos, vec3(VOXEL_X, VOXEL_Y, 1)*0.95f, c);
		}

		dcState(STATE_POLYGONMODE, POLYGON_MODE_FILL);
	}
	#endif

	endOfMainLabel:



	updateAudio(&ad->audioState, ad->dt);

	// Render.
	{
		TIMER_BLOCK_NAMED("Render");

		bindShader(SHADER_CUBE);
		executeCommandList(&ad->commandList3d);

		bindShader(SHADER_QUAD);
		glDisable(GL_DEPTH_TEST);
		ortho(rect(0, -ws->currentRes.h, ws->currentRes.w, 0));
		blitFrameBuffers("3dMsaa", "3dNoMsaa", ad->cur3dBufferRes, GL_COLOR_BUFFER_BIT, GL_LINEAR);


		bindFrameBuffer("2d");
		glViewport(0,0, ws->currentRes.x, ws->currentRes.y);
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), vec4(1), rect(0,1,1,0), 
		         getFrameBuffer("3dNoMsaa")->colorSlot[0]->id);
		// executeCommandList(&ad->commandList2d);


		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
		glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);

		bindFrameBuffer("DebugMsaa");
		executeCommandList(&ad->commandList2d);

		static double tempTime = 0;
		timerInit(&ds->tempTimer);
			executeCommandList(&ds->commandListDebug, false, reload);
		tempTime += ds->dt;
		if(tempTime >= 1) {
			ds->debugRenderTime = timerStop(&ds->tempTimer);
			tempTime = 0;
		}

		blitFrameBuffers("DebugMsaa", "DebugNoMsaa", ws->currentRes, GL_COLOR_BUFFER_BIT, GL_LINEAR);


		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);

		bindFrameBuffer("2d");
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), vec4(1,1,1,ds->guiAlpha), rect(0,1,1,0), 
		         getFrameBuffer("DebugNoMsaa")->colorSlot[0]->id);

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);



		#if USE_SRGB 
			glEnable(GL_FRAMEBUFFER_SRGB);
		#endif 

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), vec4(1), rect(0,1,1,0), 
		         getFrameBuffer("2d")->colorSlot[0]->id);

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
			printf("GLError: %#8x\n", glGetError());
		}

		swapBuffers(sd);
		glFinish();

		timerStart(&ds->swapTimer);

		if(sd->vsyncTempTurnOff) {
			wglSwapIntervalEXT(1);
			sd->vsyncTempTurnOff = false;
		}
	}

	// Directx
	#if USE_DIRECT3D		
	{
		//clear scene
		Vec4 color = vec4(0.1f,1);
		sd->d3dDeviceContext->ClearRenderTargetView( sd->renderTargetView, color.e);
		
		sd->d3dDeviceContext->Draw( 4, 0 );

		//flip buffers
		sd->swapChain->Present(1,0);
	}
	#endif

	TIMER_INFO_COUNT = __COUNTER__;
	debugMain(ds, appMemory, ad, reload, isRunning, init, threadQueue);

	// debugUpdatePlayback(ds, appMemory);

	// Save game.
	if(*isRunning == false)
	{
		threadQueueComplete(theThreadQueue);

		MSTimer timer;
		timerInit(&timer);
		timerStart(&timer);

		DArray<VoxelMesh>* voxels = &ad->voxelData.voxels;

		char* buffer = getTArray(char, VOXEL_CACHE_SIZE);

		// Store all meshs that are cached.
		for(int i = 0; i < voxels->count; i++) {
			VoxelMesh* m = voxels->data + i;
			if(!m->stored && m->generated) {

				if(threadQueueFull(theThreadQueue)) threadQueueComplete(theThreadQueue);

				threadQueueAdd(theThreadQueue, storeMeshThreaded, m);
			}
		}

		// Using zlib.
		// for(int i = 0; i < 2; i++) {

		// 	uchar* data = i == 0 ? mesh->voxels : mesh->lighting;

		// 	int size;
		// 	uchar* result = stbi_zlib_compress(data, VOXEL_SIZE, &size, 0);

		// 	fwrite(&size, sizeof(int), 1, file);
		// 	fwrite(result, size * sizeof(char), 1, file);

			//  STBIW_FREE(result);
		// }

		threadQueueComplete(theThreadQueue);

		float dt = timerUpdate(&timer);

		char* saveFile = fillString("%s%s", SAVES_FOLDER, SAVE_STATE1);
		FILE* file = fopen(saveFile, "wb");
		if(file) {
			fwrite(&ad->inventory, sizeof(Inventory), 1, file);

			fwrite(ad->entityList.e, ad->entityList.size * sizeof(Entity), 1, file);

			fwrite(&ad->voxelSettings.startX, sizeof(int), 1, file);
			fwrite(&ad->voxelSettings.startY, sizeof(int), 1, file);
			fwrite(&ad->voxelSettings.startXMod, sizeof(int), 1, file);
			fwrite(&ad->voxelSettings.startYMod, sizeof(int), 1, file);

			//

			fwrite(&voxels->count, sizeof(int), 1, file);

			for(int i = 0; i < voxels->count; i++) {
				VoxelMesh* mesh = voxels->data + i;

				mesh->compressionStep = false;
				mesh->stored = true;

				fwrite(&mesh->coord, sizeof(Vec2i), 1, file);

				fwrite(&mesh->compressedVoxelsSize, sizeof(int), 1, file);
				fwrite(&mesh->compressedLightingSize, sizeof(int), 1, file);
				int compressedDataCountTotal = mesh->compressedVoxelsSize + mesh->compressedLightingSize;
				fwrite(mesh->compressedData, compressedDataCountTotal * (sizeof(uchar)*2), 1, file);

			}

		}

		float dt2 = timerUpdate(&timer);
		// printf("%f %f\n", dt, dt2);
	}

	// Save game settings.
	if(*isRunning == false) {
		GameSettings settings = {};
		settings.fullscreen = ws->fullscreen;
		settings.vsync = ws->vsync;
		settings.resolutionScale = ad->resolutionScale;
		settings.volume = ad->audioState.masterVolume;
		settings.mouseSensitivity = ad->mouseSensitivity;
		settings.fieldOfView = ad->fieldOfView;
		settings.viewDistance = ad->voxelSettings.viewDistance;

		char* file = Game_Settings_File;
		if(fileExists(file)) {
			writeDataToFile((char*)&settings, sizeof(GameSettings), file);
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

#pragma optimize( "", on ) 
