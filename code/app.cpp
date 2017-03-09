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
 * Split up main app.cpp into mutliple files.
 * Implement hotloading of text files -> textures, variables and so on.
 * Implement hotloading for shader.
 * Dropdown console.
 	* Ctrl backspace, Ctrl Delete.
 	* Ctrl Left/Right not jumping multiple spaces.
 	* Selection Left/Right not working propberly.
 	* Ctrl + a.
 	* Clipboard.
 	* Mouse selection.
 	* Mouse release.
 	* Cleanup.
 	* Scrollbar.
 	* Line wrap.
 	* Command history.
 	* Input cursor vertical scrolling.
 	* Command hint on tab.
 	* Lag when inputting.
 	* Add function with string/float as argument.
 	* Make adding functions more robust.
 	* Move evaluate() too appMain to have acces to functionality.
 	* Select inside console output.
 	* Fix result display lag by moving things.
	* Multiline text editing.

 - The Big Cleanup.

 - 3d animation system. (Search Opengl vertex skinning.)

 - Sound perturbation. (Whatever that is.) 



 - Using makros and defines to make templated vectors and hashtables and such.
 - Look at font drawing.
 - Clean up gui.

 * Fix putting Timer_blocks somewhere.
 * Move statistics out of debug.cpp.
 * Stop text in graph when pressing key.
 * Remove f key for stopping.
 - Goto editor when clicking in graph.
 - Sort graph texts.
 - Threading in timerblocks not working.
   - Use thread id information that's already in the timerinfo.
 - Drawing the timerinfo gets really slow.
 - Zoom bars need to be on top.

//-------------------------------------
//               BUGS
//-------------------------------------
- window operations only work after first frame
- look has to be negative to work in view projection matrix
- distance jumping collision bug, possibly precision loss in distances
- gpu fucks up at some point making swapBuffers alternates between time steps 
  which makes the game stutter, restart is required
- game input gets stuck when buttons are pressed right at the start
- sort key assert firing randomly
- hotload gets stuck sometimes, thread that won't complete

- draw line crashes
- text looks wrong after srgb conversion
- release build takes forever
- threadQueueComplete(ThreadQueue* queue) doesnt work
- threadqueue do next work from main thread bug

- Water lags behind one frame when drawing. Noticeable when pushing lower FPS on higher view distances. 
- Fonts look bad. (Whatever that means.)

*/

/*
	switching shader -> 550 ticks
	using namedBufferSubData vs uniforms for vertices -> 2400 ticks vs 400 ticks
*/



// Intrinsics.

#include <iacaMarks.h>
#include <xmmintrin.h>
#include <emmintrin.h>

// External.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl\gl.h>
// #include "external\glext.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG
#include "external\stb_image.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "external\stb_truetype.h"

#define STB_VOXEL_RENDER_IMPLEMENTATION
// #define STBVOX_CONFIG_LIGHTING_SIMPLE
#define STBVOX_CONFIG_FOG_SMOOTHSTEP
// #define STBVOX_CONFIG_MODE 0
#define STBVOX_CONFIG_MODE 1
#include "external\stb_voxel_render.h"

//

struct ThreadQueue;
struct GraphicsState;
struct DrawCommandList;
struct MemoryBlock;
struct DebugState;
struct Timer;
ThreadQueue* globalThreadQueue;
GraphicsState* globalGraphicsState;
DrawCommandList* globalCommandList;
MemoryBlock* globalMemory;
DebugState* globalDebugState;
Timer* globalTimer;

// Internal.

#include "rt_types.h"
#include "rt_timer.h"
#include "rt_misc.h"
#include "rt_math.h"
#include "rt_hotload.h"
#include "rt_misc_win32.h"
#include "rt_platformWin32.h"

#include "memory.h"
#include "openglDefines.h"
#include "userSettings.h"

#include "rendering.cpp"
#include "gui.cpp"

#include "entity.cpp"
#include "voxel.cpp"

#include "debug.cpp"



struct AppData {
	
	// General.

	SystemData systemData;
	Input input;
	WindowSettings wSettings;
	GraphicsState graphicsState;

	float dt;
	float time;

	DrawCommandList commandList2d;
	DrawCommandList commandList3d;

	bool updateFrameBuffers;

	// 

	bool captureMouse;
	bool fpsMode;

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

	int selectionRadius;
	bool blockSelected;
	Vec3 selectedBlock;
	Vec3 selectedBlockFaceDir;

	VoxelNode* voxelHash[1024];
	int voxelHashSize;
	uchar* voxelCache[8];
	uchar* voxelLightingCache[8];

	MakeMeshThreadedData threadData[256];

	bool reloadWorld;

	int voxelDrawCount;
	int voxelTriangleCount;

	int skyBoxId;
	Vec3 fogColor;

	// Particles.

	GLuint testBufferId;
	char* testBuffer;
	int testBufferSize;
};



// void debugMain(DebugState* ds, AppMemory* appMemory, AppData* ad, bool reload, bool* isRunning, bool init);
void debugMain(DebugState* ds, AppMemory* appMemory, AppData* ad, bool reload, bool* isRunning, bool init, ThreadQueue* threadQueue);
void debugUpdatePlayback(DebugState* ds, AppMemory* appMemory);

#pragma optimize( "", off )
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
	SystemData* systemData = &ad->systemData;
	HWND windowHandle = systemData->windowHandle;
	WindowSettings* ws = &ad->wSettings;

	globalThreadQueue = threadQueue;
	globalGraphicsState = &ad->graphicsState;
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

		// @AppInit.

		//
		// AppData.
		//

		getPMemory(sizeof(AppData));
		*ad = {};
		
		initSystem(systemData, ws, windowsData, vec2i(1920, 1080), true, true, true);
		windowHandle = systemData->windowHandle;

		loadFunctions();
		wglSwapIntervalEXT(1);

		initInput(&ad->input);

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
		int timerSlots = 5000;
		ds->timer->bufferSize = timerSlots;
		ds->timer->timerBuffer = (TimerSlot*)getPMemoryDebug(sizeof(TimerSlot) * timerSlots);

		int cycleCount = arrayCount(ds->timings);
		ds->savedBufferMax = timerSlots * cycleCount;
		for(int i = 0; i < cycleCount; i++) {
			ds->savedBuffer[i] = (TimerSlot*)getPMemoryDebug(sizeof(TimerSlot) * timerSlots);
		}

		ds->gui = getPStructDebug(Gui);
		// gui->init(rectCenDim(vec2(0,1), vec2(300,800)));
		// gui->init(rectCenDim(vec2(1300,1), vec2(300,500)));
		ds->gui->init(rectCenDim(vec2(1300,1), vec2(300, ws->currentRes.h)), 0);

		// ds->gui->cornerPos = 

		ds->gui2 = getPStructDebug(Gui);
		// ds->gui->init(rectCenDim(vec2(1300,1), vec2(400, ws->currentRes.h)), -1);
		ds->gui2->init(rectCenDim(vec2(1300,1), vec2(300, ws->currentRes.h)), 3);

		ds->input = getPStructDebug(Input);
		// ds->showMenu = false;
		ds->showMenu = true;
		ds->showStats = true;
		ds->showConsole = true;
		ds->showHud = true;
		ds->guiAlpha = 0.95f;

		//
		// Init Folder Handles.
		//

		initWatchFolders(systemData->folderHandles, ds->assets, &ds->assetCount);

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
			attachToFrameBuffer(FRAMEBUFFER_Debug, FRAMEBUFFER_SLOT_COLOR, GL_RGBA8, 0, 0);

			ad->updateFrameBuffers = true;
		}

		//
		// AppSetup.
		//

		// Entity.

		ad->captureMouse = false;
		ad->playerMode = true;
		ad->pickMode = true;
		ad->selectionRadius = 5;

		ad->entityList.size = 1000;
		ad->entityList.e = (Entity*)getPMemory(sizeof(Entity)*ad->entityList.size);
		for(int i = 0; i < ad->entityList.size; i++) ad->entityList.e[i].init = false;

		Vec3 startDir = normVec3(vec3(1,0,0));

		Entity player;
		Vec3 playerDim = vec3(0.8f, 0.8f, 1.8f);
		float camOff = playerDim.z*0.5f - playerDim.x*0.25f;
		initEntity(&player, ET_Player, vec3(0,0,40), startDir, playerDim, vec3(0,0,camOff));
		player.rot = vec3(M_2PI,0,0);
		player.playerOnGround = false;
		ad->player = addEntity(&ad->entityList, &player);

		Entity freeCam;
		initEntity(&freeCam, ET_Camera, vec3(35,35,32), startDir, vec3(0,0,0), vec3(0,0,0));
		ad->cameraEntity = addEntity(&ad->entityList, &freeCam);

		// Voxel.

		ad->skyBoxId = CUBEMAP_5;
		ad->fogColor = colorSRGB(vec3(0.43f,0.38f,0.44f));
		ad->bombFireInterval = 0.1f;
		ad->bombButtonDown = false;

		*ad->blockMenu = {};
		ad->blockMenuSelected = 0;

		ad->voxelHashSize = sizeof(arrayCount(ad->voxelHash));
		for(int i = 0; i < ad->voxelHashSize; i++) {
			ad->voxelHash[i] = (VoxelNode*)getPMemory(sizeof(VoxelNode));
			*ad->voxelHash[i] = {};
		}

		for(int i = 0; i < arrayCount(ad->threadData); i++) {
			ad->threadData[i] = {};
		} 

		for(int i = 0; i < arrayCount(ad->voxelCache); i++) {
			ad->voxelCache[i] = (uchar*)getPMemory(sizeof(uchar)*VOXEL_CACHE_SIZE);
			ad->voxelLightingCache[i] = (uchar*)getPMemory(sizeof(uchar)*VOXEL_CACHE_SIZE);
		}

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

		for(int i = 0; i < 8; i++) {
			voxelCache[i] = ad->voxelCache[i];
			voxelLightingCache[i] = ad->voxelLightingCache[i];
		}

		// Load voxel meshes around the player at startup.
		{
			Vec2i pPos = coordToMesh(ad->player->pos);
			for(int y = -1; y < 2; y++) {
				for(int x = -1; x < 2; x++) {
					Vec2i coord = pPos - vec2i(x,y);

					VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
					makeMesh(m, ad->voxelHash, ad->voxelHashSize);
				}
			}

			threadQueueComplete(globalThreadQueue);

			// Push the player up until he is right above the ground.

			Entity* player = ad->player;

			while(collisionVoxelWidthBox(ad->voxelHash, ad->voxelHashSize, player->pos, player->dim)) {
				player->pos.z += 2;
			}
		}	
	}

	// @AppStart.

	TIMER_BLOCK_BEGIN_NAMED(reload, "Reload");

	if(reload) {
		loadFunctions();
		SetWindowLongPtr(systemData->windowHandle, GWLP_WNDPROC, (LONG_PTR)mainWindowCallBack);

		if(HOTRELOAD_SHADERS) {
			loadShaders();
		}
	}

	TIMER_BLOCK_END(reload);




	// Update timer.
	{
		LARGE_INTEGER counter;
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency); 

		if(init) {
			QueryPerformanceCounter(&counter);
			ds->lastTimeStamp = counter.QuadPart;
			ds->dt = 1/(float)60;
		} else {
			QueryPerformanceCounter(&counter);
			float timeStamp = counter.QuadPart;
			ds->dt = (timeStamp - ds->lastTimeStamp);
			ds->dt *= 1000000;
			ds->dt = ds->dt/frequency.QuadPart;
			ds->dt = ds->dt / 1000000;
			ds->dt = clampMax(ds->dt, 1/(float)20);

			ds->lastTimeStamp = timeStamp;

			ds->time += ds->dt;
		}
	}

	clearTMemory();

	// Allocate drawCommandlist.

	int clSize = kiloBytes(1000);
	drawCommandListInit(&ad->commandList3d, (char*)getTMemory(clSize), clSize);
	drawCommandListInit(&ad->commandList2d, (char*)getTMemory(clSize), clSize);
	globalCommandList = &ad->commandList3d;

	// Hotload changed files.

	reloadChangedFiles(systemData->folderHandles, ds->assets, ds->assetCount);

	// Update input.
	{
		TIMER_BLOCK_NAMED("Input");
		updateInput(ds->input, isRunning, windowHandle);

		ad->input = *ds->input;
		if(ds->console.isActive) {
			memSet(ad->input.keysPressed, 0, sizeof(ad->input.keysPressed));
			memSet(ad->input.keysDown, 0, sizeof(ad->input.keysDown));
		}

		ad->dt = ds->dt;
		ad->time = ds->time;
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

		if(ds->playbackInput) {
			ad->input = ds->recordedInput[ds->playbackIndex];
			ds->playbackIndex = (ds->playbackIndex+1)%ds->inputIndex;
			if(ds->playbackIndex == 0) ds->playbackSwapMemory = true;
		}
	} 

    if(input->keysPressed[KEYCODE_ESCAPE]) {
    	*isRunning = false;
    }

	if(input->keysPressed[KEYCODE_F1]) {
		int mode;
		if(ws->fullscreen) mode = WINDOW_MODE_WINDOWED;
		else mode = WINDOW_MODE_FULLBORDERLESS;
		setWindowMode(windowHandle, ws, mode);
	}

	if(input->keysPressed[KEYCODE_F2]) {
		static bool switchMonitor = false;

		setWindowMode(windowHandle, ws, WINDOW_MODE_WINDOWED);

		if(!switchMonitor) setWindowProperties(windowHandle, 1, 1, 1920, 0);
		else setWindowProperties(windowHandle, 1920, 1080, -1920, 0);
		switchMonitor = !switchMonitor;

		setWindowMode(windowHandle, ws, WINDOW_MODE_FULLBORDERLESS);
	}


	if(windowSizeChanged(windowHandle, ws)) {
		updateResolution(windowHandle, ws);
		ad->updateFrameBuffers = true;
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
		setDimForFrameBufferAttachmentsAndUpdate(FRAMEBUFFER_Debug, ws->currentRes.w, ws->currentRes.h);
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
		glClearColor(0,0,0,0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		for(int i = 0; i < arrayCount(gs->frameBuffers); i++) {
			bindFrameBuffer(i);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		}
	}

	// Setup opengl.
	{
		// glDepthRange(-1.0,1.0);
		glFrontFace(GL_CW);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glDisable(GL_SCISSOR_TEST);
		glEnable(GL_LINE_SMOOTH);
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



	// @AppLoop.

	if(input->keysPressed[KEYCODE_F3]) {
		ad->captureMouse = !ad->captureMouse;
	}

	ad->fpsMode = ad->captureMouse && windowHasFocus(windowHandle);
	captureMouse(windowHandle, ad->fpsMode);

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
					float turnRate = ad->dt*0.3f;
					e->rot.y += turnRate * input->mouseDeltaY;
					e->rot.x += turnRate * input->mouseDeltaX;

					float margin = 0.00001f;
					clamp(&e->rot.y, -M_PI+margin, M_PI-margin);

					e->rot.x = modFloat(e->rot.x, (float)M_PI*4);
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
						collision = collisionVoxelWidthBox(ad->voxelHash, ad->voxelHashSize, nPos, pSize, &minDistance, &collisionBox);

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
							uchar* blockType = getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, gp);

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
					float turnRate = ad->dt*0.3f;
					e->rot.y += turnRate * input->mouseDeltaY;
					e->rot.x += turnRate * input->mouseDeltaX;

					float margin = 0.00001f;
					clamp(&e->rot.y, -M_PI+margin, M_PI-margin);
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
								uchar* block = getBlockFromVoxel(ad->voxelHash, ad->voxelHashSize, coord);

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

									*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, pos+dir*rad) = 0; 
									*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos+dir*rad) = globalLumen; 
									*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, pos-dir*rad) = 0; 
									*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos-dir*rad) = globalLumen; 

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

										*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize,    pos + vec3(off2,0, z)) = 0; 
										*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos + vec3(off2,0, z)) = globalLumen; 
										*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize,    pos + vec3(off2,0,-z)) = 0; 
										*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos + vec3(off2,0,-z)) = globalLumen; 
										*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize,    pos - vec3(off2,0, z)) = 0; 
										*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos - vec3(off2,0, z)) = globalLumen; 
										*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize,    pos - vec3(off2,0,-z)) = 0; 
										*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos - vec3(off2,0,-z)) = globalLumen; 
									}
								}
							}
						}

						for(int i = 0; i < updateMeshListSize; i++) {
							VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, updateMeshList[i]);
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

		for(int i = 0; i < ad->selectionRadius; i++) {
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

				uchar* blockType = getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, block);
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
				VoxelMesh* vm = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordToMesh(intersectionBox));

				uchar* block = getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, intersectionBox);
				uchar* lighting = getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, intersectionBox);

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
							VoxelMesh* edgeMesh = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, mc);
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
						uchar* sideBlockType = getBlockFromVoxel(ad->voxelHash, ad->voxelHashSize, voxelSideBlock);
						uchar* sideBlockLighting = getLightingFromVoxel(ad->voxelHash, ad->voxelHashSize, voxelSideBlock);

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
					VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
					m->upToDate = false;
					m->meshUploaded = false;
					m->generated = false;

					makeMesh(m, ad->voxelHash, ad->voxelHashSize);
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
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);

				if(!m->meshUploaded) {
					makeMesh(m, ad->voxelHash, ad->voxelHashSize);
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
		setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, ad->fogColor);

		// TIMER_BLOCK_NAMED("D World");
		// draw world without water
		{
			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
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
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
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

			setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, ad->fogColor, vec3(0,0,WATER_LEVEL_HEIGHT*2 + 0.01f), vec3(1,1,-1));
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CLIPPLANE, true);
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE1, 0,0,-1,WATER_LEVEL_HEIGHT);

			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
				drawVoxelMesh(m, 2);
			}
			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
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
			drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), rect(0,1,1,0), vec4(1,1,1,reflectionAlpha), 
			         getFrameBuffer(FRAMEBUFFER_Reflection)->colorSlot[0]->id);

			glEnable(GL_DEPTH_TEST);

			// 	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			// 	glBlendEquation(GL_FUNC_ADD);
		}

		// draw water
		{
			setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, ad->fogColor);
			pushUniform(SHADER_VOXEL, 1, VOXEL_UNIFORM_ALPHATEST, 0.5f);

			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
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
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), rect(0,1,1,0), vec4(1), 
		         getFrameBuffer(FRAMEBUFFER_3dNoMsaa)->colorSlot[0]->id);
		executeCommandList(&ad->commandList2d);


		bindFrameBuffer(FRAMEBUFFER_Debug);
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);

		// Skipping strings for now when reloading because hardcoded ones get a new memory address after changing the dll.
		executeCommandList(&ds->commandListDebug, false, reload);


		bindFrameBuffer(FRAMEBUFFER_2d);
		// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		// glBlendEquation(GL_FUNC_ADD);
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), rect(0,1,1,0), vec4(1,1,1,ds->guiAlpha), 
		         getFrameBuffer(FRAMEBUFFER_Debug)->colorSlot[0]->id);


		#if USE_SRGB 
			glEnable(GL_FRAMEBUFFER_SRGB);
		#endif 

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		drawRect(rect(0, -ws->currentRes.h, ws->currentRes.w, 0), rect(0,1,1,0), vec4(1), 
		         getFrameBuffer(FRAMEBUFFER_2d)->colorSlot[0]->id);

		#if USE_SRGB
			glDisable(GL_FRAMEBUFFER_SRGB);
		#endif
	}



	// Swap window background buffer.
	{
		TIMER_BLOCK_NAMED("Swap");
		swapBuffers(&ad->systemData);
		glFinish();

		if(init) {
			showWindow(windowHandle);
			GLenum glError = glGetError(); printf("GLError: %i\n", glError);
		}
	}

	debugMain(ds, appMemory, ad, reload, isRunning, init, threadQueue);

	debugUpdatePlayback(ds, appMemory);

	// @AppEnd.
}



void debugMain(DebugState* ds, AppMemory* appMemory, AppData* ad, bool reload, bool* isRunning, bool init, ThreadQueue* threadQueue) {
	// @DebugStart.

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


	if(ds->showHud) {
		int fontSize = 18;
		int pi = 0;
		// Vec4 c = vec4(1.0f,0.2f,0.0f,1);
		Vec4 c = vec4(1.0f,0.4f,0.0f,1);
		Vec4 c2 = vec4(0,0,0,1);
		Font* font = getFont(FONT_CONSOLAS, fontSize);
		int sh = 1;
		Vec2 offset = vec2(6,6);
		Vec2i ali = vec2i(1,1);

		Vec2 tp = vec2(ad->wSettings.currentRes.x, 0) - offset;

		#define PVEC3(v) v.x, v.y, v.z
		#define PVEC2(v) v.x, v.y
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
	}

	if(ds->showMenu) {
		int fontSize = 18;

		bool initSections = false;

		Gui* gui = ds->gui;
		// gui->start(gInput, getFont(FONT_CONSOLAS, fontSize), ws->currentRes);
		gui->start(ds->gInput, getFont(FONT_CALIBRI, fontSize), ws->currentRes);

		if(gui->switcher("Record", &ds->recordingInput)) {
			if(ds->recordingInput) {
				for(int i = 0; i < pMemory->index; i++) {
					if(debugMemory->index < i) getExtendibleMemoryArray(pMemory->slotSize, debugMemory);

					memCpy(debugMemory->arrays[i].data, pMemory->arrays[i].data, pMemory->slotSize);
				}

				ds->recordingInput = true;
				ds->inputIndex = 0;
			} else {
				ds->recordingInput = false;
			}
		}
		if(gui->button("Playback")) ds->playbackStart = true;

		static bool sectionGuiSettings = false;
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

	if(reload) {
		for(int i = 0; i < arrayCount(ds->timer->timerInfos); i++) ds->timer->timerInfos[i].initialised = false;
		return;
	}

	ds->timer->timerInfoCount = __COUNTER__;

	int fontHeight = 18;
	Timer* timer = ds->timer;
	int cycleCount = arrayCount(ds->timings);
	int bufferIndex = timer->bufferIndex;
	timer->bufferIndex = 0;

	if(ds->setPause) {
		ds->lastCycleIndex = ds->cycleIndex;
		ds->cycleIndex = mod(ds->cycleIndex-1, arrayCount(ds->timings));

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

			// Save current timerBuffer.
			{
				memCpy(ds->savedBuffer[cycleIndex], timer->timerBuffer, bufferIndex * sizeof(TimerSlot));
				ds->savedBufferCounts[cycleIndex] = bufferIndex;
			}

			ds->cycleIndex = newCycleIndex;

			// Collate timing buffer.

			for(int threadIndex = 0; threadIndex < threadQueue->threadCount; threadIndex++) {
				TimerSlot* slots[16];
				int index = 0;

				for(int i = 0; i < bufferIndex; ++i) {
					TimerSlot* slot = ds->savedBuffer[cycleIndex] + i;

					if(slot->threadId != threadQueue->threadIds[threadIndex]) continue;

					if(slot->type == TIMER_TYPE_BEGIN) {
						slots[index] = slot;
						index++;
					} else {
						index--;
						if(index < 0) index = 0; // @Hack, to keep things running.

						Timings* timing = timings + slots[index]->timerIndex;
						uint slotSize = slot->cycles - slots[index]->cycles;
						timing->cycles += slotSize;
						timing->hits++;

						slots[index]->size = slotSize;
					}
				}
			}

			for(int i = 0; i < timer->timerInfoCount; i++) {
				Timings* t = timings + i;
				t->cyclesOverHits = t->hits > 0 ? (u64)(t->cycles/t->hits) : 0; 
			}

			for(int timerIndex = 0; timerIndex < timer->timerInfoCount; timerIndex++) {
				Statistic* stat = statistics + timerIndex;
				beginStatistic(stat);

				for(int i = 0; i < arrayCount(ds->timings); i++) {
					Timings* t = &ds->timings[i][timerIndex];
					updateStatistic(stat, t->cyclesOverHits);
				}

				endStatistic(stat);
			}
		}
	}

	TimerSlot* graphTimerBuffer = ds->savedBuffer[cycleIndex];
	int graphBufferIndex = ds->savedBufferCounts[cycleIndex];

	//
	// Draw timing info.
	//

	if(ds->showStats) 
	{
		static int highlightedIndex = -1;
		Vec4 highlightColor = vec4(1,1,1,0.05f);

		// float cyclesPerFrame = (float)((3.5f*((float)1/60))*1024*1024*1024);
		float cyclesPerFrame = (float)((3.5f*((float)1/60))*1000*1000*1000);
		fontHeight = 18;
		Vec2 textPos = vec2(550, -fontHeight);
		int infoCount = timer->timerInfoCount;

		Gui* gui = ds->gui2;
		gui->start(ds->gInput, getFont(FONT_CALIBRI, fontHeight), ws->currentRes);

		gui->label("App Statistics", 1, gui->colors.sectionColor, vec4(0,0,0,1));

		gui->div(vec2(0.2f,0));
		if(gui->switcher("Freeze", &ds->noCollating)) {
			if(ds->noCollating) ds->setPause = true;
			else ds->setPlay = true;
		}
		gui->slider(&ds->cycleIndex, 0, cycleCount-1);

		{
			int barWidth = 1;
			int barCount = arrayCount(ds->timings);
			float sectionWidths[] = {0,0,0,0,0,0,0,0, barWidth*barCount};

			char* headers[] = {"File", "Function", "Description", "Cycles", "Hits", "C/H", "Avg. Cycl.", "Total Time", "Graphs"};
			gui->div(sectionWidths, arrayCount(sectionWidths));

			float textSectionEnd;

			for(int i = 0; i < arrayCount(sectionWidths); i++) {
				// @Hack: Get the end of the text region by looking at last region.
				if(i == arrayCount(sectionWidths)-1) textSectionEnd = gui->getCurrentRegion().max.x;

				gui->label(headers[i],1, vec4(0,0,0,0.3f), vec4(0,0,0,1));
			}

			for(int i = 0; i < infoCount; i++) {
				TimerInfo* tInfo = timer->timerInfos + i;
				Timings* timing = timings + i;

				if(!tInfo->initialised) continue;

				float cycleCountPercent = (float)timing->cycles/cyclesPerFrame;
				char * percentString = getTStringDebug(50);
				percentString = floatToStr(percentString, cycleCountPercent*100, 3);

				int debugStringSize = 50;
				char* buffer = 0;

				gui->div(sectionWidths, arrayCount(sectionWidths)); 

				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s", tInfo->file + 21);
				gui->label(buffer,0);

				if(highlightedIndex == i) {
					Rect r = gui->getCurrentRegion();
					Rect line = rect(r.min, vec2(textSectionEnd,r.min.y + fontHeight));
					dcRect(line, highlightColor);
				}

				debugStringSize = 30;
				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s", tInfo->function);
				gui->label(buffer,0);

				debugStringSize = 30;
				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s", tInfo->name);
				gui->label(buffer,0);

				debugStringSize = 30;
				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64uc", timing->cycles);
				// if(timing->cycles < 1000) 
				// 	_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64uc", timing->cycles);
				// else if(timing->cycles < 1000000) 
				// 	_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64u,%0.3I64uc", (u64)(timing->cycles/1000), (u64)(timing->cycles%1000));
				// else 
				// 	_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64u,%0.3I64u,%0.3I64uc", (u64)(timing->cycles/1000000), (u64)(timing->cycles/1000), (u64)(timing->cycles%1000));

				gui->label(buffer,2);

				debugStringSize = 30;
				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%u", timing->hits);
				gui->label(buffer,2);

				debugStringSize = 30;
				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64u", timing->cyclesOverHits);
				gui->label(buffer,2);

				debugStringSize = 30;
				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%.0fc", statistics[i].avg);
				gui->label(buffer,2);

				debugStringSize = 30;
				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s%%", percentString);
				gui->label(buffer,2);

				// Bar graphs.

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
					dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,1-colorOffset,0,1));

					xOffset += barWidth;
				}
			}
		}

		{

			gui->heightPush(1.0f);
			gui->empty();
			Rect cyclesRect = gui->getCurrentRegion();
			gui->empty();
			Rect headerRect = gui->getCurrentRegion();
			headerRect.max.y = cyclesRect.max.y;
			gui->heightPop();


			float lineHeightOffset = 1.2;
			float lineHeight = fontHeight * lineHeightOffset;
			float heightP = 3*lineHeight +  2*lineHeight*(threadQueue->threadCount-1);

			gui->heightPush(heightP);
			gui->empty();

			Rect bgRect = gui->getCurrentRegion();
			Vec2 startPos = rectGetUL(bgRect);

			Vec2 dragDelta = vec2(0,0);
			gui->drag(bgRect, &dragDelta, vec4(0,0,0,0));

			float graphWidth = rectGetDim(bgRect).w;

			static Vec2 cp = vec2(graphWidth/2,0);
			static float zoom = 1;

			cp.x -= dragDelta.x * ((graphWidth*zoom)/graphWidth);

			gui->heightPop();

			if(gui->input.mouseWheel) {
				float oldWidth = graphWidth*zoom;

				float wheel = gui->input.mouseWheel;

				float offset = wheel < 0 ? 1.1f : 0.9f;
				if(!input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.2f : 0.8f;
				if(input->keysDown[KEYCODE_SHIFT] && input->keysDown[KEYCODE_CTRL]) 
					offset = wheel < 0 ? 1.4f : 0.6f;

				zoom *= offset;
				zoom = clampMax(zoom, 4.0f);

				float addedWidth = graphWidth*zoom - oldWidth;
				float mouseZoomOffset = mapRange(input->mousePos.x, bgRect.min.x, bgRect.max.x, -1, 1);
				cp.x -= (addedWidth/2)*mouseZoomOffset;
			}

			// cp.x = clamp(cp.x, (graphWidth*zoom)/2, graphWidth - (graphWidth*zoom)/2);
			cp.x = clampMin(cp.x, (graphWidth*zoom)/2);

			if(true) {
				u64 baseCycleCount = graphTimerBuffer[0].cycles;
				u64 startCycleCount = 0;
				u64 endCycleCount = cyclesPerFrame;

				float orthoLeft = cp.x - (graphWidth*zoom)/2;
				float orthoRight = cp.x + (graphWidth*zoom)/2;

				// Cycles.
				{
					// dcRect(cyclesRect, vec4(0,1,1,0.1f));

				}

				// Header.
				{
					Vec2 cyclesDim = rectGetDim(cyclesRect);

					dcRect(headerRect, vec4(1,1,1,0.1f));

					float startLine = mapRange(0, orthoLeft, orthoRight, headerRect.min.x, headerRect.max.x);
					float endLine = mapRange(graphWidth, orthoLeft, orthoRight, headerRect.min.x, headerRect.max.x);

					Vec2 bgCen = rectGetCen(headerRect);
					Vec2 bgDim = rectGetDim(headerRect);
					float g = 0.7f;
					dcRect(rectCenDim(vec2(startLine + 1, bgCen.y), vec2(4,bgDim.h)), vec4(g,g,g,1));
					dcRect(rectCenDim(vec2(endLine - 1, bgCen.y), vec2(3,bgDim.h)), vec4(g,g,g,1));

					g = 0.7f;
					float div = 4;
					float cyclesInWidth = mapRange(cyclesPerFrame*div, startCycleCount, endCycleCount, 0, graphWidth);
					uint cyc = cyclesPerFrame*div;
					float heightMod = 1.3f;
					float heightSub = 0.15f;

					float zoomBarInterval = cyclesInWidth;
					float orthoWidth = graphWidth*zoom;
					while(zoomBarInterval/orthoWidth > 0.3f) {
						zoomBarInterval /= div;
						heightMod -= heightSub;
						cyc /= div;
					}

					heightMod = clampMax(heightMod, 1);

					float pos = roundMod(orthoLeft, zoomBarInterval);
					while(pos < orthoRight) {
						float p = mapRange(pos, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);

						float h = bgDim.h*heightMod;
						dcRect(rectCenDim(vec2(p, headerRect.min.y + h/2), vec2(3,h)), vec4(g,g,g,1));

						Vec2 textPos = vec2(p,cyclesRect.min.y + cyclesDim.h/2);
						// dcText("abc", gui->font, textPos, gui->colors.textColor, vec2i(0,0), 0, gui->settings.textShadow, gui->colors.shadowColor);
						// dcText(fillString("%i", cyc), gui->font, textPos, gui->colors.textColor, vec2i(0,0), 0, gui->settings.textShadow, gui->colors.shadowColor);

						pos += zoomBarInterval;
					}

					pos = roundMod(orthoLeft, zoomBarInterval);
					zoomBarInterval /= div;
					heightMod -= heightSub;
					int index = 0;
					while(pos < orthoRight) {
						if((index%(int)div) != 0) {
							float p = mapRange(pos, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);

							float h = bgDim.h*heightMod;
							dcRect(rectCenDim(vec2(p, headerRect.min.y + h/2), vec2(1,h)), vec4(g,g,g,1));
						}

						pos += zoomBarInterval;
						index++;
					}
				}

				dcRect(bgRect, vec4(1,1,1,0.2f));

				bool mouseHighlight = false;
				Rect hRect;
				Vec4 hc;
				char* hText;

				startPos -= vec2(0, lineHeight);
				for(int threadIndex = 0; threadIndex < threadQueue->threadCount; threadIndex++) {
					int index = 0;

					// Horizontal lines to distinguish thread bars.
					if(threadIndex > 0) {
						Vec2 p = startPos;
						float g = 0.8f;
						dcRect(rect(p, vec2(bgRect.max.x, p.y+1)), vec4(g,g,g,1));
					}

					for(int i = 0; i < graphBufferIndex; ++i) {
						TimerSlot* slot = graphTimerBuffer + i;
						if(slot->threadId != threadQueue->threadIds[threadIndex]) continue;

						if(slot->type == TIMER_TYPE_BEGIN) {

							Timings* t = timings + slot->timerIndex;
							TimerInfo* tInfo = timer->timerInfos + slot->timerIndex;

							float barLeft = mapRange(slot->cycles - baseCycleCount, startCycleCount, endCycleCount, 0, graphWidth);
							float barRight = mapRange(slot->cycles - baseCycleCount + (u64)slot->size, startCycleCount, endCycleCount, 0, graphWidth);

							barLeft = mapRange(barLeft, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);
							barRight = mapRange(barRight, orthoLeft, orthoRight, bgRect.min.x, bgRect.max.x);

							// Skip nonvisible bars.
							if(!(barRight < bgRect.min.x || barLeft > bgRect.max.x)) {
								if(barRight - barLeft < 1) barRight = barLeft + 1;

								float y = startPos.y+index*-lineHeight;
								Rect r = rect(vec2(barLeft,y), vec2(barRight, y + lineHeight));

								float cOff = slot->timerIndex/(float)timer->timerInfoCount;
								Vec4 c = vec4(1-cOff, 0, cOff, 1);
								char* text = fillString("%s %s", tInfo->function, tInfo->name);

								if(gui->getMouseOver(gui->input.mousePos, r)) {
									mouseHighlight = true;
									hRect = r;
									hc = c;
									hText = text;
									highlightedIndex = slot->timerIndex;
								} else {
									float g = 0.1f;
									gui->drawRect(r, vec4(g,g,g,1));

									if(rectGetDim(r).w > 3) {
										if(barLeft < bgRect.min.x) r.min.x = bgRect.min.x;
										gui->drawTextBox(rect(r.min+vec2(1,1), r.max-vec2(1,1)), text, c);
									}
								}
							}

							index++;
						}

						if(slot->type == TIMER_TYPE_END) {
							index--;
						}
					}

					startPos.y -= lineHeight*2;

				}

				if(mouseHighlight) {
					float tw = getTextDim(hText, gui->font).w + 2;
					if(tw > rectGetDim(hRect).w) hRect.max.x = hRect.min.x + tw;

					float g = 0.8f;
					gui->drawRect(hRect, vec4(g,g,g,1));

					if(hRect.min.x < bgRect.min.x) hRect.min.x = bgRect.min.x;
					gui->drawTextBox(rect(hRect.min+vec2(1,1), hRect.max-vec2(1,1)), hText, hc);
				} else {
					highlightedIndex = -1;
				}

			}

			gui->div(0.1f,0); 

			if(gui->button("Reset")) {
				cp = vec2(graphWidth/2,0);
				zoom = 1;
			}

			gui->label(fillString("Cam: %f, Zoom: %f",cp.x, zoom));
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

	if(*isRunning == false) {
		guiSave(ds->gui, 2, 0);
		if(globalDebugState->gui2) guiSave(globalDebugState->gui2, 2, 3);
	}
}

void debugUpdatePlayback(DebugState* ds, AppMemory* appMemory) {
	if(ds->playbackStart) {
		ds->playbackStart = false;
		ds->playbackInput = true;
		ds->playbackIndex = 0;

		ds->playbackSwapMemory = true;
	}

	if(ds->playbackSwapMemory) {
		ds->playbackSwapMemory = false;

		ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[1];
		ExtendibleMemoryArray* pMemory = globalMemory->pMemory;

		for(int i = 0; i < pMemory->index; i++) {
			memCpy(pMemory->arrays[i].data, debugMemory->arrays[i].data, pMemory->slotSize);
		}
	}
}

#pragma optimize( "", on ) 
