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


Changing course for now:
 * Split up main app.cpp into mutliple files.
 - Implement hotloading of text files -> shaders, textures, variables and so on.
 - Dropdown console.
 - 3d animation system. (Search Opengl vertex skinning)
 - Sound perturbation (Whatever that is). 

 - Should test hot code reloading again!


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
*/

/*
	switching shader -> 550 ticks
	using namedBufferSubData vs uniforms for vertices -> 2400 ticks vs 400 ticks
*/



// #pragma optimize( "", off )
#pragma optimize( "", on )

#include <iacaMarks.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl\gl.h>
// #include "glext.h"

#include "rt_misc.h"
#include "rt_math.h"
#include "rt_hotload.h"
#include "rt_misc_win32.h"
#include "rt_platformWin32.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG

#include "stb_image.h"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#define STB_VOXEL_RENDER_IMPLEMENTATION
// #define STBVOX_CONFIG_LIGHTING_SIMPLE
#define STBVOX_CONFIG_FOG_SMOOTHSTEP
// #define STBVOX_CONFIG_MODE 0
#define STBVOX_CONFIG_MODE 1
#include "stb_voxel_render.h"

#define USE_SRGB 1
 
// #pragma optimize( "", off )
// #pragma optimize( "", on )

#include "openglDefines.cpp"
#include "memory.cpp"

#include "rendering.cpp"
#include "gui.cpp"
#include "debug.cpp"

#include "entity.cpp"
#include "voxel.cpp"


ThreadQueue* globalThreadQueue;
GraphicsState* globalGraphicsState;
DrawCommandList* globalCommandList;
MemoryBlock* globalMemory;
DebugState* globalDebugState;



struct AppData {
	bool showHud;
	bool updateFrameBuffers;
	float guiAlpha;

	uint cubemapTextureId[16];
	uint cubemapSamplerId;
	int cubeMapCount;
	int cubeMapDrawIndex;

	SystemData systemData;
	Input input;
	WindowSettings wSettings;

	GraphicsState graphicsState;
	DrawCommandList commandList2d;
	DrawCommandList commandList3d;

	LONGLONG lastTimeStamp;
	float dt;
	float time;

	bool* treeNoise;

	EntityList entityList;
	Entity* player;
	Entity* cameraEntity;

	Camera activeCam;

	uint samplers[16];
	uint frameBuffers[16];
	uint renderBuffers[16];
	uint frameBufferTextures[16];

	float aspectRatio;
	float fieldOfView;
	float nearPlane;
	float farPlane;

	Vec2i curRes;
	int msaaSamples;
	Vec2i fboRes;
	bool useNativeRes;

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

	GLuint voxelSamplers[3];
	GLuint voxelTextures[3];

	GLuint testBufferId;
	char* testBuffer;
	int testBufferSize;
};

#pragma optimize( "", off )

extern "C" APPMAINFUNCTION(appMain) {

	if(init) {
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

	MemoryBlock gMemory = {};
	gMemory.pMemory = &appMemory->extendibleMemoryArrays[0];
	gMemory.tMemory = &appMemory->memoryArrays[0];
	gMemory.dMemory = &appMemory->extendibleBucketMemories[0];
	gMemory.pDebugMemory = &appMemory->memoryArrays[1];
	gMemory.tMemoryDebug = &appMemory->memoryArrays[2];
	gMemory.debugMemory = &appMemory->extendibleMemoryArrays[1];

	globalMemory = &gMemory;

	AppData* ad = (AppData*)getBaseExtendibleMemoryArray(globalMemory->pMemory);
	Input* input = &ad->input;
	SystemData* systemData = &ad->systemData;
	HWND windowHandle = systemData->windowHandle;
	WindowSettings* wSettings = &ad->wSettings;

	globalThreadQueue = threadQueue;
	globalGraphicsState = &ad->graphicsState;
	threadData = ad->threadData;

	DebugState* ds = (DebugState*)globalMemory->pDebugMemory->data;
	globalDebugState = ds;

	treeNoise = ad->treeNoise;

	for(int i = 0; i < 8; i++) {
		voxelCache[i] = ad->voxelCache[i];
		voxelLightingCache[i] = ad->voxelLightingCache[i];
	}

	if(init) {
		getPMemoryDebug(sizeof(DebugState));
		*ds = {};

		ds->inputCapacity = 600;
		ds->recordedInput = (Input*)getPMemoryDebug(sizeof(Input) * ds->inputCapacity);

		int timerSlots = 10000;
		globalDebugState->bufferSize = timerSlots;
		globalDebugState->timerBuffer = (TimerSlot*)getPMemoryDebug(sizeof(TimerSlot) * timerSlots);
		globalDebugState->savedTimerBuffer	= (TimerSlot*)getPMemoryDebug(sizeof(TimerSlot) * timerSlots);
		globalDebugState->cycleIndex = 0;

		ds->gui = getPStructDebug(Gui);
		// gui->init(rectCenDim(vec2(0,1), vec2(300,800)));
		// gui->init(rectCenDim(vec2(1300,1), vec2(300,500)));
		ds->gui->init(rectCenDim(vec2(1300,1), vec2(300, wSettings->currentRes.h)), 0);

		ds->gui2 = getPStructDebug(Gui);
		// ds->gui->init(rectCenDim(vec2(1300,1), vec2(400, wSettings->currentRes.h)), -1);
		ds->gui2->init(rectCenDim(vec2(1300,1), vec2(300, wSettings->currentRes.h)), 3);

		ds->input = getPStructDebug(Input);



		getPMemory(sizeof(AppData));
		*ad = {};

		ad->dt = 1/(float)60;

		initInput(&ad->input);
		
		wSettings->res.w = 1920;
		wSettings->res.h = 1080;
		wSettings->fullscreen = false;
		wSettings->fullRes.x = GetSystemMetrics(SM_CXSCREEN);
		wSettings->fullRes.y = GetSystemMetrics(SM_CYSCREEN);
		wSettings->style = (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU  | WS_MINIMIZEBOX  | WS_VISIBLE);
		initSystem(systemData, windowsData, 0, 0,0,0,0);

		// DEVMODE devMode;
		// int index = 0;
		// int dW = 0, dH = 0;
		// Vec2i resolutions[90] = {};
		// int resolutionCount = 0;
		// while(bool result = EnumDisplaySettings(0, index, &devMode)) {
		// 	Vec2i nRes = vec2i(devMode.dmPelsWidth, devMode.dmPelsHeight);
		// 	if(resolutionCount == 0 || resolutions[resolutionCount-1] != nRes) {
		// 		resolutions[resolutionCount++] = nRes;
		// 	}
		// 	index++;
		// }

		loadFunctions();

		// for(int i = 0; i < GL_NUM_EXTENSIONS; i++) {
		// 	char* s = (char*)glGetStringi(GL_EXTENSIONS, i);
		// 	printf("%s\n", s);
		// }

		// typedef int wglGetSwapIntervalEXTFunction(void);
		// wglGetSwapIntervalEXTFunction* wglGetSwapIntervalEXT;
		// typedef int wglSwapIntervalEXTFunction(void);
		// wglSwapIntervalEXTFunction* wglSwapIntervalEXT;

		// gl##name = (name##Function*)wglGetProcAddress("gl" #name);
		wglGetSwapIntervalEXT = (wglGetSwapIntervalEXTFunction*)wglGetProcAddress("wglGetSwapIntervalEXT");
		wglSwapIntervalEXT = (wglSwapIntervalEXTFunction*)wglGetProcAddress("wglSwapIntervalEXT");
		wglSwapIntervalEXT(1);

		int interval = wglGetSwapIntervalEXT();

		// @setup

		// ad->fieldOfView = 55;
		ad->fieldOfView = 60;
		ad->msaaSamples = 4;
		ad->fboRes = vec2i(0, 120);
		ad->useNativeRes = true;
		ad->nearPlane = 0.1f;
		// ad->farPlane = 2000;
		ad->farPlane = 3000;
		ad->showHud = true;

		ad->guiAlpha = 0.95f;

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

		ad->playerMode = true;
		ad->pickMode = true;
		ad->selectionRadius = 5;
		// input->captureMouse = true;
		input->captureMouse = false;

		*ad->blockMenu = {};
		ad->blockMenuSelected = 0;


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

		ad->bombFireInterval = 0.1f;
		ad->bombButtonDown = false;

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

		uint vao = 0;
		glCreateVertexArrays(1, &vao);
		glBindVertexArray(vao);

		// setup textures
		for(int i = 0; i < TEXTURE_SIZE; i++) {
			#ifdef USE_SRGB 
				globalGraphicsState->textures[i] = loadTextureFile(texturePaths[i], 1, GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE);
			#else 
				globalGraphicsState->textures[i] = loadTextureFile(texturePaths[i], 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
			#endif
		}

		// load cubemap
		glCreateTextures(GL_TEXTURE_CUBE_MAP_ARRAY, arrayCount(ad->cubemapTextureId), &ad->cubemapTextureId[0]);

		char* texturePaths[] = {
			 					// "..\\data\\skybox\\sb1.png",
								// "..\\data\\skybox\\sb2.png", 
								// "..\\data\\skybox\\sb3.jpg", 
								// "..\\data\\skybox\\sb4.png", 
								"..\\data\\skybox\\xoGVD3X.jpg", 
								// "..\\data\\skybox\\xoGVD3X.jpg", 
								// "C:\\Projects\\Hmm\\data\\skybox\\xoGVD3X.jpg", 
								};

		ad->cubeMapCount = arrayCount(texturePaths);

		// for(int textureIndex = 0; textureIndex < arrayCount(ad->cubemapTextureId); textureIndex++) {
		for(int textureIndex = 0; textureIndex < arrayCount(texturePaths); textureIndex++) {
			int texWidth, texHeight, n;
			uint* stbData = (uint*)stbi_load(texturePaths[textureIndex], &texWidth, &texHeight, &n, 4);

			int skySize = texWidth/(float)4;

			glTextureStorage3D(ad->cubemapTextureId[textureIndex], 5, GL_SRGB8_ALPHA8, skySize, skySize, 6);

			uint* skyTex = getTArray(uint, skySize*skySize);
			Vec2i texOffsets[] = {{2,1}, {0,1}, {1,0}, {1,2}, {1,1}, {3,1}};
			for(int i = 0; i < 6; i++) {
				Vec2i offset = texOffsets[i] * skySize;

				for(int x = 0; x < skySize; x++) {
					for(int y = 0; y < skySize; y++) {
						skyTex[y*skySize + x] = stbData[(offset.y+y)*texWidth + (offset.x+x)];
					}
				}

				glTextureSubImage3D(ad->cubemapTextureId[textureIndex], 0, 0, 0, i, skySize, skySize, 1, GL_RGBA, GL_UNSIGNED_BYTE, skyTex);
			}
			// glGenerateTextureMipmap(ad->cubemapTextureId);

			stbi_image_free(stbData);
		}




		// setup shaders and uniforms
		for(int i = 0; i < SHADER_SIZE; i++) {
			MakeShaderInfo* info = makeShaderInfo + i; 
			Shader* s = globalGraphicsState->shaders + i;

			s->program = createShader(info->vertexString, info->fragmentString, &s->vertex, &s->fragment);
			s->uniformCount = info->uniformCount;
			s->uniforms = getPArray(ShaderUniform, s->uniformCount);

			for(int i = 0; i < s->uniformCount; i++) {
				ShaderUniform* uni = s->uniforms + i;
				uni->type = info->uniformNameMap[i].type;	
				uni->vertexLocation = glGetUniformLocation(s->vertex, info->uniformNameMap[i].name);
				uni->fragmentLocation = glGetUniformLocation(s->fragment, info->uniformNameMap[i].name);
			}
		}



		// setup meshs
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



		ad->voxelSamplers[0] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR);
		ad->voxelSamplers[1] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR);
		ad->voxelSamplers[2] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);

		glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, ad->voxelTextures);

		ad->samplers[0] = createSampler(16.0f, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);

		// voxel textures
		const int mipMapCount = 5;
		char* p = getTString(34);
		strClear(p);
		strAppend(p, "..\\data\\minecraft textures\\");

		char* fullPath = getTString(234);
		#ifdef USE_SRGB
			glTextureStorage3D(ad->voxelTextures[0], mipMapCount, GL_SRGB8_ALPHA8, 32, 32, BX_Size);
		#else 
			glTextureStorage3D(ad->voxelTextures[0], mipMapCount, GL_RGBA8, 32, 32, BX_Size);
		#endif 

		for(int layerIndex = 0; layerIndex < BX_Size; layerIndex++) {
			int x,y,n;
			unsigned char* stbData = stbi_load(textureFilePaths[layerIndex], &x, &y, &n, 4);

			if(layerIndex == BX_Water) {
				uint* data = (uint*)stbData;
				for(int x = 0; x < 32; x++) {
					for(int y = 0; y < 32; y++) {
						Vec4 c;
						colorGetRGBA(data[y*32 + x], c.e);
						c.r = waterAlpha;
						data[y*32 + x] = mapRGBA(c.e);
					}
				}
			}

			glTextureSubImage3D(ad->voxelTextures[0], 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);

			stbi_image_free(stbData);
		}
		glGenerateTextureMipmap(ad->voxelTextures[0]);


		float alphaCoverage[mipMapCount] = {};
		int size = 32;
		Vec4* pixels = (Vec4*)getTMemory(sizeof(Vec4)*size*size);
		for(int i = 0; i < mipMapCount; i++) {
			glGetTextureSubImage(ad->voxelTextures[0], i, 0,0,BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, size*size*sizeof(Vec4), &pixels[0]);

			for(int y = 0; y < size; y++) {
				for(int x = 1; x < size; x++) {
					alphaCoverage[i] += pixels[y*size + x].a;
				}
			}
	
			alphaCoverage[i] = alphaCoverage[i] / (size*size);
			size /= 2;
		}

		float alphaCoverage2[mipMapCount] = {};
		size = 16;
		for(int i = 1; i < mipMapCount; i++) {
			glGetTextureSubImage(ad->voxelTextures[0], i, 0,0,BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, size*size*sizeof(Vec4), &pixels[0]);

			// float alphaScale = (size*size*alphaCoverage[0]) / (alphaCoverage[i]*size*size);
			float alphaScale = (alphaCoverage[0]) / (alphaCoverage[i]);

			for(int y = 0; y < size; y++) {
				for(int x = 1; x < size; x++) {
					pixels[y*size + x].a *= alphaScale;
					alphaCoverage2[i] += pixels[y*size + x].a;
				}
			}
		
			glTextureSubImage3D(ad->voxelTextures[0], i, 0, 0, BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, pixels);

			alphaCoverage2[i] = alphaCoverage2[i] / (size*size);
			size /= 2;
		}


		#ifdef USE_SRGB 
			glTextureStorage3D(ad->voxelTextures[1], 1, GL_SRGB8_ALPHA8, 32, 32, BX2_Size);
		#else 
			glTextureStorage3D(ad->voxelTextures[1], 1, GL_RGBA8, 32, 32, BX2_Size);
		#endif

		for(int layerIndex = 0; layerIndex < BX2_Size; layerIndex++) {
			int x,y,n;
			unsigned char* stbData = stbi_load(textureFilePaths2[layerIndex], &x, &y, &n, 4);
			
			glTextureSubImage3D(ad->voxelTextures[1], 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);
			stbi_image_free(stbData);
		}


		glCreateFramebuffers(5, ad->frameBuffers);
		glCreateRenderbuffers(2, ad->renderBuffers);
		glCreateTextures(GL_TEXTURE_2D, 6, ad->frameBufferTextures);
		GLenum result = glCheckNamedFramebufferStatus(ad->frameBuffers[0], GL_FRAMEBUFFER);

		return; // window operations only work after first frame?
	}

	if(second) {
		// setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, -1920, 0);
		// setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, 0, 0);
		setWindowStyle(windowHandle, wSettings->style);
		// setWindowMode(windowHandle, wSettings, WINDOW_MODE_FULLBORDERLESS);

		setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, 300, 300);
		setWindowMode(windowHandle, wSettings, WINDOW_MODE_WINDOWED);


		ad->updateFrameBuffers = true;
	}

	if(reload) {
		loadFunctions();
		SetWindowLongPtr(systemData->windowHandle, GWLP_WNDPROC, (LONG_PTR)mainWindowCallBack);
	}

	TIMER_BLOCK_BEGIN(Main)

	clearTMemory();

	LARGE_INTEGER counter;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency); 

	if(second) {
		QueryPerformanceCounter(&counter);
		ad->lastTimeStamp = counter.QuadPart;
		ad->dt = 1/(float)60;
	} else {
		QueryPerformanceCounter(&counter);
		float timeStamp = counter.QuadPart;
		ad->dt = (timeStamp - ad->lastTimeStamp);
		ad->dt *= 1000000;
		ad->dt = ad->dt/frequency.QuadPart;
		ad->dt = ad->dt / 1000000;
		ad->dt = clampMax(ad->dt, 1/(float)20);

		ad->lastTimeStamp = timeStamp;

		ad->time += ad->dt;
	}
	// printf("%f \n", ad->dt);
	// ad->dt = 0.016f;



	// alloc drawcommandlist	
	int clSize = kiloBytes(1000);
	drawCommandListInit(&ad->commandList3d, (char*)getTMemory(clSize), clSize);
	drawCommandListInit(&ad->commandList2d, (char*)getTMemory(clSize), clSize);
	globalCommandList = &ad->commandList3d;



	{
		// TIMER_BLOCK_NAMED("Input");
		updateInput(ds->input, isRunning, windowHandle);
		ad->input = *ds->input;
	}

	{
		// @NOTE: cant use f10 or f12 while debugging...

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

	if(input->keysPressed[KEYCODE_F1]) {
		int mode;
		if(wSettings->fullscreen) mode = WINDOW_MODE_WINDOWED;
		else mode = WINDOW_MODE_FULLBORDERLESS;
		setWindowMode(windowHandle, wSettings, mode);

		ad->updateFrameBuffers = true;
	}

	if(input->keysPressed[KEYCODE_F2]) {
		input->captureMouse = !input->captureMouse;
	}

	bool focus = GetFocus() == windowHandle;
	bool fpsMode = input->captureMouse && focus;

	if(fpsMode) {
		int w,h;
		Vec2i wPos;
		getWindowProperties(systemData->windowHandle, &w, &h, 0, 0, &wPos.x, &wPos.y);
		SetCursorPos(wPos.x + wSettings->currentRes.x/2, wPos.y + wSettings->currentRes.y/2);

		while(ShowCursor(false) >= 0);
	} else {
		while(ShowCursor(true) < 0);
	}

	if(input->keysPressed[KEYCODE_F3]) {
		static bool switchMonitor = false;

		setWindowMode(windowHandle, wSettings, WINDOW_MODE_WINDOWED);

		if(!switchMonitor) setWindowProperties(windowHandle, 1, 1, 1920, 0);
		else setWindowProperties(windowHandle, 1920, 1080, -1920, 0);
		switchMonitor = !switchMonitor;

		setWindowMode(windowHandle, wSettings, WINDOW_MODE_FULLBORDERLESS);

		ad->updateFrameBuffers = true;
	}

	if(ad->updateFrameBuffers) {
		ad->updateFrameBuffers = false;
		ad->aspectRatio = wSettings->aspectRatio;
		
		ad->fboRes.x = ad->fboRes.y*ad->aspectRatio;

		if(ad->useNativeRes) ad->curRes = wSettings->currentRes;
		else ad->curRes = ad->fboRes;

		Vec2i s = ad->curRes;


		glNamedRenderbufferStorageMultisample(ad->renderBuffers[0], ad->msaaSamples, GL_RGBA8, s.w, s.h);
		glNamedFramebufferRenderbuffer(ad->frameBuffers[0], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ad->renderBuffers[0]);

		glNamedRenderbufferStorageMultisample(ad->renderBuffers[1], ad->msaaSamples, GL_DEPTH_STENCIL, s.w, s.h);
		glNamedFramebufferRenderbuffer(ad->frameBuffers[0], GL_DEPTH_STENCIL_ATTACHMENT,  GL_RENDERBUFFER, ad->renderBuffers[1]);

		glDeleteTextures(1, &ad->frameBufferTextures[0]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[0]);
		glTextureStorage2D(ad->frameBufferTextures[0], 1, GL_RGBA8, s.w, s.h);
		glNamedFramebufferTexture(ad->frameBuffers[1], GL_COLOR_ATTACHMENT0, ad->frameBufferTextures[0], 0);

		glDeleteTextures(1, &ad->frameBufferTextures[3]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[3]);
		glTextureStorage2D(ad->frameBufferTextures[3], 1, GL_DEPTH24_STENCIL8, s.w, s.h);
		glNamedFramebufferTexture(ad->frameBuffers[1], GL_DEPTH_STENCIL_ATTACHMENT, ad->frameBufferTextures[3], 0);


		Vec2 reflectionRes = vec2(s);

		glDeleteTextures(1, &ad->frameBufferTextures[1]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[1]);
		glTextureStorage2D(ad->frameBufferTextures[1], 1, GL_RGBA8, reflectionRes.w, reflectionRes.h);
		glNamedFramebufferTexture(ad->frameBuffers[2], GL_COLOR_ATTACHMENT0, ad->frameBufferTextures[1], 0);

		glDeleteTextures(1, &ad->frameBufferTextures[2]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[2]);
		glTextureStorage2D(ad->frameBufferTextures[2], 1, GL_DEPTH24_STENCIL8, reflectionRes.w, reflectionRes.h);
		glNamedFramebufferTexture(ad->frameBuffers[2], GL_DEPTH_STENCIL_ATTACHMENT, ad->frameBufferTextures[2], 0);



		glDeleteTextures(1, &ad->frameBufferTextures[4]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[4]);
		// glTextureStorage2D(ad->frameBufferTextures[4], 1, GL_RGBA8, s.w, s.h);
		glTextureStorage2D(ad->frameBufferTextures[4], 1, GL_RGBA8, wSettings->currentRes.w, wSettings->currentRes.h);
		glNamedFramebufferTexture(ad->frameBuffers[3], GL_COLOR_ATTACHMENT0, ad->frameBufferTextures[4], 0);


		glDeleteTextures(1, &ad->frameBufferTextures[5]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[5]);
		glTextureStorage2D(ad->frameBufferTextures[5], 1, GL_RGBA8, wSettings->currentRes.w, wSettings->currentRes.h);
		// glTextureStorage2D(ad->frameBufferTextures[5], 1, GL_SRGB8_ALPHA8, wSettings->currentRes.w, wSettings->currentRes.h);
		glNamedFramebufferTexture(ad->frameBuffers[4], GL_COLOR_ATTACHMENT0, ad->frameBufferTextures[5], 0);



		GLenum result = glCheckNamedFramebufferStatus(ad->frameBuffers[0], GL_FRAMEBUFFER);
		GLenum result2 = glCheckNamedFramebufferStatus(ad->frameBuffers[1], GL_FRAMEBUFFER);
		GLenum result3 = glCheckNamedFramebufferStatus(ad->frameBuffers[2], GL_FRAMEBUFFER);
	}


	// 2d camera controls
	// Vec3* cam = &ad->camera;	
	// if(input->mouseButtonDown[0]) {
	// 	cam->x += input->mouseDeltaX*(cam->z/wSettings->currentRes.w);
	// 	cam->y -= input->mouseDeltaY*((cam->z/wSettings->currentRes.h)/ad->aspectRatio);
	// }

	// if(input->mouseWheel) {
	// 	float zoom = cam->z;
	// 	zoom -= input->mouseWheel/(float)1;
	// 	cam->z = zoom;
	// }

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

	// make sure the meshs around the player are loaded at startup
	if(second) {
		// Vec2i pPos = coordToMesh(ad->activeCam.pos);
		Vec2i pPos = coordToMesh(ad->player->pos);
		// for(int i = 0; i < 2; i++) {
			for(int y = -1; y < 2; y++) {
				for(int x = -1; x < 2; x++) {
					Vec2i coord = pPos - vec2i(x,y);

					VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
					makeMesh(m, ad->voxelHash, ad->voxelHashSize);
				}
			}

			threadQueueComplete(globalThreadQueue);
		// }
	}	

	// @update entities
	// TIMER_BLOCK_BEGIN_NAMED(entities, "Upd Entities");
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

				if((!fpsMode && input->mouseButtonDown[1]) || fpsMode) {
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

						// get mesh coords that touch the player box
						Rect3 box = rect3CenDim(nPos, pSize);
						Vec3i voxelMin = coordToVoxel(box.min);
						Vec3i voxelMax = coordToVoxel(box.max+1);

						Vec3 collisionBox;
						collision = false;
						float minDistance = 100000;

							// check collision with the voxel thats closest
						for(int x = voxelMin.x; x < voxelMax.x; x++) {
							for(int y = voxelMin.y; y < voxelMax.y; y++) {
								for(int z = voxelMin.z; z < voxelMax.z; z++) {
									Vec3i coord = vec3i(x,y,z);
									uchar* block = getBlockFromVoxel(ad->voxelHash, ad->voxelHashSize, coord);

									if(*block > 0) {
										Vec3 cBox = voxelToVoxelCoord(coord);
										float distance = lenVec3(nPos - cBox);
										if(minDistance == 100000 || distance > minDistance) {
											minDistance = distance;
											collisionBox = cBox;
										}
										collision = true;
									}
								}
							}
						}

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

				if((!fpsMode && input->mouseButtonDown[1]) || fpsMode) {
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
	// TIMER_BLOCK_END(entities);

	if(ad->playerMode) {
		ad->activeCam = getCamData(ad->player->pos, ad->player->rot, ad->player->camOff);
	} else {
		ad->activeCam = getCamData(ad->cameraEntity->pos, ad->cameraEntity->rot, ad->cameraEntity->camOff);
	}

	// selecting blocks and modifying them
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

			if(ad->playerMode && fpsMode) {
				VoxelMesh* vm = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordToMesh(intersectionBox));

				uchar* block = getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, intersectionBox);
				uchar* lighting = getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, intersectionBox);

				bool mouse1 = input->mouseButtonPressed[0];
				bool mouse2 = input->mouseButtonPressed[1];
				bool placeBlock = (!fpsMode && ad->pickMode && mouse1) || (fpsMode && mouse1);
				bool removeBlock = (!fpsMode && !ad->pickMode && mouse1) || (fpsMode && mouse2);

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

	// opengl init

	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

	const int count = 10;
	GLenum sources;
	GLenum types;
	GLuint ids;
	GLenum severities;
	GLsizei lengths;

	int bufSize = 1000;
	char* messageLog = getTString(bufSize);

	uint fetchedLogs = 1;
	while(fetchedLogs = glGetDebugMessageLog(count, bufSize, &sources, &types, &ids, &severities, &lengths, messageLog)) {
		if(severities == GL_DEBUG_SEVERITY_NOTIFICATION) continue;

		if(severities == GL_DEBUG_SEVERITY_HIGH) printf("HIGH: \n");
		else if(severities == GL_DEBUG_SEVERITY_MEDIUM) printf("MEDIUM: \n");
		else if(severities == GL_DEBUG_SEVERITY_LOW) printf("LOW: \n");
		else if(severities == GL_DEBUG_SEVERITY_NOTIFICATION) printf("NOTE: \n");

		printf("\t%s \n", messageLog);
	}

	// glEnable(GL_FRAMEBUFFER_SRGB);
	glEnable(GL_CULL_FACE);

	// Vec3 skyColor = vec3(0.90f, 0.90f, 0.95f);
	// Vec3 skyColor = vec3(0.95f);
	Vec3 skyColor = vec3(0.90f);
	// Vec3 fogColor = vec3(0.75f, 0.85f, 0.95f);
	// Vec3 fogColor = vec3(0.43f,0.38f,0.44f);
	Vec3 fogColor = vec3(0.43f,0.38f,0.44f);
	fogColor.x = powf(fogColor.x, (float)2.2f);
	fogColor.y = powf(fogColor.y, (float)2.2f);
	fogColor.z = powf(fogColor.z, (float)2.2f);

	// for tech showcase
	#ifdef STBVOX_CONFIG_LIGHTING_SIMPLE
	skyColor = skyColor * vec3(0.3f);pushUniform
	fogColor = fogColor * vec3(0.3f);
	#endif 

	glViewport(0,0, ad->curRes.x, ad->curRes.y);
	// glClearColor(0,0,0, 1.0f);
	// glClearColor(skyColor.x, skyColor.y, skyColor.z, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer (GL_FRAMEBUFFER, ad->frameBuffers[4]);
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindFramebuffer (GL_FRAMEBUFFER, ad->frameBuffers[1]);
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[0]);
	glClearColor(skyColor.x, skyColor.y, skyColor.z, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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

	lookAt(ad->activeCam.pos, -ad->activeCam.look, ad->activeCam.up, ad->activeCam.right);
	perspective(degreeToRadian(ad->fieldOfView), ad->aspectRatio, ad->nearPlane, ad->farPlane);

	Mat4 view, proj; 
	viewMatrix(&view, ad->activeCam.pos, -ad->activeCam.look, ad->activeCam.up, ad->activeCam.right);
	projMatrix(&proj, degreeToRadian(ad->fieldOfView), ad->aspectRatio, ad->nearPlane, ad->farPlane);

	globalGraphicsState->textureUnits[0] = ad->voxelTextures[0];
	globalGraphicsState->textureUnits[1] = ad->voxelTextures[1];
	globalGraphicsState->samplerUnits[0] = ad->voxelSamplers[0];
	globalGraphicsState->samplerUnits[1] = ad->voxelSamplers[1];
	globalGraphicsState->samplerUnits[2] = ad->voxelSamplers[2];

	setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, fogColor);



	// draw cubemap
	bindShader(SHADER_CUBEMAP);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glBindTextures(0, 1, &ad->cubemapTextureId[ad->cubeMapDrawIndex]);
	glBindSamplers(0, 1, ad->samplers);

	Vec3 skyBoxRot;
	if(ad->playerMode) skyBoxRot = ad->player->rot;
	else skyBoxRot = ad->cameraEntity->rot;
	skyBoxRot.x += M_PI;

	Camera skyBoxCam = getCamData(vec3(0,0,0), skyBoxRot, vec3(0,0,0), vec3(0,1,0), vec3(0,0,1));

	Mat4 viewMat; viewMatrix(&viewMat, skyBoxCam.pos, -skyBoxCam.look, skyBoxCam.up, skyBoxCam.right);
	Mat4 projMat; projMatrix(&projMat, degreeToRadian(ad->fieldOfView), ad->aspectRatio, 0.001f, 2);
	pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_VIEW, viewMat.e);
	pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_PROJ, projMat.e);

	pushUniform(SHADER_CUBEMAP, 2, CUBEMAP_UNIFORM_CLIPPLANE, false);

	glDepthMask(false);
	glFrontFace(GL_CCW);
	glDrawArrays(GL_TRIANGLES, 0, 6*6);
	glFrontFace(GL_CW);
	glDepthMask(true);
	glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);


	// #if 1



	// static float worldTimer = 0;
	// worldTimer += ad->dt;

	// if(worldTimer >= 1) {
	// 	worldTimer = 0;

	// 	int radius = VIEW_DISTANCE/VOXEL_X;

	// 	for(int y = -radius; y < radius; y++) {
	// 		for(int x = -radius; x < radius; x++) {
	// 			Vec2i coord = vec2i(y, x);
	// 			// Vec2i coord = vec2i(0,0);	
	// 			VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
	// 			m->upToDate = false;
	// 			m->meshUploaded = false;
	// 			m->generated = false;
	// 		}
	// 	}
	// }

	// bool worldLoaded = false;
	// while(worldLoaded == false) {
	// 	int radius = VIEW_DISTANCE/VOXEL_X;

	// 	worldLoaded = true;

	// 	for(int y = -radius; y < radius; y++) {
	// 		for(int x = -radius; x < radius; x++) {
	// 			Vec2i coord = vec2i(y, x);
	// 			VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);

	// 			if(!m->meshUploaded) {
	// 				makeMesh(m, ad->voxelHash, ad->voxelHashSize);

	// 				worldLoaded = false;
	// 			}
	// 		}
	// 	}
	// }

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
			// return;
		} 
	}


	#if 0
	// @worldgen
	if(reload) {
		int radius = VIEW_DISTANCE/VOXEL_X;

		for(int y = -radius; y < radius; y++) {
			for(int x = -radius; x < radius; x++) {
				Vec2i coord = vec2i(y, x);
				// Vec2i coord = vec2i(0,0);	
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
				// m->upToDate = false;
				// m->meshUploaded = false;
				// m->generated = false;
			}
		}
		return;
	}
	#endif

	// TIMER_BLOCK_BEGIN_NAMED(world, "Upd World");

	Vec2i* coordList = (Vec2i*)getTMemory(sizeof(Vec2i)*2000);
	int coordListSize = 0;

	int meshGenerationCount = 0;
	int radCounter = 0;
	int triangleCount = 0;
	int drawCounter = 0;

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
				triangleCount += m->quadCount/(float)2;
				drawCounter++;
			}
		}
	}

	SortPair* sortList = (SortPair*)getTMemory(sizeof(SortPair)*coordListSize);
	int sortListSize = 0;

	for(int i = 0; i < coordListSize; i++) {
		Vec2 c = meshToMeshCoord(coordList[i]).xy;
		float distanceToCamera = lenVec2(ad->activeCam.pos.xy - c);
		sortList[sortListSize++] = {distanceToCamera, i};
	}

	radixSortPair(sortList, sortListSize);

	for(int i = 0; i < sortListSize-1; i++) {
		assert(sortList[i].key <= sortList[i+1].key);
	}

	// TIMER_BLOCK_END(world);

	{
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

			glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[2]);
			glClearColor(0,0,0,0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

			Vec2i reflectionRes = ad->curRes;
			glBlitNamedFramebuffer (ad->frameBuffers[0], ad->frameBuffers[2], 
				0,0, ad->curRes.x, ad->curRes.y,
				0,0, ad->curRes.x, ad->curRes.y,
				                   // 0,0, reflectionRes.x-1, reflectionRes.y-1,
				                   // 0,0, ad->curRes.x*0.5f, ad->curRes.y*0.5f,
				                   // GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT,
				GL_STENCIL_BUFFER_BIT,
				GL_NEAREST);

			glEnable(GL_CLIP_DISTANCE0);
			// glEnable(GL_CLIP_DISTANCE1);
			glEnable(GL_DEPTH_TEST);
			glFrontFace(GL_CCW);

				// draw cubemap reflection
				bindShader(SHADER_CUBEMAP);
				glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
				glBindTextures(0, 1, &ad->cubemapTextureId[ad->cubeMapDrawIndex]);
				glBindSamplers(0, 1, ad->samplers);

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

			setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, fogColor, vec3(0,0,WATER_LEVEL_HEIGHT*2 + 0.01f), vec3(1,1,-1));
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

			glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[0]);
			glDisable(GL_DEPTH_TEST);

			bindShader(SHADER_QUAD);
			drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1,1,1,reflectionAlpha), ad->frameBufferTextures[1]);

			glEnable(GL_DEPTH_TEST);

			// 	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			// 	glBlendEquation(GL_FUNC_ADD);
		}

		// draw water
		{
			setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, fogColor);
			pushUniform(SHADER_VOXEL, 1, VOXEL_UNIFORM_ALPHATEST, 0.5f);

			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
				drawVoxelMesh(m, 1);
			}
		}
	}


	// Vec3 off = vec3(0.5f, 0.5f, 0.5f);
	// Vec3 s = vec3(1.01f, 1.01f, 1.01f);

	// for(int i = 0; i < 10; i++) dcCube({vec3(i*10,0,0) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});
	// for(int i = 0; i < 10; i++) dcCube({vec3(0,i*10,0) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});
	// for(int i = 0; i < 10; i++) dcCube({vec3(0,0,i*10) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});

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

	// if(0)
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

	particleEmitterUpdate(&emitter, ad->dt);

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
		glBindSamplers(0,1,ad->samplers);


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


	// dcRect({rectCenDim(400,-400,200,200), rect(0,0,1,1), rColor, getTexture(TEXTURE_WHITE)->id});
	// dcRect({rectCenDim(});
	// dcRect({rectCenDim(});

		// dcText({fillString("Pos  : (%f,%f,%f)", PVEC3(ad->activeCam.pos)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow});

	// Vec2 tPos = vec2(900,-300);
	// char* text = "This is a Test String!";
	// // Font* f = getFont(FONT_ARIAL);
	// Font* f = getFont(FONT_ARIAL, 40);
	// dcText({text, f, tPos, vec4(1,1,0,1), 0, 2, 1});
	// // dcRect({rectCenDim(tPos-vec2(0,10), vec2(1,20)), rect(0,0,1,1), vec4(1,0,0,1), getTexture(TEXTURE_WHITE)->id});

	// float xOff = getTextPos(text, 5, f);
	// dcRect({rectCenDim(tPos + vec2(xOff,-20), vec2(1,40)), rect(0,0,1,1), vec4(1,0,0,1), getTexture(TEXTURE_WHITE)->id});



	// @menu
	if(ad->playerMode) {
		globalCommandList = &ad->commandList2d;

		for(int i = 0; i < 10; i++) {
			ad->blockMenu[i] = i+1;
		}

		Vec2 res = vec2(wSettings->currentRes);
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

			uint textureId = ad->voxelTextures[0];
			dcRect(rectCenDim(pos, iconSize*iconOff), rect(0,0,1,1), color, (int)textureId, texture1Faces[i+1][0]+1);
		}

		globalCommandList = &ad->commandList3d;
	}

	TIMER_BLOCK_END(Main)




	// @debug
	{
		clearTMemoryDebug();

		ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[1];
		ExtendibleMemoryArray* pMemory = globalMemory->pMemory;

		int clSize = megaBytes(2);
		drawCommandListInit(&ds->commandListDebug, (char*)getTMemoryDebug(clSize), clSize);
		globalCommandList = &ds->commandListDebug;

		input = ds->input;

		{
			int fontSize = 18;
			int pi = 0;
			// Vec4 c = vec4(1.0f,0.2f,0.0f,1);
			Vec4 c = vec4(1.0f,0.4f,0.0f,1);
			Vec4 c2 = vec4(0,0,0,1);
			Font* font = getFont(FONT_CONSOLAS, fontSize);
			// Font* font = getFont(FONT_CALIBR, fontSize);
			int shadow = 1;
			// float shadow = 0;
			float xo = 6;
			int ali = 2;

			Vec2i tp = ad->wSettings.currentRes - vec2i(xo, 0);
			#define PVEC3(v) v.x, v.y, v.z
			#define PVEC2(v) v.x, v.y
			dcText(fillString("Pos  : (%f,%f,%f)", PVEC3(ad->activeCam.pos)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
			dcText(fillString("Look : (%f,%f,%f)", PVEC3(ad->activeCam.look)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
			dcText(fillString("Up   : (%f,%f,%f)", PVEC3(ad->activeCam.up)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
			dcText(fillString("Right: (%f,%f,%f)", PVEC3(ad->activeCam.right)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
			dcText(fillString("Rot  : (%f,%f)", 	PVEC2(player->rot)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
			dcText(fillString("Vec  : (%f,%f,%f)", PVEC3(player->vel)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
			dcText(fillString("Acc  : (%f,%f,%f)", PVEC3(player->acc)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
			dcText(fillString("Draws: (%i)", 		drawCounter), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
			dcText(fillString("Quads: (%i)", 		triangleCount), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);

			dcText(fillString("Threads: (%i, %i)",	threadQueue->completionCount, threadQueue->completionGoal), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		}

		{			
			if(input->keysPressed[KEYCODE_F5]) ad->showHud = !ad->showHud;

			if(ad->showHud) {
				int fontSize = 18;

				bool initSections = false;

				Gui* gui = ds->gui;
				GuiInput gInput = { vec2(input->mousePos), input->mouseWheel, input->mouseButtonPressed[0], input->mouseButtonDown[0], 
									input->keysPressed[KEYCODE_ESCAPE], input->keysPressed[KEYCODE_RETURN], input->keysPressed[KEYCODE_SPACE], input->keysPressed[KEYCODE_BACKSPACE], input->keysPressed[KEYCODE_DEL], input->keysPressed[KEYCODE_HOME], input->keysPressed[KEYCODE_END], 
									input->keysPressed[KEYCODE_LEFT], input->keysPressed[KEYCODE_RIGHT], input->keysPressed[KEYCODE_UP], input->keysPressed[KEYCODE_DOWN], 
									input->keysDown[KEYCODE_SHIFT], input->keysDown[KEYCODE_CTRL], input->inputCharacters, input->inputCharacterCount};
				// gui->start(gInput, getFont(FONT_CONSOLAS, fontSize), wSettings->currentRes);
				gui->start(gInput, getFont(FONT_CALIBRI, fontSize), wSettings->currentRes);



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
					gui->div(vec2(0,0)); gui->label("GuiAlpha", 0); gui->slider(&ad->guiAlpha, 0.1f, 1);
					gui->div(vec2(0,0)); gui->label("FoV", 0); gui->slider(&ad->fieldOfView, 1, 180);
					gui->div(vec2(0,0)); gui->label("MSAA", 0); gui->slider(&ad->msaaSamples, 1, 8);
					gui->switcher("Native Res", &ad->useNativeRes);
					gui->div(0,0,0); gui->label("FboRes", 0); gui->slider(&ad->fboRes.x, 150, ad->curRes.x); gui->slider(&ad->fboRes.y, 150, ad->curRes.y);
					gui->div(0,0,0); gui->label("NFPlane", 0); gui->slider(&ad->nearPlane, 0.01, 2); gui->slider(&ad->farPlane, 1000, 5000);
				} gui->endSection();

				static bool sectionWorld = initSections;
				if(gui->beginSection("World", &sectionWorld)) { 
					if(gui->button("Reload World") || input->keysPressed[KEYCODE_TAB]) ad->reloadWorld = true;
					
					gui->div(vec2(0,0)); gui->label("CubeMap", 0); gui->slider(&ad->cubeMapDrawIndex, 0, ad->cubeMapCount);

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
		}



		int globalTimingsCount = __COUNTER__;

		DebugState* ds = globalDebugState;
		ds->timerInfoCount = globalTimingsCount;

		int bufferIndex = ds->bufferIndex;
		Timings* timings = ds->timings[ds->cycleIndex];
		zeroMemory(timings, ds->timerInfoCount*sizeof(Timings));

		ds->cycleIndex = (ds->cycleIndex + 1)%arrayCount(ds->timings);

		ds->bufferIndex = 0;

		int fontHeight = 18;

		if(reload) {
			for(int i = 0; i < arrayCount(ds->timerInfos); i++) ds->timerInfos[i].initialised = false;
			return;
		}



		// collate timing buffer
		TimerStatistic stats[16] = {};
		int index = 0;

		for(int i = 0; i < bufferIndex; ++i) {
			TimerSlot* slot = ds->timerBuffer + i;

			if(slot->type == TIMER_TYPE_BEGIN) {
				stats[index].cycles = slot->cycles;
				stats[index].timerIndex = slot->timerIndex;
				index++;
			}

			if(slot->type == TIMER_TYPE_END) {
				index--;
				Timings* timing = timings + stats[index].timerIndex;
				timing->cycles += slot->cycles - stats[index].cycles;
				timing->hits++;
			}
		}

		for(int i = 0; i < ds->timerInfoCount; i++) {
			Timings* t = timings + i;
			t->cyclesOverHits = t->hits > 0 ? (u64)(t->cycles/t->hits) : 0; 
		}

		Statistic statistics[32] = {};
		for(int timerIndex = 0; timerIndex < ds->timerInfoCount; timerIndex++) {
			Statistic* stat = statistics + timerIndex;
			beginStatistic(stat);

			for(int i = 0; i < arrayCount(ds->timings); i++) {
				Timings* t = &ds->timings[i][timerIndex];
				updateStatistic(stat, t->cyclesOverHits);
			}

			endStatistic(stat);
		}

		Font* debugFont = getFont(FONT_CALIBRI, 18);

		// draw timing info
		float cyclesPerFrame = (float)((3*((float)1/60))*1024*1024*1024);
		fontHeight = 18;
		Vec2 textPos = vec2(550, -fontHeight);
		int infoCount = ds->timerInfoCount;

		bool initSections = false;

		GuiInput gInput = { vec2(input->mousePos), input->mouseWheel, input->mouseButtonPressed[0], input->mouseButtonDown[0], 
							input->keysPressed[KEYCODE_ESCAPE], input->keysPressed[KEYCODE_RETURN], input->keysPressed[KEYCODE_SPACE], input->keysPressed[KEYCODE_BACKSPACE], input->keysPressed[KEYCODE_DEL], input->keysPressed[KEYCODE_HOME], input->keysPressed[KEYCODE_END], 
							input->keysPressed[KEYCODE_LEFT], input->keysPressed[KEYCODE_RIGHT], input->keysPressed[KEYCODE_UP], input->keysPressed[KEYCODE_DOWN], 
							input->keysDown[KEYCODE_SHIFT], input->keysDown[KEYCODE_CTRL], input->inputCharacters, input->inputCharacterCount};
		Gui* gui = ds->gui2;
		gui->start(gInput, getFont(FONT_CALIBRI, fontHeight), wSettings->currentRes);

		static bool statsSection = false;
		static bool graphSection = false;
		gui->div(0.2f,0.2f,0); gui->switcher("Stats", &statsSection); gui->switcher("Graph", &graphSection); gui->empty();


		if(statsSection) {
			int barWidth = 1;
			int barCount = arrayCount(ds->timings);
			float sectionWidths[] = {0,0,0,0,0,0,0,0, barWidth*barCount};

			char* headers[] = {"File", "Function", "Description", "Cycles", "Hits", "C/H", "Avg. Cycl.", "Total Time", ""};
			gui->div(sectionWidths, arrayCount(sectionWidths));
			for(int i = 0; i < arrayCount(sectionWidths); i++) gui->label(headers[i],1);

			for(int i = 0; i < infoCount; i++) {
				TimerInfo* tInfo = ds->timerInfos + i;
				Timings* timing = timings + i;

				float cycleCountPercent = (float)timing->cycles/cyclesPerFrame;
				char * percentString = getTStringDebug(50);
				percentString = floatToStr(percentString, cycleCountPercent*100, 3);

				int debugStringSize = 50;
				char* buffer = 0;

				gui->div(sectionWidths, arrayCount(sectionWidths)); 

				buffer = getTStringDebug(debugStringSize);
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s", tInfo->file + 21);
				gui->label(buffer,0);

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

				gui->empty();
				Rect r = gui->getCurrentRegion();
				float rheight = gui->getDefaultHeight();

				float xOffset = 0;
				for(int statIndex = 0; statIndex < barCount; statIndex++) {
					Statistic* stat = statistics + i;
					u64 coh = ds->timings[statIndex][i].cyclesOverHits;

					float height = mapRangeClamp(coh, stat->min, stat->max, 1, rheight);
					Vec2 rmin = r.min + vec2(xOffset,-2);
					float colorOffset = mapRange(coh, stat->min, stat->max, 0, 1);
					// dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,0,1-colorOffset,1));
					dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,1-colorOffset,0,1));

					xOffset += barWidth;
				}
			}
		}

		// gui->button("asddsf");

		// float xOffset = 0;
		// for(int statIndex = 0; statIndex < bufferIndex; statIndex++) {
		// 	Statistic* stat = statistics + ds->cycleIndex;
		// 	u64 coh = ds->timings[statIndex][ds->cycleIndex].cyclesOverHits;

		// 	int debugStringSize = 30;
		// 	char* buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
		// 	_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64uc ", coh);
		// 	gui->label(buffer,0);

		// 	// float height = mapRangeClamp(coh, stat->min, stat->max, 1, rheight);
		// 	// Vec2 rmin = r.min + vec2(xOffset,-2);
		// 	// float colorOffset = mapRange(coh, stat->min, stat->max, 0, 1);
		// 	// dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,0,1-colorOffset,1));
		// 	// dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,1-colorOffset,0,1));

		// 	// xOffset += barWidth;
		// }





		// save timer buffer
		if(input->keysPressed[KEYCODE_F6]) {
			if(!ds->frozenGraph) {
				memCpy(ds->savedTimerBuffer, ds->timerBuffer, bufferIndex*sizeof(TimerSlot));
				ds->savedBufferIndex = bufferIndex;
				memCpy(ds->savedTimings, timings, ds->timerInfoCount*sizeof(Timings));
			}

			ds->frozenGraph = !ds->frozenGraph;
		}


		if(graphSection) {
			static Vec2 trans = vec2(10,0);
			static float zoom = 1;

			gui->div(vec2(0.1f,0)); 
			if(gui->button("Reset")) {
				trans = vec2(10,0);
				zoom = 1;
			}
			gui->empty();
			gui->heightPush(3);
			gui->empty();
			
			Rect bgRect = gui->getCurrentRegion();
			Vec2 dragDelta = vec2(0,0);
			gui->drag(bgRect, &dragDelta, vec4(0,0,0,0));

			trans.x += dragDelta.x;
			// trans.x = clampMax(trans.x, 10);
			gui->heightPop();

			if(gui->input.mouseWheel) {
				zoom *= 1 + gui->input.mouseWheel*0.1f;
				// zoom += gui->input.mouseWheel*10;
			}


			Timings* graphTimings = timings;
			TimerSlot* graphTimerBuffer = ds->timerBuffer;
			u64 graphBufferIndex = bufferIndex;

			if(ds->frozenGraph) {
				graphTimings = ds->savedTimings;
				graphTimerBuffer = ds->savedTimerBuffer;
				graphBufferIndex = ds->savedBufferIndex;
			}

			float graphWidth = rectGetDim(bgRect).w;
			Vec2 startPos = rectGetUL(bgRect);

			if(true) {
				u64 baseCycleCount = graphTimerBuffer[0].cycles;
				u64 startCycleCount = 0;
				u64 endCycleCount = cyclesPerFrame;

				float orthoLeft = 0 + trans.x;
				float orthoRight = graphWidth*zoom + trans.x;

				Rect bgRect = rectULDim(startPos, vec2(graphWidth, fontHeight*3));
				dcRect(bgRect, vec4(1,1,1,0.2f));

				float lineWidth = 1;
				int lineCount = 10;
				for(int i = 0; i < lineCount+1; i++) {
					float linePos = ((graphWidth-20)/(float)lineCount) * i;
					linePos *= zoom;
					linePos += trans.x;
					linePos += bgRect.min.x;
					Vec4 color = vec4(0.7f,0.7f,0.7f,1);
					float lw = lineWidth;
					if(i == 0 || i == lineCount) {
						color = vec4(1,1,1,1);
						lw = lineWidth * 3;
					}
					dcRect(rect(linePos, bgRect.min.y, linePos+lw, bgRect.max.y), color);
				} 

				startPos -= vec2(0, fontHeight);
				index = 0;
				for(int i = 0; i < graphBufferIndex; ++i) {
					TimerSlot* slot = graphTimerBuffer + i;

					if(slot->type == TIMER_TYPE_BEGIN) {

						Timings* t = graphTimings + slot->timerIndex;
						TimerInfo* tInfo = ds->timerInfos + slot->timerIndex;

						float xoff = mapRange(slot->cycles - baseCycleCount, startCycleCount, endCycleCount, orthoLeft, orthoRight);
						float barWidth = mapRange(t->cycles, startCycleCount, endCycleCount, 0, graphWidth*zoom);
						Vec2 pos = startPos + vec2(xoff,index*-fontHeight);
						// Vec2 pos = startPos + vec2(xoff,0);
						
						Rect r = rect(pos, pos + vec2(barWidth,fontHeight));
						float cOff = slot->timerIndex/(float)ds->timerInfoCount;
						Vec4 c = vec4(1-cOff, 0, cOff, 1);

						int debugStringSize = 50;
						char* buffer = getTStringDebug(debugStringSize);
						_snprintf_s(buffer, debugStringSize, debugStringSize, "%s %s", tInfo->function, tInfo->name);

						gui->drawRect(r, vec4(0,0,0,1));
						gui->drawTextBox(rect(r.min+vec2(1,1), r.max-vec2(1,1)), buffer, c);

						index++;
					}

					if(slot->type == TIMER_TYPE_END) {
						index--;
					}
				}

			}
		}

		gui->end();
	}



	{
		TIMER_BLOCK_NAMED("Render");

		bindShader(SHADER_CUBE);
		executeCommandList(&ad->commandList3d);

		ortho(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0));
		glDisable(GL_DEPTH_TEST);
		bindShader(SHADER_QUAD);
		glBlitNamedFramebuffer(ad->frameBuffers[0], ad->frameBuffers[1], 0,0, ad->curRes.x, ad->curRes.y, 0,0, ad->curRes.x, ad->curRes.y, GL_COLOR_BUFFER_BIT /*GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT*/, GL_LINEAR /*GL_NEAREST*/);

		glDisable(GL_DEPTH_TEST);
		bindShader(SHADER_QUAD);
		glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[3]);
		glViewport(0,0, wSettings->currentRes.x, wSettings->currentRes.y);
		drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1), ad->frameBufferTextures[0]);

		executeCommandList(&ad->commandList2d);




		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);
		glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[4]);
		executeCommandList(&ds->commandListDebug);
		glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[3]);

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);
		drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1,1,1,ad->guiAlpha), ad->frameBufferTextures[5]);



		#ifdef USE_SRGB 
			glEnable(GL_FRAMEBUFFER_SRGB);
		#endif 

		glBindFramebuffer (GL_FRAMEBUFFER, 0);
		bindShader(SHADER_QUAD);
		glBindSamplers(0, 1, ad->samplers);

		drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1), ad->frameBufferTextures[4]);

		#ifdef USE_SRGB
			glDisable(GL_FRAMEBUFFER_SRGB);
		#endif

		if(second) {
			GLenum glError = glGetError(); printf("GLError: %i\n", glError);
		}

	}

	{
		TIMER_BLOCK_NAMED("Swap");
		swapBuffers(&ad->systemData);
		glFinish();
	}






	if(*isRunning == false) {
		guiSave(ds->gui, 2, 0);
		if(globalDebugState->gui2) guiSave(globalDebugState->gui2, 2, 3);
	}

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

POSTMAINFUNCTION(postMain) {
	
}

// #pragma optimize( "", off)
#pragma optimize( "", on ) 
