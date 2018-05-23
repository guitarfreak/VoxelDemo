
int TIMER_INFO_COUNT;

#define DEBUG_NOTE_LENGTH 50
#define DEBUG_NOTE_DURATION 3

enum StructType {
	STRUCTTYPE_INT = 0,
	STRUCTTYPE_FLOAT,
	STRUCTTYPE_CHAR,
	STRUCTTYPE_BOOL,

	STRUCTTYPE_VEC3,
	STRUCTTYPE_ENTITY,

	STRUCTTYPE_SIZE,
};

enum ArrayType {
	ARRAYTYPE_CONSTANT, 
	ARRAYTYPE_DYNAMIC, 

	ARRAYTYPE_SIZE, 
};

struct ArrayInfo {
	int type;
	int sizeMode;

	union {
		int size;
		int offset;
	};
};

struct StructMemberInfo {
	char* name;
	int type;
	int offset;
	int arrayCount;
	ArrayInfo arrays[2];
};

struct StructInfo {
	char* name;
	int size;
	int memberCount;
	StructMemberInfo* list;
};

struct Asset {
	int index;
	int folderIndex;
	char* filePath;
	FILETIME lastWriteTime;
};

struct DebugState {
	Asset* assets;
	int assetCount;

	i64 lastTimeStamp;
	f64 dt;
	f64 time;

	f64 debugTime;
	f64 debugRenderTime;

	MSTimer swapTimer;
	MSTimer frameTimer;
	MSTimer tempTimer;

	f64 fpsTime;
	int fpsCounter;
	float avgFps;

	bool showMenu;
	bool showStats;
	bool showConsole;
	bool showHud;

	DrawCommandList commandListDebug;
	Input* input;

	Timer* timer;
	Timings timings[120][32];
	Statistic statistics[120][32];
	int cycleIndex;
	bool stopCollating;

	bool setPause;
	bool setPlay;
	bool noCollating;
	int lastCycleIndex;
	GraphSlot* savedBuffer;
	int savedBufferIndex;
	int savedBufferMax;
	int savedBufferCount;
	int lastBufferIndex;

	GraphSlot graphSlots[16][8]; // threads, stackIndex
	int graphSlotCount[16];

	//

	int mode;
	int graphSortingIndex;
	
	double timelineCamPos;
	double timelineCamSize;

	float lineGraphCamPos;
	float lineGraphCamSize;
	float lineGraphHeight;
	int lineGraphHighlight;

	//

	int fontHeight;
	int fontHeightScaled;
	float fontScale;

	GuiInput gInput;
	Gui* gui;
	Gui* gui2;
	float guiAlpha;

	Input* recordedInput;
	int inputCapacity;
	bool recordingInput;
	int inputIndex;

	char* snapShotMemory[8];
	int snapShotCount;
	int snapShotMemoryIndex;

	bool playbackInput;
	int playbackIndex;
	bool playbackSwapMemory;
	bool playbackPause;
	bool playbackBreak;
	int playbackBreakIndex;

	Console console;

	char* notificationStack[10];
	float notificationTimes[10];
	int notificationCount;

	char* infoStack[10];
	int infoStackCount;
};



StructMemberInfo initMemberInfo(char* name, int type, int offset) {
	StructMemberInfo info;
	info.name = name;
	info.type = type;
	info.offset = offset;
	info.arrayCount = 0;

	return info;
}

StructMemberInfo initMemberInfo(char* name, int type, int offset, ArrayInfo aInfo1, ArrayInfo aInfo2 = {-1}) {
	StructMemberInfo info = initMemberInfo(name, type, offset);
	info.arrayCount = 1;
	if(aInfo2.type != -1) info.arrayCount = 2;

	info.arrays[0] = aInfo1;
	info.arrays[1] = aInfo2;

	return info;
}

StructMemberInfo vec3StructMemberInfos[] = {
	initMemberInfo("x", STRUCTTYPE_FLOAT, offsetof(Vec3, x)),
	initMemberInfo("y", STRUCTTYPE_FLOAT, offsetof(Vec3, y)),
	initMemberInfo("z", STRUCTTYPE_FLOAT, offsetof(Vec3, z)),
};

StructMemberInfo entityStructMemberInfos[] = {
	initMemberInfo("init", STRUCTTYPE_INT, offsetof(Entity, init)),
	initMemberInfo("type", STRUCTTYPE_INT, offsetof(Entity, type)),
	initMemberInfo("id", STRUCTTYPE_INT, offsetof(Entity, id)),
	initMemberInfo("name", STRUCTTYPE_CHAR, offsetof(Entity, name), {ARRAYTYPE_CONSTANT, 2, memberSize(Entity, name)}),
	initMemberInfo("pos", STRUCTTYPE_VEC3, offsetof(Entity, pos)),
	// initMemberInfo("dir", STRUCTTYPE_VEC3, offsetof(Entity, dir)),
	initMemberInfo("rot", STRUCTTYPE_VEC3, offsetof(Entity, rot)),
	initMemberInfo("rotAngle", STRUCTTYPE_FLOAT, offsetof(Entity, rotAngle)),
	initMemberInfo("dim", STRUCTTYPE_VEC3, offsetof(Entity, dim)),
	initMemberInfo("camOff", STRUCTTYPE_VEC3, offsetof(Entity, camOff)),
	initMemberInfo("vel", STRUCTTYPE_VEC3, offsetof(Entity, vel)),
	initMemberInfo("acc", STRUCTTYPE_VEC3, offsetof(Entity, acc)),
	initMemberInfo("movementType", STRUCTTYPE_INT, offsetof(Entity, movementType)),
	initMemberInfo("spatial", STRUCTTYPE_INT, offsetof(Entity, spatial)),
	initMemberInfo("deleted", STRUCTTYPE_BOOL, offsetof(Entity, deleted)),
	initMemberInfo("isMoving", STRUCTTYPE_BOOL, offsetof(Entity, isMoving)),
	initMemberInfo("isColliding", STRUCTTYPE_BOOL, offsetof(Entity, isColliding)),
	initMemberInfo("exploded", STRUCTTYPE_BOOL, offsetof(Entity, exploded)),
	initMemberInfo("onGround", STRUCTTYPE_BOOL, offsetof(Entity, onGround)),
};

StructInfo structInfos[] = {
	{ "int", sizeof(int), 0 },
	{ "float", sizeof(float), 0 },
	{ "char", sizeof(char), 0 },
	{ "bool", sizeof(bool), 0 },
	{ "Vec3", sizeof(Vec3), arrayCount(vec3StructMemberInfos), vec3StructMemberInfos }, 
	{ "Entity", sizeof(Entity), arrayCount(entityStructMemberInfos), entityStructMemberInfos }, 
};

bool typeIsPrimitive(int type) {
	return structInfos[type].memberCount == 0;
}

char* castTypeArray(char* base, ArrayInfo aInfo) {
	char* arrayBase = (aInfo.type == ARRAYTYPE_CONSTANT) ? base :*((char**)(base));
	return arrayBase;
}

int getTypeArraySize(char* structBase, ArrayInfo aInfo, char* arrayBase = 0) {
	int arraySize;
	if(aInfo.sizeMode == 0) arraySize = aInfo.size;
	else if(aInfo.sizeMode == 1) arraySize = *(int*)(structBase + aInfo.offset);
	else arraySize = strLen(arrayBase);

	return arraySize;
}

void printType(int structType, char* data, int depth = 0);
void printTypeArray(char* structBase, StructMemberInfo member, char* data, int arrayIndex, int depth) {
	int arrayPrintLimit = 5;

	bool isPrimitive = typeIsPrimitive(member.type);
	ArrayInfo aInfo = member.arrays[arrayIndex];
	int typeSize = structInfos[member.type].size;

	char* arrayBase = castTypeArray(data, aInfo);
	int arraySize = getTypeArraySize(structBase, aInfo, arrayBase);

	printf("[ ");
	
	if(arrayIndex == member.arrayCount-1) {
		int size = min(arraySize, arrayPrintLimit);
		bool isPrimitive = typeIsPrimitive(member.type);

		for(int i = 0; i < size; i++) {
			char* value = arrayBase + (typeSize*i);
			if(!isPrimitive) printf("{ ");
			printType(member.type, value, depth + 1);
			if(!isPrimitive) printf("}, ");
		}
		if(arraySize > arrayPrintLimit) printf("... ");
	} else {
		ArrayInfo aInfo2 = member.arrays[arrayIndex+1];

		int arrayOffset;
		if(aInfo.type == ARRAYTYPE_CONSTANT && aInfo2.type == ARRAYTYPE_CONSTANT) {
			int arraySize2 = getTypeArraySize(data, aInfo2);
			arrayOffset = arraySize2*typeSize;
		} else arrayOffset = sizeof(int*); // Pointers have the same size, so any will suffice.

		for(int i = 0; i < arraySize; i++) {
			printTypeArray(structBase, member, arrayBase + i*arrayOffset, arrayIndex+1, depth+1);
		}
	}

	printf("], ");
}

void printType(int structType, char* data, int depth) {
	int arrayPrintLimit = 5;
	int indent = 2;

	StructInfo* info = structInfos + structType;

	if(depth == 0) printf("{\n");

	if(typeIsPrimitive(structType)) {
		switch(structType) {
			case STRUCTTYPE_INT: printf("%i", *(int*)data); break;
			case STRUCTTYPE_FLOAT: printf("%f", *(float*)data); break;
			case STRUCTTYPE_CHAR: printf("%c", *data); break;
			case STRUCTTYPE_BOOL: printf((*((bool*)data)) ? "true" : "false"); break;
			default: break;
		};

		printf(", ");
		if(depth == 0) printf("\n");
		return;
	}

	for(int i = 0; i < info->memberCount; i++) {
		StructMemberInfo member = info->list[i];

		bool isPrimitive = typeIsPrimitive(member.type);

		if(depth == 0) printf("%*s", indent*(depth+1), "");

		if(member.arrayCount == 0) {
			if(!isPrimitive) printf("{ ");
			printType(member.type, data + member.offset, depth + 1);
			if(!isPrimitive) printf("}, ");
			if(depth == 0) printf("\n");
		} else {
			ArrayInfo aInfo = member.arrays[0];
			printTypeArray(data, member, data + member.offset, 0, depth + 1);

			if(depth == 0) printf("\n");
		}

	}

	if(depth == 0) printf("}\n");

}

void guiPrintIntrospection(Gui* gui, int structType, char* data, int depth = 0);
void guiPrintIntrospection(Gui* gui, char* structBase, StructMemberInfo member, char* data, int arrayIndex, int depth) {
	int arrayPrintLimit = 5;

	bool isPrimitive = typeIsPrimitive(member.type);
	ArrayInfo aInfo = member.arrays[arrayIndex];
	int typeSize = structInfos[member.type].size;

	char* arrayBase = castTypeArray(data, aInfo);
	int arraySize = getTypeArraySize(structBase, aInfo, arrayBase);

	if(arrayIndex == member.arrayCount-1) {
		int size = min(arraySize, arrayPrintLimit);
		bool isPrimitive = typeIsPrimitive(member.type);

		if(aInfo.sizeMode == 2) {
			gui->label(data);
		} else {
			for(int i = 0; i < size; i++) {
				char* value = arrayBase + (typeSize*i);
				printType(member.type, value, depth + 1);
			}
			if(arraySize > arrayPrintLimit) printf("... ");
		}
	} else {
		ArrayInfo aInfo2 = member.arrays[arrayIndex+1];

		int arrayOffset;
		if(aInfo.type == ARRAYTYPE_CONSTANT && aInfo2.type == ARRAYTYPE_CONSTANT) {
			int arraySize2 = getTypeArraySize(data, aInfo2);
			arrayOffset = arraySize2*typeSize;
		} else arrayOffset = sizeof(int*); // Pointers have the same size, so any will suffice.

		for(int i = 0; i < arraySize; i++) {
			guiPrintIntrospection(gui, structBase, member, arrayBase + i*arrayOffset, arrayIndex+1, depth+1);
		}
	}
}

void guiPrintIntrospection(Gui* gui, int structType, char* data, int depth) {
	int arrayPrintLimit = 5;
	int indent = 2;

	StructInfo* info = structInfos + structType;

	if(typeIsPrimitive(structType) == false) gui->label(fillString("%s", info->name), 0);

	gui->startPos.x += 30;
	gui->panelWidth -= 30;

	if(typeIsPrimitive(structType)) {
		switch(structType) {
			case STRUCTTYPE_INT: gui->label(fillString("%i", *(int*)data)); break;
			case STRUCTTYPE_FLOAT: gui->label(fillString("%f", *(float*)data)); break;
			case STRUCTTYPE_CHAR: gui->label(fillString("%c", *data)); break;
			case STRUCTTYPE_BOOL: gui->label(fillString("%b", *(bool*)data)); break;
			default: break;
		};
	} else {

		for(int i = 0; i < info->memberCount; i++) {
			StructMemberInfo member = info->list[i];

			bool isPrimitive = typeIsPrimitive(member.type);
			if(isPrimitive) {
				gui->div(vec2(0,0));
				gui->label(fillString("%s", member.name), 0);
			}

			if(member.arrayCount == 0) {
				guiPrintIntrospection(gui, member.type, data + member.offset, depth + 1);
			} else {
				ArrayInfo aInfo = member.arrays[0];
				guiPrintIntrospection(gui, data, member, data + member.offset, 0, depth + 1);
			}

		}
	}

	gui->startPos.x -= 30;
	gui->panelWidth += 30;

}



void addDebugNote(char* string, float duration = DEBUG_NOTE_DURATION) {
	DebugState* ds = theDebugState;

	assert(strLen(string) < DEBUG_NOTE_LENGTH);
	if(ds->notificationCount >= arrayCount(ds->notificationStack)) return;

	int count = ds->notificationCount;
	strClear(ds->notificationStack[count]);
	ds->notificationTimes[count] = duration;
	strCpy(ds->notificationStack[count], string);
	ds->notificationCount++;
}

void addDebugInfo(char* string) {
	DebugState* ds = theDebugState;

	if(ds->infoStackCount >= arrayCount(ds->infoStack)) return;
	ds->infoStack[ds->infoStackCount++] = string;
}



void initWatchFolders(HANDLE* folderHandles, Asset* assets, int* assetCount) {
	for(int i = 0; i < 3; i++) {
		HANDLE fileChangeHandle = FindFirstChangeNotification(watchFolders[i], false, FILE_NOTIFY_CHANGE_LAST_WRITE);

		if(fileChangeHandle == INVALID_HANDLE_VALUE) {
			printf("Could not set folder change notification.\n");
		}

		folderHandles[i] = fileChangeHandle;
	}

	int fileCounts[] = {TEXTURE_SIZE, CUBEMAP_SIZE, BX_Size};
	char** paths[] = {texturePaths, cubeMapPaths, (char**)textureFilePaths};
	for(int folderIndex = 0; folderIndex < arrayCount(watchFolders); folderIndex++) {

		int fileCount = fileCounts[folderIndex];

		for(int i = 0; i < fileCount; i++) {
			char* path = paths[folderIndex][i];
			Asset* asset = assets + *assetCount; 
			*assetCount = (*assetCount) + 1;
			asset->lastWriteTime = getLastWriteTime(path);
			asset->index = i;
			asset->filePath = getPArray(char, strLen(path) + 1);
			strCpy(asset->filePath, path);
			asset->folderIndex = folderIndex;
		}
	}

}

void reloadChangedFiles(HANDLE* folderHandles, Asset* assets, int assetCount) {
	// Todo: Get ReadDirectoryChangesW to work instead of FindNextChangeNotification.

	/*
	// HANDLE directory = CreateFile("..\\data\\Textures", FILE_LIST_DIRECTORY, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	// HANDLE directory = CreateFile("C:\\Projects", FILE_LIST_DIRECTORY, FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,  OPEN_EXISTING,  FILE_FLAG_BACKUP_SEMANTICS, NULL);
	// HANDLE directory = CreateFile("..\\data\\Textures", FILE_LIST_DIRECTORY, FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,  OPEN_EXISTING,  FILE_FLAG_BACKUP_SEMANTICS, NULL);
	HANDLE directory = CreateFile("..\\data\\Textures", FILE_LIST_DIRECTORY, FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,  OPEN_EXISTING,  FILE_FLAG_BACKUP_SEMANTICS, NULL);
	// HANDLE directory = CreateFile("C:\\Projects\\", FILE_LIST_DIRECTORY, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if(directory == INVALID_HANDLE_VALUE) {
		printf("Could not open directory.\n");
	}

	FILE_NOTIFY_INFORMATION notifyInformation[1024];
	DWORD bytesReturned = 0;
	bool result = ReadDirectoryChangesW(directory, (LPVOID)&notifyInformation, sizeof(notifyInformation), false, FILE_NOTIFY_CHANGE_LAST_WRITE, &bytesReturned, 0, 0);
	CloseHandle(directory);
	*/

	DWORD fileStatus = WaitForMultipleObjects(arrayCount(watchFolders), folderHandles, false, 0);
	if(between(fileStatus, WAIT_OBJECT_0, WAIT_OBJECT_0 + 9)) {
		// Note: WAIT_OBJECT_0 is defined as 0, so we can use it as an index.
		int folderIndex = fileStatus;

		FindNextChangeNotification(folderHandles[folderIndex]);


		for(int i = 0; i < assetCount; i++) {
			Asset* asset = assets + i;
			if(asset->folderIndex != folderIndex) continue;

			FILETIME newWriteTime = getLastWriteTime(asset->filePath);
			if(CompareFileTime(&asset->lastWriteTime, &newWriteTime) != 0) {

				// if(folderIndex == 0) {
				// 	loadTextureFromFile(theGraphicsState->textures + asset->index, texturePaths[asset->index], -1, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE, true);

				// } else if(folderIndex == 1) {
				// 		loadCubeMapFromFile(theGraphicsState->cubeMaps + asset->index, (char*)cubeMapPaths[asset->index], 5, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE, true);

				// } else if(folderIndex == 2) {
				// 	loadVoxelTextures(MINECRAFT_TEXTURE_FOLDER, INTERNAL_TEXTURE_FORMAT, true, asset->index);
				// }

				// if(folderIndex == 0) {
				// 	Texture* tex = theGraphicsState->textures;
				// 	loadTextureFromFile(theGraphicsState->textures + asset->index, texturePaths[asset->index], -1, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE, true);
				// }

				asset->lastWriteTime = newWriteTime;
			}
		}
	}

}

//

void debugMain(DebugState* ds, AppMemory* appMemory, AppData* ad, bool reload, bool* isRunning, bool init, ThreadQueue* threadQueue) {
	// @DebugStart.

	theMemory->debugMode = true;

	timerStart(&ds->tempTimer);

	Input* input = ds->input;
	WindowSettings* ws = &ad->wSettings;

	clearTMemoryDebug();

	ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[1];
	ExtendibleMemoryArray* pMemory = theMemory->pMemory;

	int clSize = megaBytes(2);
	drawCommandListInit(&ds->commandListDebug, (char*)getTMemoryDebug(clSize), clSize);
	theCommandList = &ds->commandListDebug;


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
		gui->start(ds->gInput, getFont("LiberationSans-Regular.ttf", fontSize), ws->currentRes);

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
			if(gui->button("Update Buffers")) ad->updateFrameBuffers = true;
			gui->div(vec2(0,0)); gui->label("FoV", 0); gui->slider(&ad->fieldOfView, 1, 180);
			gui->div(vec2(0,0)); gui->label("MSAA", 0); gui->slider(&ad->msaaSamples, 1, 8);
			gui->div(0,0,0); gui->label("NFPlane", 0); gui->slider(&ad->nearPlane, 0.01, 2); gui->slider(&ad->farPlane, 1000, 5000);
		} gui->endSection();

		static bool sectionEntities = false;
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

		VoxelWorldSettings* vs = &ad->voxelSettings;

		static bool sectionWorld = initSections;
		if(gui->beginSection("World", &sectionWorld)) { 
			if(gui->button("Reload World") || input->keysPressed[KEYCODE_TAB]) ad->reloadWorld = true;
			
			gui->div(vec2(0,0)); gui->label("RefAlpha", 0); gui->slider(&vs->reflectionAlpha, 0, 1);
			gui->div(vec2(0,0)); gui->label("Light", 0); gui->slider(&vs->globalLumen, 0, 255);
			gui->div(0,0,0);     gui->label("MinMax", 0); gui->slider(&vs->worldMin, 0, 255); gui->slider(&vs->worldMax, 0, 255);
			gui->div(vec2(0,0)); gui->label("WaterLevel", 0); 
			                     if(gui->slider(&vs->waterLevelValue, 0, 0.2f)) vs->waterLevelHeight = lerp(vs->waterLevelValue, vs->worldMin, vs->worldMax);

			gui->div(vec2(0,0)); gui->label("WFreq", 0);          gui->slider(&vs->worldFreq, 0.0001f, 0.02f);
			gui->div(vec2(0,0)); gui->label("WDepth", 0);         gui->slider(&vs->worldDepth, 1, 10);
			gui->div(vec2(0,0)); gui->label("MFreq", 0);          gui->slider(&vs->modFreq, 0.001f, 0.1f);
			gui->div(vec2(0,0)); gui->label("MDepth", 0);         gui->slider(&vs->modDepth, 1, 10);
			gui->div(vec2(0,0)); gui->label("MOffset", 0);        gui->slider(&vs->modOffset, 0, 1);
			gui->div(vec2(0,0)); gui->label("PowCurve", 0);       gui->slider(&vs->worldPowCurve, 1, 6);
			gui->div(0,0,0,0);   gui->slider(vs->heightLevels+0,0,1); gui->slider(vs->heightLevels+1,0,1);
								 gui->slider(vs->heightLevels+2,0,1); gui->slider(vs->heightLevels+3,0,1);
		} gui->endSection();

		gui->end();
	}

	// ds->timer->timerInfoCount = __COUNTER__;
	ds->timer->timerInfoCount = TIMER_INFO_COUNT;

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
			Vec3 color = hslToRgb(360*ss, 0.5f, 0.5f+h);

			info->color[0] = color.r;
			info->color[1] = color.g;
			info->color[2] = color.b;

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
		gui->start(ds->gInput, getFont("LiberationSans-Regular.ttf", fontHeight), ws->currentRes);

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
			if(ds->noCollating) {
				ds->timelineCamSize = -1;
				ds->timelineCamPos = -1;
				ds->setPause = true;
			}
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

		   		if(between(abs(ds->graphSortingIndex), 3, 7)) 
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
					float colorOffset = mapRange01((double)coh, stat->min, stat->max);
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
				clamp(&ds->timelineCamSize, 1000.0, cyclesSize);
				double diff = ds->timelineCamSize - oldZoom;

				float zoomOffset = mapRange(input->mousePos.x, bgRect.min.x, bgRect.max.x, -0.5f, 0.5f);
				ds->timelineCamPos -= diff*zoomOffset;
			}


			Vec2 dragDelta = vec2(0,0);
			gui->drag(bgRect, &dragDelta, vec4(0,0,0,0));

			ds->timelineCamPos -= dragDelta.x * (ds->timelineCamSize/graphWidth);
			clamp(&ds->timelineCamPos, cyclesLeft + ds->timelineCamSize/2, cyclesRight - ds->timelineCamSize/2);


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
					float viewAreaLeft = mapRange((float)orthoLeft, (float)cyclesLeft, (float)cyclesRight, cyclesRect.left, cyclesRect.right);
					float viewAreaRight = mapRange((float)orthoRight, (float)cyclesLeft, (float)cyclesRight, cyclesRect.left, cyclesRect.right);

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

				clampMax(&heightMod, 1.0f);

				dcState(STATE_LINEWIDTH, 3);
				double startPos = roundMod(orthoLeft, timelineSection) - timelineSection;
				double pos = startPos;
				while(pos < orthoRight + timelineSection) {
					double p = mapRange((float)pos, (float)orthoLeft, (float)orthoRight, bgRect.left, bgRect.right);

					// Big line.
					{
						float h = headerDim.h*heightMod;
						dcLine2d(vec2(p,headerRect.min.y), vec2(p,headerRect.min.y + h), vec4(g,g,g,1));
					}

					// Text
					{
						Vec2 textPos = vec2(p,cyclesRect.min.y + cyclesDim.h/2);
						float percent = mapRange(pos, cyclesLeft, cyclesRight, 0.0, 100.0);
						int percentInterval = mapRange(timelineSection, 0.0, cyclesSize, 0.0, 100.0);

						char* s;
						if(percentInterval > 10) s = fillString("%i%%", (int)percent);
						else if(percentInterval > 1) s = fillString("%.1f%%", percent);
						else if(percentInterval > 0.1) s = fillString("%.2f%%", percent);
						else s = fillString("%.3f%%", percent);

						float tw = getTextDim(s, gui->font).w;
						if(between(bgRect.min.x, textPos.x - tw/2, textPos.x + tw/2)) textPos.x = bgRect.min.x + tw/2;
						if(between(bgRect.max.x, textPos.x - tw/2, textPos.x + tw/2)) textPos.x = bgRect.max.x - tw/2;

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
						double p = mapRange(pos, orthoLeft, orthoRight, (double)bgRect.left, (double)bgRect.right);
						float h = headerDim.h*heightMod;
						dcLine2d(vec2(p,headerRect.min.y), vec2(p,headerRect.min.y + h), vec4(g,g,g,1));
					}

					// Cycle text.
					{
						float pMid = mapRange(pos - timelineSection/2.0, orthoLeft, orthoRight, (double)bgRect.left, (double)bgRect.right);
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


					double barLeft = mapRange((double)slot->cycles, orthoLeft, orthoRight, (double)bgRect.left, (double)bgRect.right);
					double barRight = mapRange((double)slot->cycles + slot->size, orthoLeft, orthoRight, (double)bgRect.left, (double)bgRect.right);

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

				float statMin = mapRange((float)stat->min, orthoBottom, orthoTop, rBottom, rTop);
				float statMax = mapRange((float)stat->max, orthoBottom, orthoTop, rBottom, rTop);
				if(statMax < rBottom || statMin > rTop) continue;

				Vec4 color = vec4(info->color[0], info->color[1], info->color[2], 1);

				float yAvg = mapRange((float)stat->avg, orthoBottom, orthoTop, rBottom, rTop);
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
				else p.y = mapRange((float)ds->timings[0][timerIndex].cyclesOverHits, orthoBottom, orthoTop, rBottom, rTop);
				for(int i = 1; i < cycleCount; i++) {
					Timings* t = &ds->timings[i][timerIndex];

					bool lastElementEmpty = false;
					if(t->cyclesOverHits == 0) {
						if(i != cycleCount-1) continue;
						else lastElementEmpty = true;
					}

					float y = mapRange((float)t->cyclesOverHits, orthoBottom, orthoTop, rBottom, rTop);
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

		con->update(ds->input, vec2(ws->currentRes), ds->fontHeight, ad->dt, smallExtension, bigExtension);

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
		Font* font = getFont("LiberationSans-Regular.ttf", fontSize);
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
		Font* font = getFont("consola.ttf", fontSize);
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
		dcText(fillString("Chunk: (%i,%i)", PVEC2(coordToMesh(ad->player->pos))), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("Chunk: (%i,%i)", PVEC2(ad->player->chunk)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
		dcText(fillString("ChunkOff: (%i,%i)", PVEC2(ad->chunkOffset)), font, tp, c, ali, 0, sh, c2); tp.y -= fontSize;
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
		if(theDebugState->gui2) guiSave(theDebugState->gui2, 2, 3);
	}

	// Update debugTime every second.
	static f64 tempTime = 0;
	tempTime += ds->dt;
	if(tempTime >= 1) {
		ds->debugTime = timerStop(&ds->tempTimer);
		tempTime = 0;
	}
}