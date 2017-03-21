
#define DEBUG_NOTE_LENGTH 50
#define DEBUG_NOTE_DURATION 3

struct Asset;
struct DebugState {
	Asset* assets;
	int assetCount;

	i64 lastTimeStamp;
	f64 dt;
	f64 time;

	f64 debugTime;
	f64 debugRenderTime;

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

void addDebugNote(char* string, float duration = DEBUG_NOTE_DURATION) {
	DebugState* ds = globalDebugState;

	assert(strLen(string) < DEBUG_NOTE_LENGTH);
	if(ds->notificationCount >= arrayCount(ds->notificationStack)) return;

	int count = ds->notificationCount;
	strClear(ds->notificationStack[count]);
	ds->notificationTimes[count] = duration;
	strCpy(ds->notificationStack[count], string);
	ds->notificationCount++;
}

void addDebugInfo(char* string) {
	DebugState* ds = globalDebugState;

	if(ds->infoStackCount >= arrayCount(ds->infoStack)) return;
	ds->infoStack[ds->infoStackCount++] = string;
}



struct Asset {
	int index;
	int folderIndex;
	char* filePath;
	FILETIME lastWriteTime;
};

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
	if(valueBetweenInt(fileStatus, WAIT_OBJECT_0, WAIT_OBJECT_0 + 9)) {
		// Note: WAIT_OBJECT_0 is defined as 0, so we can use it as an index.
		int folderIndex = fileStatus;

		FindNextChangeNotification(folderHandles[folderIndex]);


		for(int i = 0; i < assetCount; i++) {
			Asset* asset = assets + i;
			if(asset->folderIndex != folderIndex) continue;

			FILETIME newWriteTime = getLastWriteTime(asset->filePath);
			if(CompareFileTime(&asset->lastWriteTime, &newWriteTime) != 0) {

				if(folderIndex == 0) {
					loadTextureFromFile(globalGraphicsState->textures + asset->index, texturePaths[asset->index], -1, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE, true);

				} else if(folderIndex == 1) {
						loadCubeMapFromFile(globalGraphicsState->cubeMaps + asset->index, (char*)cubeMapPaths[asset->index], 5, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE, true);

				} else if(folderIndex == 2) {
					loadVoxelTextures((char*)minecraftTextureFolderPath, INTERNAL_TEXTURE_FORMAT, true, asset->index);

				}

				asset->lastWriteTime = newWriteTime;
			}
		}
	}

}