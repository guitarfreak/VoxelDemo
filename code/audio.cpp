
struct AudioState;
extern AudioState* theAudioState;

// WAVE_FORMAT_PCM
// WAVE_FORMAT_IEEE_FLOAT

// typedef struct tWAVEFORMATEX {
//   WORD  wFormatTag;
//   WORD  nChannels;
//   DWORD nSamplesPerSec;
//   DWORD nAvgBytesPerSec;
//   WORD  nBlockAlign;
//   WORD  wBitsPerSample;
//   WORD  cbSize;
// } WAVEFORMATEX;

struct WaveFileHeader {
	char chunkId[4];
	int chunkSize;
	char format[4];

	char subchunk1Id[4];
	int subchunk1Size;
	short audioFormat;
	short numChannels;
	int sampleRate;
	int byteRate;
	short blockAlign;
	short bitsPerSample;
	// short cbSize;
	// short validBitsPerSample;
	// int channelMask;
	// char subFormat[16];

	// char subchunk2Id[4];
	// int subchunk2Size;
	// short data;
};

struct WaveFileChunk {
	char id[4];
	int size;
};

struct Audio {
	char* name;

	char* file;

	int channels;
	int sampleRate;
	int bitsPerSample;

	short* data;
	int frameCount;
	int totalLength;
};

struct Track {
	Audio* audio;

	float volume;

	bool used;

	bool playing;
	bool paused;
	int index;
	float speed;
};

struct AudioState {
	IMMDeviceEnumerator* deviceEnumerator;
	IMMDevice* immDevice;
	IAudioClient* audioClient;
	IAudioRenderClient* renderClient;
	float latencyInSeconds;

	WAVEFORMATEX* waveFormat;
	uint bufferFrameCount;
	float latency;
	int channelCount;
	int sampleRate;

	//

	Audio* files;
	int fileCount;
	int fileCountMax;

	// Mixer.

	float masterVolume;
	Track tracks[10];
};

void addAudio(AudioState* as, char* filePath, char* name) {

	if(as->fileCount == as->fileCountMax) return;

	// Only load wav files.

	char* extension = getFileExtension(filePath);
	if(!extension) return;

	int result = strFind(extension, "wav");
	if(result == -1) return;

	int size = fileSize(filePath);
	char* file = (char*)getPMemory(size);
	readFileToBuffer(file, filePath);

	WaveFileHeader* waveHeader = (WaveFileHeader*)file;

	// Only accept specific wave formats for now.
	assert(waveHeader->audioFormat == WAVE_FORMAT_PCM);
	assert(waveHeader->bitsPerSample == 16);

	WaveFileChunk* chunk = (WaveFileChunk*)(waveHeader+1);

	char* dataString = "data";
	bool dataSection = true;

	while(!strCompare("data", chunk->id, 4)) {
		chunk = (WaveFileChunk*)((char*)chunk + sizeof(WaveFileChunk) + chunk->size);
	}

	int dataSize = chunk->size;
	short* data = (short*)(chunk+1);

	Audio audio = {};

	audio.name = getPStringCpy(name);
	audio.file = file;
	audio.data = data;
	audio.sampleRate = waveHeader->sampleRate;
	audio.channels = waveHeader->numChannels;
	audio.bitsPerSample = waveHeader->bitsPerSample;

	int bytesPerSample = audio.bitsPerSample/8;
	audio.totalLength = dataSize/bytesPerSample;
	audio.frameCount = audio.totalLength/audio.channels;

	as->files[as->fileCount++] = audio;
}

Audio* getAudio(AudioState* as, char* name) {
	for(int i = 0; i < as->fileCount; i++) {
		if(strCompare(as->files[i].name, name)) {
			return (as->files + i);
		}
	}

	return 0;
}

#define PITCH_PERCENT 0.05946309435929526456

void addTrack(Audio* audio, float volume = 1.0f, bool modulate = false) {
	AudioState* as = theAudioState;

	Track* track = 0;
	for(int i = 0; i < arrayCount(as->tracks); i++) {
		if(!as->tracks[i].used) {
			track = as->tracks + i;
		}
	}

	if(!track) return;

	track->used = true;
	track->audio = audio;
	track->playing = true;
	track->paused = false;
	track->index = 0;
	track->speed = 1;

	if(modulate) {
		float percent = 1 + (PITCH_PERCENT * 0.25f);
		track->speed = randomFloatPCG(1/percent, percent, 0.00001f);
	}

	track->volume = volume;
}

void addTrack(char* name, float volume = 1.0f, bool modulate = false) {
	Audio* audio = getAudio(theAudioState, name);
	if(!audio) return;

	return addTrack(audio, volume, modulate);
}
