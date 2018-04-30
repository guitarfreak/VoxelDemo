
#define PITCH_PERCENT 0.05946309435929526456

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

	i64 startTime;
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
	Track tracks[20];

	float defaultModulation = 0.25;
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


void addTrack(Audio* audio, float volume = 1.0f, bool modulate = false, float modulationAmount = 0) {
	AudioState* as = theAudioState;

	if(modulate && modulationAmount == 0) modulationAmount = as->defaultModulation; 

	Track* track = 0;
	for(int i = 0; i < arrayCount(as->tracks); i++) {
		if(!as->tracks[i].used) {
			track = as->tracks + i;
		}
	}

	// Get oldest track and overwrite with new one.
	if(!track) {
		i64 oldestStartTime = LLONG_MAX;
		for(int i = 0; i < arrayCount(as->tracks); i++) {
			if(as->tracks[i].startTime < oldestStartTime) {
				oldestStartTime = as->tracks[i].startTime;
				track = as->tracks + i;
			}
		}
	}

	track->used = true;
	track->audio = audio;
	track->playing = true;
	track->paused = false;
	track->index = 0;
	track->speed = 1;

	LARGE_INTEGER timeStamp;
	QueryPerformanceCounter(&timeStamp);
	track->startTime = timeStamp.QuadPart;

	if(modulate) {
		float percent = 1 + (PITCH_PERCENT * modulationAmount);
		track->speed = randomFloatPCG(1/percent, percent, 0.00001f);
	}

	track->volume = volume;
}

void addTrack(char* name, float volume = 1.0f, bool modulate = false, float modulationAmount = 0) {
	Audio* audio = getAudio(theAudioState, name);
	if(!audio) return;

	return addTrack(audio, volume, modulate, modulationAmount);
}


void updateAudio(AudioState* as, float dt) {
	{
		TIMER_BLOCK_NAMED("Audio");

		int framesPerFrame = dt * as->waveFormat->nSamplesPerSec * as->latency;

		uint numFramesPadding;
		as->audioClient->GetCurrentPadding(&numFramesPadding);
		uint numFramesAvailable = as->bufferFrameCount - numFramesPadding;
		framesPerFrame = min(framesPerFrame, numFramesAvailable);

		// int framesToPush = framesPerFrame - numFramesPadding;
		// printf("%i %i %i %i\n", numFramesAvailable, numFramesPadding, as->bufferFrameCount, framesToPush);

		// numFramesPadding = framesPerFrame;
		// framesPerFrame = framesToPush;

		// if(framesPerFrame && framesToPush > 0) 
		if(framesPerFrame) 
		{

			float* buffer;
			as->renderClient->GetBuffer(numFramesAvailable, (BYTE**)&buffer);

			// Clear to zero.

			for(int i = 0; i < numFramesAvailable*2; i++) buffer[i] = 0.0f;

			for(int trackIndex = 0; trackIndex < arrayCount(as->tracks); trackIndex++) {
				Track* track = as->tracks + trackIndex;
				Audio* audio = track->audio;

				if(!track->used) continue;


				int index = track->index;
				int channels = audio->channels;

				int normalFrameCount = audio->frameCount;
				float speed = track->speed * ((float)audio->sampleRate / as->sampleRate);
				int frameCount = audio->frameCount;
				if(speed != 1.0f) {
					frameCount = roundFloat((frameCount-1) * (float)(1/speed)) + 1;
				}

				int availableFrames = min(frameCount - index, numFramesAvailable);

				float volume = as->masterVolume * track->volume;
				for(int i = 0; i < availableFrames; i++) {
					for(int channelIndex = 0; channelIndex < channels; channelIndex++) {
						float finalValue;

						if(speed == 1.0f) {
							short value = audio->data[(index + i)*channels + channelIndex];
							finalValue = (float)value/SHRT_MAX;

						} else {
							float fIndex = (index + i) * speed;

							if(fIndex > (normalFrameCount-1)) break;

							int leftIndex = (int)fIndex;
							int rightIndex = leftIndex + 1;

							short leftValue = audio->data[leftIndex*channels + channelIndex];
							short rightValue = audio->data[rightIndex*channels + channelIndex];

							float fLeftValue = (float)leftValue/SHRT_MAX;
							float fRightValue = (float)rightValue/SHRT_MAX;

							float percent = fIndex - leftIndex;
							finalValue = fLeftValue + (fRightValue-fLeftValue) * percent;

						}

						buffer[(i*2) + channelIndex] += finalValue * volume;
						if(channels == 1) buffer[(i*2) + 1] += finalValue * volume;
					}
				}

				track->index += framesPerFrame;

				if(track->index >= audio->frameCount) {
					track->used = false;
				}
			}

			as->renderClient->ReleaseBuffer(framesPerFrame, 0);
		}
	}
}