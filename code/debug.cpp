
struct DebugState {
	bool showHud;
	bool showConsole;

	DrawCommandList commandListDebug;
	Input* input;

	bool isInitialised;
	int timerInfoCount;
	TimerInfo timerInfos[32]; // timerInfoCount
	TimerSlot* timerBuffer;
	int bufferSize;
	u64 bufferIndex;

	Timings timings[120][32];
	int cycleIndex;

	bool frozenGraph;
	TimerSlot* savedTimerBuffer;
	u64 savedBufferIndex;
	Timings savedTimings[32];

	GuiInput gInput;
	Gui* gui;
	Gui* gui2;

	Input* recordedInput;
	int inputCapacity;
	bool recordingInput;
	int inputIndex;

	bool playbackStart;
	bool playbackInput;
	int playbackIndex;
	bool playbackSwapMemory;

	// Console.

	float consolePos;
	float targetPos;
	bool consoleActive;

	char* consoleBuffer[256];
	int consoleBufferSize;

	char inputBuffer[256];
	// int inputBufferSize;

	int cursorPos;
};

#define CONSOLE_SMALL_PERCENT 0.3f
#define CONSOLE_BIG_PERCENT 0.8f


extern DebugState* globalDebugState;

inline uint getThreadID() {
	char *threadLocalStorage = (char *)__readgsqword(0x30);
	uint threadID = *(uint *)(threadLocalStorage + 0x48);

	return threadID;
}

void addTimerSlot(int timerIndex, int type) {
	// uint id = atomicAdd64(&globalDebugState->bufferIndex, 1);
	uint id = globalDebugState->bufferIndex++;
	TimerSlot* slot = globalDebugState->timerBuffer + id;
	slot->cycles = __rdtsc();
	slot->type = type;
	slot->threadId = getThreadID();
	slot->timerIndex = timerIndex;
}

void addTimerSlotAndInfo(int timerIndex, int type, const char* file, const char* function, int line, char* name = "") {

	TimerInfo* timerInfo = globalDebugState->timerInfos + timerIndex;

	if(!timerInfo->initialised) {
		timerInfo->initialised = true;
		timerInfo->file = file;
		timerInfo->function = function;
		timerInfo->line = line;
		timerInfo->type = type;
		timerInfo->name = name;
	}

	addTimerSlot(timerIndex, type);
}

struct TimerBlock {
	int counter;

	TimerBlock(int counter, const char* file, const char* function, int line, char* name = "") {

		this->counter = counter;
		addTimerSlotAndInfo(counter, TIMER_TYPE_BEGIN, file, function, line, name);
	}

	~TimerBlock() {
		addTimerSlot(counter, TIMER_TYPE_END);
	}
};

#define TIMER_BLOCK() \
	TimerBlock timerBlock##__LINE__(__COUNTER__, __FILE__, __FUNCTION__, __LINE__);

#define TIMER_BLOCK_NAMED(name) \
	TimerBlock timerBlock##__LINE__(__COUNTER__, __FILE__, __FUNCTION__, __LINE__, name);

#define TIMER_BLOCK_BEGIN(ID) \
	const int timerCounter##ID = __COUNTER__; \
	addTimerSlotAndInfo(timerCounter##ID, TIMER_TYPE_BEGIN, __FILE__, __FUNCTION__, __LINE__); 

#define TIMER_BLOCK_BEGIN_NAMED(ID, name) \
	const int timerCounter##ID = __COUNTER__; \
	addTimerSlotAndInfo(timerCounter##ID, TIMER_TYPE_BEGIN, __FILE__, __FUNCTION__, __LINE__, name); 

#define TIMER_BLOCK_END(ID) \
	addTimerSlot(timerCounter##ID, TIMER_TYPE_END);

struct Statistic {
	f64 min;
	f64 max;
	f64 avg;
	int count;
};

void beginStatistic(Statistic* stat) {
	stat->min = DBL_MAX; 
	stat->max = -DBL_MAX; 
	stat->avg = 0; 
	stat->count = 0; 
}

void updateStatistic(Statistic* stat, f64 value) {
	if(value < stat->min) stat->min = value;
	if(value > stat->max) stat->max = value;
	stat->avg += value;
	++stat->count;
}

void endStatistic(Statistic* stat) {
	stat->avg /= stat->count;
}

struct TimerStatistic {
	u64 cycles;
	int timerIndex;
};


