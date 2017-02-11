

#define CONSOLE_SMALL_PERCENT 0.3f
#define CONSOLE_BIG_PERCENT 0.8f

struct Console {
	float pos;
	float targetPos;
	bool isActive;

	char* mainBuffer[256];
	int mainBufferSize;

	char inputBuffer[256];
	// int inputBufferSize;

	int cursorIndex;
	int markerIndex;
	float cursorTime;

	void init(float windowHeight) {
		float smallPos = -windowHeight * CONSOLE_SMALL_PERCENT;
		pos = smallPos;
		targetPos = smallPos;
		cursorIndex = 0;
		markerIndex = 0;

		const char* text = "This is a test String!";
		mainBuffer[mainBufferSize] = getPArrayDebug(char, strLen((char*)text) + 1);
		strCpy(mainBuffer[mainBufferSize++], (char*)text);

		const char* result = "123.4543";
		mainBuffer[mainBufferSize] = getPArrayDebug(char, strLen((char*)result) + 1);
		strCpy(mainBuffer[mainBufferSize++], (char*)result);


		const char* text2 = "This doesn't produce a result.";
		mainBuffer[mainBufferSize] = getPArrayDebug(char, strLen((char*)text2) + 1);
		strCpy(mainBuffer[mainBufferSize++], (char*)text2);

		mainBuffer[mainBufferSize] = getPArrayDebug(char, 1);
		strClear(mainBuffer[mainBufferSize++]);



		mainBuffer[mainBufferSize] = getPArrayDebug(char, strLen((char*)text) + 1);
		strCpy(mainBuffer[mainBufferSize++], (char*)text);

		mainBuffer[mainBufferSize] = getPArrayDebug(char, strLen((char*)result) + 1);
		strCpy(mainBuffer[mainBufferSize++], (char*)result);


	}

	void update(Input* input, Vec2 currentRes, float dt) {
		float closedPos = 0;
		float smallPos = -currentRes.y * CONSOLE_SMALL_PERCENT;
		float bigPos = -currentRes.y * CONSOLE_BIG_PERCENT;

		if(input->keysPressed[KEYCODE_F6]) {
			if(input->keysDown[KEYCODE_CTRL]) {
				if(targetPos == closedPos) {
					targetPos = bigPos;
				} else if(targetPos == smallPos) {
					targetPos = bigPos;
				} else targetPos = closedPos;
			} else {
				if(targetPos == closedPos) {
					targetPos = smallPos;
				} else if(targetPos == bigPos) {
					targetPos = smallPos;
				} else targetPos = closedPos;
			}
		}

		// Calculate smoothstep.

		float consoleSpeed = 10;
		float consoleMovement = consoleSpeed * dt;

		float distance = pos - targetPos;
		if(abs(distance) > 0) {
			pos = lerp(consoleMovement, pos, targetPos);

			// Clamp if we overstepped the target position.
			float newDistance = pos - targetPos;
			if(abs(newDistance) <= 1.0f) pos = targetPos;

			// if(newDistance != 0 && (!sameSign(distance, newDistance))) {
				// pos = targetPos;
			// }
		}

		Vec2 res = currentRes;
		// float consoleTotalHeight = abs(smallPos);
		float consoleTotalHeight = abs(pos);
		clampMin(&consoleTotalHeight, abs(smallPos));

		// Font* bodyFont = getFont(FONT_SOURCESANS_PRO, bodyFontSize);
		// Font* bodyFont = getFont(FONT_CALIBRI, bodyFontSize);
		float bodyFontHeightPadding = 1.0;
		float bodyFontSize = 22;
		Font* bodyFont = getFont(FONT_SOURCESANS_PRO, bodyFontSize);
		float inputFontSize = 16;
		Font* inputFont = getFont(FONT_CONSOLAS, inputFontSize);

		float inputHeightPadding = 1.5;
		float fontDrawHeightOffset = 0.2f;
		float consolePadding = 10;
		float cursorWidth = 2;
		float cursorSpeed = 3;
		float cursorColorMod = 0.2f;


		Vec4 bodyColor 		= vec4(0.3f,0,0.6f,1.0f);
		Vec4 inputColor 	= vec4(0.37f,0,0.6f,1.0f);
		Vec4 bodyFontColor 	= vec4(1,1,1,1);
		Vec4 inputFontColor = vec4(1,1,1,1);
		Vec4 cursorColor 	= vec4(0.2f,0.8f,0.1f,0.9f);
		Vec4 selectionColor = vec4(0.1f,0.4f,0.4f,0.9f);



		float inputHeight = inputFontSize * inputHeightPadding;
		float bodyTextHeight = bodyFontSize * bodyFontHeightPadding;
		float consoleBodyHeight = consoleTotalHeight - inputHeight;
		Rect consoleBody = rectMinDim(vec2(0, pos + inputHeight), vec2(res.w, consoleBodyHeight));
		Rect consoleInput = rectMinDim(vec2(0, pos), vec2(res.w, inputHeight));

		if(pointInRect(input->mousePosNegative, consoleInput)) {
			float t = 0.1f;
			inputColor += vec4(t,t,t,0);

			if(input->mouseButtonPressed[0]) {
				isActive = true;
				strClear(inputBuffer);
			}
		}

		if(input->keysPressed[KEYCODE_ESCAPE]) {
			isActive = false;
		}

		bool visible = true;
		if(pos == closedPos) {
			isActive = false;
			visible = false;
		}

		if(visible) {
			dcRect(consoleBody, bodyColor);
			dcRect(consoleInput, inputColor);

			if(mainBufferSize > 0) {
				dcEnable(STATE_SCISSOR);
				dcScissor(scissorRectScreenSpace(consoleBody, res.h));

				float textPos = pos + consoleTotalHeight -bodyTextHeight/2;

				for(int i = 0; i < mainBufferSize; i++) {
					if(i%2 == 0) {
						char* pre = "> ";
						dcText(pre, bodyFont, vec2(consolePadding,textPos), bodyFontColor, 0, 1);

						dcText(mainBuffer[i], bodyFont, vec2(consolePadding + getTextDim(pre, bodyFont).w,textPos), bodyFontColor, 0, 1);
						textPos -= bodyTextHeight;
					} else {
						if(strLen(mainBuffer[i]) > 0) {
							char* pre = "    ";
							dcText(pre, bodyFont, vec2(consolePadding,textPos), bodyFontColor, 0, 1);

							dcText(mainBuffer[i], bodyFont, vec2(consolePadding + getTextDim(pre, bodyFont).w,textPos), vec4(1,0,0,1), 0, 1);

							textPos -= bodyTextHeight;
						}
					}
				}

				dcDisable(STATE_SCISSOR);
			}
		}

		// bool isActive = pos != closedPos;
		if(isActive) {
			inputColor.g += 0.2;

			int startCursorIndex = cursorIndex;

			if(input->keysPressed[KEYCODE_LEFT]) {
				if(input->keysDown[KEYCODE_CTRL]) {
					if(cursorIndex > 0) {
						cursorIndex = strFindBackwards(inputBuffer, ' ', cursorIndex-1);
					}
				} else cursorIndex--;
			}

			if(input->keysPressed[KEYCODE_RIGHT]) {
				if(input->keysDown[KEYCODE_CTRL]) {
					if(cursorIndex <= strLen(inputBuffer)) {
						cursorIndex = strFindOrEnd(inputBuffer, ' ', cursorIndex+1);
						if(cursorIndex != strLen(inputBuffer)) cursorIndex--;
					} else cursorIndex++;
				} else cursorIndex++;
			}

			if(input->keysPressed[KEYCODE_HOME]) {
				cursorIndex = 0;
			}

			if(input->keysPressed[KEYCODE_END]) {
				cursorIndex = strLen(inputBuffer);
			}

			cursorIndex = clamp(cursorIndex, 0, strLen(inputBuffer));

			if((startCursorIndex != cursorIndex) && !input->keysDown[KEYCODE_SHIFT]) {
				markerIndex = cursorIndex;
			}

			bool isSelected = cursorIndex != markerIndex;

			if(input->keysPressed[KEYCODE_BACKSPACE] || input->keysPressed[KEYCODE_DEL] || 
			   (input->inputCharacterCount > 0)) {
				if(isSelected) {
					int delIndex = min(cursorIndex, markerIndex);
					int delAmount = abs(cursorIndex - markerIndex);
					strRemoveX(inputBuffer, delIndex, delAmount);
					cursorIndex = delIndex;
				}

				markerIndex = cursorIndex;
			}

			// Add input characters to input buffer.
			if(input->inputCharacterCount > 0) {
				if(input->inputCharacterCount + strLen(inputBuffer) < arrayCount(inputBuffer)) {
					strInsert(inputBuffer, cursorIndex, input->inputCharacters, input->inputCharacterCount);
					cursorIndex += input->inputCharacterCount;
					markerIndex = cursorIndex;
				}
			}

			if(input->keysPressed[KEYCODE_BACKSPACE] && !isSelected) {
				if(cursorIndex > 0) {
					strRemove(inputBuffer, cursorIndex);
					cursorIndex--;
				}
				markerIndex = cursorIndex;
			}

			if(input->keysPressed[KEYCODE_DEL] && !isSelected) {
				if(cursorIndex+1 <= strLen(inputBuffer)) {
					strRemove(inputBuffer, cursorIndex+1);
				}
				markerIndex = cursorIndex;
			}

			if(input->keysPressed[KEYCODE_RETURN]) {
				if(strLen(inputBuffer) > 0) {
					// Copy over input buffer to console buffer.
					pushToMainBuffer(inputBuffer);
					evaluateInput();
					strClear(inputBuffer);
				}
			}

			if(startCursorIndex != cursorIndex) {
				cursorTime = 0;
			}

			// Drawing.

			float inputMid = pos + inputHeight/2;

			float cursorX = getTextPos(inputBuffer, cursorIndex, inputFont) + consolePadding;
			Rect cursorRect = rectCenDim(cursorX, inputMid, cursorWidth, inputFontSize);
			if(cursorIndex == strLen(inputBuffer)) {
				float width = getTextDim("M", inputFont).w;
				cursorRect = rectCenDim(cursorX + width/2, inputMid, width, inputFontSize);
			}

			float markerX = getTextPos(inputBuffer, markerIndex, inputFont) + consolePadding;

			if(cursorIndex != markerIndex) {
				float selectionWidth = abs(cursorX - markerX);
				Vec2 selectionMid = vec2(min(cursorX, markerX) + selectionWidth/2, inputMid);
				Rect selectionRect = rectCenDim(selectionMid, vec2(selectionWidth, inputFontSize));

				dcRect(selectionRect, selectionColor);
			}

			dcText(inputBuffer, inputFont, vec2(consolePadding, consoleInput.min.y + inputHeight/2 + inputFontSize * fontDrawHeightOffset), inputFontColor, 0, 1);

			cursorTime += dt*cursorSpeed;
			Vec4 cmod = vec4(0,cos(cursorTime)*cursorColorMod - cursorColorMod,0,0);
			dcRect(cursorRect, cursorColor + cmod);
		}
	}

	char* eatWhiteSpace(char* str) {
		int index = 0;
		while(str[index] == ' ') index++;
		return str + index;
	}

	char* getNextArgument(char** s) {
		*s = eatWhiteSpace(*s);
		char* str = *s;

		if(str[0] == '\0') return 0;
		int wpos = strFindX(str, ' ');
		if(wpos == -1) wpos = strLen(str) + 1;
		wpos--;

		char* argument = getTStringDebug(wpos + 1);
		strCpy(argument, str, wpos);

		*s += wpos;

		return argument;
	}

	void pushToMainBuffer(char* str) {
		char* newString = getPArrayDebug(char, strLen(str) + 1);
		strCpy(newString, str);

		mainBuffer[mainBufferSize] = newString;
		mainBufferSize++;
	}

	void evaluateInput() {
		char* com = inputBuffer;

		char* argument = getNextArgument(&com);
		printf("Argument: %s.", argument);
		if(argument == 0) return;

		if(strCompare(argument, "add")) {
			int a = strToInt(getNextArgument(&com));
			int b = strToInt(getNextArgument(&com));

			pushToMainBuffer(fillString("%i + %i = %i", a, b, a+b));
		} else if(strCompare(argument, "donothing")) {
			pushToMainBuffer("");
		} else {
			pushToMainBuffer(fillString("Unknown command \"%s\"", argument));
		}
	}
};




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

	Console console;
};




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


