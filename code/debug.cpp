

#define CONSOLE_SMALL_PERCENT 0.3f
#define CONSOLE_BIG_PERCENT 0.8f

struct Console {
	float pos;
	float targetPos;
	int mode;
	bool isActive;

	float scrollMode;
	float scrollPercent;
	float lastDiff;

	char* mainBuffer[256];
	int mainBufferSize;

	char inputBuffer[256];
	// int inputBufferSize;

	int cursorIndex;
	int markerIndex;
	bool mouseSelectMode;
	float cursorTime;

	/*
		TODO's:
		* Ctrl backspace, Ctrl Delete.
		* Ctrl Left/Right not jumping multiple spaces.
		* Selection Left/Right not working propberly.
		* Ctrl + a.
		* Clipboard.
		* Mouse selection.
		* Mouse release.
		* Cleanup.
		- Scrollbar.

		- Command history.
		- Command hint on tab.
	*/

	void init(float windowHeight) {
		float smallPos = -windowHeight * CONSOLE_SMALL_PERCENT;
		pos = 0;
		mode = 1;
		targetPos = smallPos;
		cursorIndex = 0;
		markerIndex = 0;
		scrollPercent = 1;

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

		// Console variables.

		float consoleSpeed = 10;

		float bodyFontHeightPadding = 1.0f;
		float bodyFontHeightResultPadding = 1.2f;
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

		float scrollBarWidth = 20;
		float scrollCursorMinHeight = 20;

		// Logic.

		float closedPos = 0;
		float smallPos = -currentRes.y * CONSOLE_SMALL_PERCENT;
		float bigPos = -currentRes.y * CONSOLE_BIG_PERCENT;

		if(input->keysPressed[KEYCODE_F6]) {
			if(input->keysDown[KEYCODE_CTRL]) {
				if(mode == 0) mode = 2;
				else if(mode == 1) mode = 2;
				else mode = 0;
			} else {
				if(mode == 0) mode = 1;
				else if(mode == 2) mode = 1;
				else mode = 0;
			}
		}

		if(mode == 0) targetPos = 0;
		else if(mode == 1) targetPos = smallPos;
		else if(mode == 2) targetPos = bigPos;

		// Calculate smoothstep.

		float distance = pos - targetPos;
		if(abs(distance) > 0) {
			float consoleMovement = consoleSpeed * dt;
			pos = lerp(consoleMovement, pos, targetPos);

			// Clamp if we overstepped the target position.
			float newDistance = pos - targetPos;
			if(abs(newDistance) <= 1.0f) pos = targetPos;
		}



		Vec2 res = currentRes;
		float consoleTotalHeight = abs(pos);
		clampMin(&consoleTotalHeight, abs(smallPos));

		float inputHeight = inputFontSize * inputHeightPadding;
		float bodyTextHeight = bodyFontSize * bodyFontHeightPadding;
		float consoleBodyHeight = consoleTotalHeight - inputHeight;
		Rect consoleBody = rectMinDim(vec2(0, pos + inputHeight), vec2(res.w, consoleBodyHeight));
		Rect consoleInput = rectMinDim(vec2(0, pos), vec2(res.w, inputHeight));
		Rect scrollRect = consoleBody;
		scrollRect.min.x = scrollRect.max.x - scrollBarWidth;

		if(!isActive && pointInRect(input->mousePosNegative, consoleInput)) {
			float t = 0.1f;
			inputColor += vec4(t,t,t,0);

			if(input->mouseButtonPressed[0]) {
				isActive = true;
				strClear(inputBuffer);
				cursorIndex = 0;
				markerIndex = 0;
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


			if(input->mouseButtonPressed[0] && pointInRect(input->mousePosNegative, scrollRect)) {
				scrollMode = true;
			}

			if(input->mouseButtonReleased[0]) {
				scrollMode = false;
			}

			if(scrollMode) {
				scrollPercent = mapRangeClamp(input->mousePosNegative.y, scrollRect.min.y, scrollRect.max.y, 0, 1);
				scrollPercent = 1-scrollPercent;
			}


			if(mainBufferSize > 0) {
				dcEnable(STATE_SCISSOR);
				dcScissor(scissorRectScreenSpace(consoleBody, res.h));

				float scrollOffset = 0;
				if(lastDiff > 0) scrollOffset = scrollPercent*lastDiff; 

				float textPos = pos + consoleTotalHeight -bodyTextHeight/2 + scrollOffset;
				float textStart = textPos;

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

							textPos -= bodyTextHeight * bodyFontHeightResultPadding;
						}
					}
				}



				if(scrollOffset != 0) {
					dcRect(scrollRect, vec4(1,1,1,1));

					float consoleHeight = rectGetDim(consoleBody).h;
					float scrollCursorHeight = consoleHeight / lastDiff * consoleHeight;
					clampMin(&scrollCursorHeight, scrollCursorMinHeight);

					float scrollCursorPos = lerp(scrollPercent, scrollRect.max.y, scrollRect.min.y);
					Rect scrollCursorRect = rectCenDim(vec2(scrollRect.max.x - scrollBarWidth/2, scrollCursorPos), vec2(scrollBarWidth,scrollCursorHeight));
					dcRect(scrollCursorRect, vec4(0,1,1,1));
				}


				lastDiff = textStart - textPos - rectGetDim(consoleBody).h;


				dcDisable(STATE_SCISSOR);
			}
		}

		if(isActive) {
			if(input->mouseButtonPressed[0]) {
				if(pointInRect(input->mousePosNegative, consoleInput)) {
					mouseSelectMode = true;
				} else {
					markerIndex = cursorIndex;
				}
			}

			if(input->mouseButtonReleased[0]) {
				mouseSelectMode = false;
			}

			if(mouseSelectMode && (strLen(inputBuffer) >= 1)) {
				int inputSize = strLen(inputBuffer);
				int mouseIndex;

				float left = consolePadding + getCharWidth(inputBuffer[0], inputFont) * 0.5;
				if(input->mousePos.x < left) {
					mouseIndex = 0;
				} else {
					float p = consolePadding;
					bool found = false;
					for(int i = 0; i < inputSize - 1; i++) {
						float p1 = p;
						float p2 = p1 + getCharWidth(inputBuffer[i], inputFont);
						float p3 = p2 + getCharWidth(inputBuffer[i+1], inputFont);
						p = p2;

						float x1 = p1 + (p2-p1) * 0.5f;
						float x2 = p2 + (p3-p2) * 0.5f;

						if(valueBetween(input->mousePos.x, x1, x2)) {
							mouseIndex = i+1;
							found = true;
							break;
						}
					}

					if(!found) mouseIndex = inputSize;
				}

				if(input->mouseButtonPressed[0]) {
					markerIndex = mouseIndex;
				}

				cursorIndex = mouseIndex;
			}

			if(!mouseSelectMode) {
				bool left = input->keysPressed[KEYCODE_LEFT];
				bool right = input->keysPressed[KEYCODE_RIGHT];
				bool up = input->keysPressed[KEYCODE_UP];
				bool down = input->keysPressed[KEYCODE_DOWN];

				bool a = input->keysPressed[KEYCODE_A];
				bool x = input->keysPressed[KEYCODE_X];
				bool c = input->keysPressed[KEYCODE_C];
				bool v = input->keysPressed[KEYCODE_V];
				
				bool home = input->keysPressed[KEYCODE_HOME];
				bool end = input->keysPressed[KEYCODE_END];
				bool backspace = input->keysPressed[KEYCODE_BACKSPACE];
				bool del = input->keysPressed[KEYCODE_DEL];
				bool enter = input->keysPressed[KEYCODE_RETURN];

				bool ctrl = input->keysDown[KEYCODE_CTRL];
				bool shift = input->keysDown[KEYCODE_SHIFT];


				int startCursorIndex = cursorIndex;

				if(ctrl && backspace) {
					shift = true;
					left = true;
				}

				if(ctrl && del) {
					shift = true;
					right = true;
				}

				if(left) {
					if(ctrl) {
						if(cursorIndex > 0) {
							while(inputBuffer[cursorIndex-1] == ' ') cursorIndex--;

							if(cursorIndex > 0)
						 		cursorIndex = strFindBackwards(inputBuffer, ' ', cursorIndex-1);
						}
					} else {
						bool isSelected = cursorIndex != markerIndex;
						if(isSelected && !shift) {
							if(cursorIndex < markerIndex) {
								markerIndex = cursorIndex;
							} else {
								cursorIndex = markerIndex;
							}
						} else {
							if(cursorIndex > 0) cursorIndex--;
						}
					}
				}

				if(right) {
					if(ctrl) {
						while(inputBuffer[cursorIndex] == ' ') cursorIndex++;
						if(cursorIndex <= strLen(inputBuffer)) {
							cursorIndex = strFindOrEnd(inputBuffer, ' ', cursorIndex+1);
							if(cursorIndex != strLen(inputBuffer)) cursorIndex--;
						}
					} else {
						bool isSelected = cursorIndex != markerIndex;
						if(isSelected && !shift) {
							if(cursorIndex > markerIndex) {
								markerIndex = cursorIndex;
							} else {
								cursorIndex = markerIndex;
							}
						} else {
							if(cursorIndex < strLen(inputBuffer)) cursorIndex++;
						}
					}
				}

				if(home) {
					cursorIndex = 0;
				}

				if(end) {
					cursorIndex = strLen(inputBuffer);
				}

				if((startCursorIndex != cursorIndex) && !shift) {
					markerIndex = cursorIndex;
				}

				if(ctrl && a) {
					cursorIndex = 0;
					markerIndex = strLen(inputBuffer);
				}

				bool isSelected = cursorIndex != markerIndex;

				if(ctrl && x) {
					c = true;
					del = true;
				}

				if(ctrl && c) {
					float selectionWidth = abs(cursorIndex - markerIndex);
					char* selection = getTStringDebug(selectionWidth);
					strCpy(selection, inputBuffer + (int)min(cursorIndex, markerIndex), selectionWidth);

					setClipboard(selection);
				}

				if(backspace || del || (input->inputCharacterCount > 0) || (ctrl && v)) {
					if(isSelected) {
						int delIndex = min(cursorIndex, markerIndex);
						int delAmount = abs(cursorIndex - markerIndex);
						strRemoveX(inputBuffer, delIndex, delAmount);
						cursorIndex = delIndex;
					}

					markerIndex = cursorIndex;
				}

				if(ctrl && v) {
					char* clipboard = (char*)getClipboard();
					int clipboardSize = strLen(clipboard);
					if(clipboardSize + strLen(inputBuffer) < arrayCount(inputBuffer)) {
						strInsert(inputBuffer, cursorIndex, clipboard);
						cursorIndex += clipboardSize;
						markerIndex = cursorIndex;
					}
				}

				// Add input characters to input buffer.
				if(input->inputCharacterCount > 0) {
					if(input->inputCharacterCount + strLen(inputBuffer) < arrayCount(inputBuffer)) {
						strInsert(inputBuffer, cursorIndex, input->inputCharacters, input->inputCharacterCount);
						cursorIndex += input->inputCharacterCount;
						markerIndex = cursorIndex;
					}
				}

				if(backspace && !isSelected) {
					if(cursorIndex > 0) {
						strRemove(inputBuffer, cursorIndex);
						cursorIndex--;
					}
					markerIndex = cursorIndex;
				}

				if(del && !isSelected) {
					if(cursorIndex+1 <= strLen(inputBuffer)) {
						strRemove(inputBuffer, cursorIndex+1);
					}
					markerIndex = cursorIndex;
				}

				if(enter) {
					if(strLen(inputBuffer) > 0) {
						// Copy over input buffer to console buffer.
						pushToMainBuffer(inputBuffer);
						evaluateInput();

						strClear(inputBuffer);
						cursorIndex = 0;
						markerIndex = 0;
					}
				}

				if(startCursorIndex != cursorIndex) {
					cursorTime = 0;
				}
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

	char* eatSign(char* str) {
		char c = str[0];
		if(c == '-' || c == '+') return str + 1;
		else return str;
	}

	bool charIsDigit(char c) {
		return (c >= (int)'0') && (c <= (int)'9');
	}

	char* eatDigits(char* str) {
		while(charIsDigit(str[0])) str++;
		return str;
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

	bool strIsInteger(char* str) {
		char* s = str;
		s = eatSign(s);
		if(!charIsDigit(s[0])) return false;
		s = eatDigits(s);

		if(s[0] != '\0') return false;

		return true;
	}

	void evaluateInput() {
		char* com = inputBuffer;

		char* argument = getNextArgument(&com);
		if(argument == 0) return;

		if(strCompare(argument, "add")) {
			char* arguments[2];
			arguments[0] = getNextArgument(&com);
			arguments[1] = getNextArgument(&com);
			if(!(arguments + 0) || !(arguments + 1)) {
				pushToMainBuffer(fillString("Function is missing arguments."));
				return;
			}

			if(!strIsInteger(arguments[0]) || !strIsInteger(arguments[1])) {
				pushToMainBuffer(fillString("Arguments are not integers."));
				return;
			}

			int a = strToInt(arguments[0]);
			int b = strToInt(arguments[1]);

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


