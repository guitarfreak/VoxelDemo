
#define CONSOLE_SMALL_PERCENT 0.3f
#define CONSOLE_BIG_PERCENT 0.8f
#define CONSOLE_ARGUMENT_COUNT_MAX 10

enum ArgumentTypes {
	ATYPE_INT = 0,
	ATYPE_FLOAT,
	ATYPE_BOOL,
	ATYPE_STRING,

	ATYPE_SIZE,
};

char* argumentTypeStrings[] = { "Int", "Float", "Bool", "String", };

struct FunctionInfo {
	char* name;
	int typeCount;
	int types[10];
};

FunctionInfo functionInfo[] = { 
	{"add", 		2, ATYPE_INT, ATYPE_INT}, 
	{"addFloat", 	2, ATYPE_FLOAT, ATYPE_FLOAT}, 
	{"cls", 		0}, 
	{"doNothing", 	0},
	{"exit", 		0},
	{"print", 		1, ATYPE_STRING},
	{"setGuiAlpha", 1, ATYPE_FLOAT},
};


struct ConsoleSettings {
	float consoleSpeed;

	float bodyFontHeightPadding;
	float bodyFontHeightResultPadding;

	Font* bodyFont;
	Font* inputFont;

	float inputHeightPadding;
	float fontDrawHeightOffset;
	Vec2 consolePadding;
	float cursorWidth;
	float cursorSpeed;
	float cursorColorMod;

	Vec4 mouseHoverColorMod;
	Vec4 bodyColor;
	Vec4 inputColor;
	Vec4 bodyFontColor;
	Vec4 bodyFontResultColor;
	Vec4 inputFontColor;
	Vec4 cursorColor;
	Vec4 selectionColor;
	Vec4 scrollBarBackgroundColor;
	Vec4 scrollBarColor;

	float scrollBarWidth;
	float scrollCursorMinHeight;
	float scrollCursorMod;
	float scrollWheelAmount;
	float scrollWheelAmountMod;

	float inputScrollMargin;

	char* commandPreText;
};

struct Console {

	ConsoleSettings cs;

	// Temporary.

	float bodyTextWrapWidth;
	Input* input;
	Vec2 currentRes;
	bool visible;

	//

	float pos;
	float targetPos;
	int mode;
	bool isActive;

	float scrollMode;
	float scrollPercent;
	// float lastDiff;
	float mouseAnchor;

	char* mainBuffer[256];
	int mainBufferSize;
	float mainBufferTextHeight;

	char inputBuffer[1024];

	bool commandAvailable;
	char* comName;
	char* comArgs[CONSOLE_ARGUMENT_COUNT_MAX];

	int cursorIndex;
	int markerIndex;
	bool mouseSelectMode;
	float cursorTime;
	float inputOffset;

	int bodySelectionIndex;
	int bodySelectionMarker1, bodySelectionMarker2;
	bool bodySelectionMode;
	bool mousePressedInside;

	bool historyMode;
	char historyBuffer[20][256];
	int historyReadIndex;
	int historyWriteIndex;

	bool autoCompleteMode;
	int autoCompleteIndex;
	int autoCompleteCursor;
	char autoCompleteWord[64];



	void init(float windowHeight) {
		*this = {};

		float smallPos = -windowHeight * CONSOLE_SMALL_PERCENT;
		targetPos = smallPos;

		// mode = 1;
		mode = 0;
		cursorIndex = 0;
		markerIndex = 0;
		scrollPercent = 1;
		bodySelectionIndex = -1;
		inputOffset = 0;

		historyMode = false;
		historyReadIndex = 0;
		historyWriteIndex = 1;
		for(int i = 0; i < arrayCount(historyBuffer); i++) historyBuffer[i][0] = '\0';

		autoCompleteMode = false;
		commandAvailable = false;
	}

	void initSettings() {
		cs.consoleSpeed = 10;

		cs.bodyFontHeightPadding = 1.0f;
		cs.bodyFontHeightResultPadding = 1.2f;

		cs.bodyFont = getFont(FONT_SOURCESANS_PRO, 22);
		cs.inputFont = getFont(FONT_CONSOLAS, 16);

		cs.inputHeightPadding = 1.5;
		cs.fontDrawHeightOffset = 0.2f;
		cs.consolePadding = vec2(10,5);
		cs.cursorWidth = 2;
		cs.cursorSpeed = 3;
		cs.cursorColorMod = 0.2f;

		float g = 0.1f;
		cs.mouseHoverColorMod       = vec4(g,g,g,0);
		cs.bodyColor                = vec4(0.3f,0,0.6f,1.0f);
		cs.inputColor               = vec4(0.37f,0,0.6f,1.0f);
		cs.bodyFontColor            = vec4(1,1,1,1);
		cs.bodyFontResultColor      = vec4(0.9f,0,0.1f,1);
		cs.inputFontColor           = vec4(1,1,1,1);
		cs.cursorColor              = vec4(0.2f,0.8f,0.1f,0.9f);
		cs.selectionColor           = vec4(0.1f,0.4f,0.4f,0.9f);
		cs.scrollBarBackgroundColor = vec4(0.2,0,0,0) + cs.bodyColor;
		cs.scrollBarColor           = vec4(0,0.5f,0.7f,1);

		cs.scrollBarWidth = 20;
		cs.scrollCursorMinHeight = 60;
		cs.scrollCursorMod = 0.3f;
		cs.scrollWheelAmount = cs.bodyFont->height;
		cs.scrollWheelAmountMod = 4;

		cs.inputScrollMargin = 10;

		cs.commandPreText = "> ";
	}

	void update(Input* input, Vec2 currentRes, float dt) {

		initSettings();

		this->input = input;
		this->currentRes = currentRes;

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
			float consoleMovement = cs.consoleSpeed * dt;
			pos = lerp(consoleMovement, pos, targetPos);

			// Clamp if we overstepped the target position.
			float newDistance = pos - targetPos;
			if(abs(newDistance) <= 1.0f) pos = targetPos;
		}



		Vec2 res = currentRes;
		float consoleTotalHeight = abs(pos);
		clampMin(&consoleTotalHeight, abs(smallPos));

		float inputHeight = cs.inputFont->height * cs.inputHeightPadding;
		float bodyTextHeight = cs.bodyFont->height * cs.bodyFontHeightPadding;
		float consoleBodyHeight = consoleTotalHeight - inputHeight;
		Rect consoleBody = rectMinDim(vec2(0, pos + inputHeight), vec2(res.w, consoleBodyHeight));
		Rect consoleInput = rectMinDim(vec2(0, pos), vec2(res.w, inputHeight));

		float preTextSize = getTextDim(cs.commandPreText, cs.bodyFont).w;
		bodyTextWrapWidth = consoleBody.max.x - cs.scrollBarWidth - cs.consolePadding.x*2 - preTextSize;



		if(!isActive && pointInRect(input->mousePosNegative, consoleInput)) {
			cs.inputColor += cs.mouseHoverColorMod;

			if(input->mouseButtonPressed[0]) {
				isActive = true;
				strClear(inputBuffer);
				cursorIndex = 0;
				markerIndex = 0;
				
				historyMode = false;
				historyReadIndex = mod(historyWriteIndex-1, arrayCount(historyBuffer));
			}
		}

		if(input->keysPressed[KEYCODE_ESCAPE]) {
			isActive = false;
		}

		visible = true;
		if(pos == closedPos) {
			isActive = false;
			visible = false;
		}

		if(visible) {
			dcRect(consoleInput, cs.inputColor);
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
				Vec2 inputStartPos = (consoleInput.min + vec2(cs.consolePadding.x, 0)) + vec2(-inputOffset, inputHeight/2);

				int mouseIndex = textMouseToIndex(inputBuffer, cs.inputFont, inputStartPos, input->mousePosNegative, vec2i(-1,0));

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
				bool tab = input->keysPressed[KEYCODE_TAB];

				bool ctrl = input->keysDown[KEYCODE_CTRL];
				bool shift = input->keysDown[KEYCODE_SHIFT];

				// History.

				if(historyMode) {
					if(up || down) {
						int newPos;
						if(up) newPos = mod(historyReadIndex-1, arrayCount(historyBuffer));
						else if(down) newPos = mod(historyReadIndex+1, arrayCount(historyBuffer));

						bool skip = false;
						if(up && (newPos == mod(historyWriteIndex-1, arrayCount(historyBuffer)))) skip = true;
						if(down && newPos == historyWriteIndex) skip = true;

						if(!skip && strLen(historyBuffer[newPos]) != 0) {
							historyReadIndex = newPos;
							strCpy(inputBuffer, historyBuffer[historyReadIndex]);

							cursorIndex = strLen(inputBuffer);
							markerIndex = cursorIndex;
						}
					}
				}

				if(up && !historyMode) {
					historyMode = true;
					historyReadIndex = mod(historyWriteIndex-1, arrayCount(historyBuffer));

					if(strLen(historyBuffer[historyReadIndex]) != 0) {
						strCpy(inputBuffer, historyBuffer[historyReadIndex]);
						
						cursorIndex = strLen(inputBuffer);
						markerIndex = cursorIndex;
					}
				}

				// Auto complete.

				if(tab && !autoCompleteMode) {
					// Search backwards for a word.

					autoCompleteIndex = 0;
					int wordIndex = strFindBackwards(inputBuffer, ' ', cursorIndex);
					int wordLength = cursorIndex - wordIndex;

					if(wordLength < arrayCount(autoCompleteWord)) {
						autoCompleteMode = true;
						
						strCpy(autoCompleteWord, inputBuffer + wordIndex, wordLength);

						autoCompleteCursor = wordIndex;
					}
				}
				
				if(autoCompleteMode) {
					if(tab) {
						int nameCount = arrayCount(functionInfo);
						bool found = false;
						for(int i = 0; i < nameCount; i++) {
							int index = mod(autoCompleteIndex+i, nameCount);
							bool result;
							if(strLen(autoCompleteWord) == 0) result = true;
							else result = strStartsWith(functionInfo[index].name, autoCompleteWord);

							if(result) {
								autoCompleteIndex = index;
								found = true;

								int amount = cursorIndex - autoCompleteCursor;
								strRemoveX(inputBuffer, autoCompleteCursor, amount);

								char* word = functionInfo[autoCompleteIndex].name;
								int wordLength = strLen(word);

								strInsert(inputBuffer, autoCompleteCursor, word);

								cursorIndex = autoCompleteCursor + wordLength;
								markerIndex = cursorIndex;

								break;
							}
						}

						if(!found) autoCompleteMode = false;

						autoCompleteIndex = mod(autoCompleteIndex+1, nameCount);

					} else if(input->anyKey) {
						autoCompleteMode = false;
					}
				}

				// Main navigation and things.

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

				if((ctrl && x) && isSelected) {
					c = true;
					del = true;
				}

				if((ctrl && c) && isSelected) {
					char* selection = textSelectionToString(inputBuffer, cursorIndex, markerIndex);
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
						// Push to history buffer.

						int stringLength = max(strLen(inputBuffer), arrayCount(historyBuffer));
						strCpy(historyBuffer[historyWriteIndex], inputBuffer, stringLength);
						historyReadIndex = historyWriteIndex;
						historyWriteIndex = mod(historyWriteIndex+1, arrayCount(historyBuffer));
						historyMode = false;

						// Copy over input buffer to console buffer.

						pushToMainBuffer(inputBuffer);
						
						evaluateInput();

						strClear(inputBuffer);
						cursorIndex = 0;
						markerIndex = 0;

						scrollPercent = 1;
					}
				}

				if(startCursorIndex != cursorIndex) {
					cursorTime = 0;
				}
			}

			// Scroll input vertically.

			{
				Rect inputRect = rectExpand(consoleInput, vec2(-cs.consolePadding.x, 0));
				Vec2 inputStartPos = inputRect.min + vec2(-inputOffset, inputHeight/2);

				Vec2 cursorPos = textIndexToPos(inputBuffer, cs.inputFont, inputStartPos, cursorIndex, vec2i(-1,0));

				float cursorDiffLeft = (inputRect.min.x + cs.inputScrollMargin) - cursorPos.x;
				if(cursorDiffLeft > 0) {
					inputOffset = clampMin(inputOffset - cursorDiffLeft, consoleInput.min.x);
				}

				float cursorDiffRight = cursorPos.x - (inputRect.max.x - cs.inputScrollMargin);
				if(cursorDiffRight > 0) {
					inputOffset = inputOffset + cursorDiffRight;
				}
			}

			// 
			// Drawing.
			//
			
			Rect inputRect = rectExpand(consoleInput, vec2(-cs.consolePadding.x, 0));
			Vec2 inputStartPos = inputRect.min + vec2(-inputOffset, inputHeight/2);

			// Selection.

			drawTextSelection(inputBuffer, cs.inputFont, inputStartPos, cursorIndex, markerIndex, cs.selectionColor, vec2i(-1,0));

			// Text.

			dcEnable(STATE_SCISSOR);
			dcScissor(scissorRectScreenSpace(inputRect, res.h));

			dcText(inputBuffer, cs.inputFont, inputStartPos, cs.inputFontColor, vec2i(-1,0));

			dcDisable(STATE_SCISSOR);

			// Cursor.

			cursorTime += dt*cs.cursorSpeed;
			Vec4 cmod = vec4(0,cos(cursorTime)*cs.cursorColorMod - cs.cursorColorMod,0,0);

			Vec2 cursorPos = textIndexToPos(inputBuffer, cs.inputFont, inputStartPos, cursorIndex, vec2i(-1,0));

			bool cursorAtEnd = cursorIndex == strLen(inputBuffer);
			float cWidth = cursorAtEnd ? getTextDim("M", cs.inputFont).w : cs.cursorWidth;
			Rect cursorRect = rectCenDim(cursorPos, vec2(cWidth, cs.inputFont->height));

			if(cursorAtEnd) cursorRect = rectAddOffset(cursorRect, vec2(rectGetDim(cursorRect).w/2, 0));
			dcRect(cursorRect, cs.cursorColor + cmod);

		}

	}

	void updateBody() {

		float smallPos = -currentRes.y * CONSOLE_SMALL_PERCENT;

		Vec2 res = currentRes;
		float consoleTotalHeight = abs(pos);
		clampMin(&consoleTotalHeight, abs(smallPos));

		float inputHeight = cs.inputFont->height * cs.inputHeightPadding;
		float bodyTextHeight = cs.bodyFont->height * cs.bodyFontHeightPadding;
		float consoleBodyHeight = consoleTotalHeight - inputHeight;
		Rect consoleBody = rectMinDim(vec2(0, pos + inputHeight), vec2(res.w, consoleBodyHeight));
		Rect consoleInput = rectMinDim(vec2(0, pos), vec2(res.w, inputHeight));



		if(visible) {
			dcRect(consoleBody, cs.bodyColor);

			float scrollOffset = 0;

			float consoleTextHeight = rectGetDim(consoleBody).h - cs.consolePadding.h*2;
			float heightDiff = mainBufferTextHeight - consoleTextHeight;

			if(heightDiff >= 0) {
				scrollOffset = scrollPercent*heightDiff; 
				
				// Scrollbar background.

				Rect scrollRect = consoleBody;
				scrollRect.min.x = scrollRect.max.x - cs.scrollBarWidth;

				// Scrollbar cursor.

				float consoleHeight = rectGetDim(consoleBody).h;
				float scrollCursorHeight = (consoleHeight / (consoleHeight + heightDiff)) * consoleHeight;
				clampMin(&scrollCursorHeight, cs.scrollCursorMinHeight);


				if(input->mouseWheel && pointInRect(input->mousePosNegative, consoleBody)) {
					if(input->keysDown[KEYCODE_CTRL]) cs.scrollWheelAmount *= cs.scrollWheelAmountMod;

					scrollPercent += -input->mouseWheel * (cs.scrollWheelAmount / heightDiff);
					clamp(&scrollPercent, 0, 1);
				}

				// Enable scrollbar keyboard navigation if we're not typing anything into the input console.
				if(strLen(inputBuffer) == 0) {
					if(input->keysPressed[KEYCODE_HOME]) scrollPercent = 0;
					if(input->keysPressed[KEYCODE_END]) scrollPercent = 1;
					if(input->keysPressed[KEYCODE_PAGEUP] || input->keysPressed[KEYCODE_PAGEDOWN]) {
						float dir;
						if(input->keysPressed[KEYCODE_PAGEUP]) dir = -1;
						else dir = 1;

						scrollPercent += dir * (consoleHeight*0.8f / heightDiff);
					}

					clamp(&scrollPercent, 0, 1);
				}

				// @CodeDuplication.

				float scrollCursorPos = lerp(scrollPercent, scrollRect.max.y - scrollCursorHeight/2, scrollRect.min.y + scrollCursorHeight/2);
				Rect scrollCursorRect = rectCenDim(vec2(scrollRect.max.x - cs.scrollBarWidth/2, scrollCursorPos), vec2(cs.scrollBarWidth,scrollCursorHeight));


				if(input->mouseButtonReleased[0]) {
					scrollMode = false;
				}

				Vec4 scrollBarColorFinal = cs.scrollBarColor;

				if(pointInRect(input->mousePosNegative, scrollCursorRect)) {
					if(input->mouseButtonPressed[0]) {
						scrollMode = true;
						mouseAnchor = scrollCursorPos - input->mousePosNegative.y;
					}

					if(!scrollMode) {
						scrollBarColorFinal += cs.mouseHoverColorMod;
					}
				}

				if(scrollMode) {
					scrollBarColorFinal += cs.mouseHoverColorMod;

					scrollPercent = mapRangeClamp(input->mousePosNegative.y + mouseAnchor, scrollRect.min.y + scrollCursorHeight/2, scrollRect.max.y - scrollCursorHeight/2, 0, 1);
					scrollPercent = 1-scrollPercent;
				}

				// @CodeDuplication.
				// Recalculate scroll rect to reduce lag.

				scrollCursorPos = lerp(scrollPercent, scrollRect.max.y - scrollCursorHeight/2, scrollRect.min.y + scrollCursorHeight/2);
				scrollCursorRect = rectCenDim(vec2(scrollRect.max.x - cs.scrollBarWidth/2, scrollCursorPos), vec2(cs.scrollBarWidth,scrollCursorHeight));

				// Draw scrollbar.

				dcRect(scrollRect, cs.scrollBarBackgroundColor);
				dcRect(scrollCursorRect, scrollBarColorFinal);
			}

			// Main window.
			{
				dcEnable(STATE_SCISSOR);

				Rect scrollRect = consoleBody;
				scrollRect.min.x = scrollRect.max.x - cs.scrollBarWidth;

				Rect consoleTextRect = consoleBody;
				consoleTextRect = rectExpand(consoleTextRect, -cs.consolePadding*2);
				if(heightDiff >= 0) consoleTextRect.max.x -= cs.scrollBarWidth;

				dcScissor(scissorRectScreenSpace(consoleTextRect, res.h));

				float preSize = getTextDim(cs.commandPreText, cs.bodyFont).w;

				Vec2 textPos = vec2(cs.consolePadding.x + preSize, pos + consoleTotalHeight + scrollOffset - cs.consolePadding.y);
				float textStart = textPos.y;
				float textStartX = textPos.x;
				float wrappingWidth = rectGetDim(consoleTextRect).w - textStartX;

				bool mousePressed = input->mouseButtonPressed[0];
				bool mouseInsideConsole = pointInRect(input->mousePosNegative, consoleTextRect);
				bool mouseInsideScrollbar = pointInRect(input->mousePosNegative, scrollRect);
				if(mousePressed) {
					if(mouseInsideConsole) bodySelectionMode = true;
					else if(mouseInsideScrollbar) bodySelectionMode = false;
					else {
						bodySelectionMode = false;
						bodySelectionIndex = -1;
					}
				}

				if(input->mouseButtonReleased[0]) {
					bodySelectionMode = false;

					if(bodySelectionMarker1 == bodySelectionMarker2) {
						bodySelectionIndex = -1;
					}
				}

				for(int i = 0; i < mainBufferSize; i++) {
					if(i%2 == 0) {
						dcText(cs.commandPreText, cs.bodyFont, textPos - vec2(preSize,0), cs.bodyFontColor, vec2i(-1,1));
					} else {
						if(strIsEmpty(mainBuffer[i])) continue;
					}

					// Cull texts that are above or below the console body.
					int textHeight = getTextDim(mainBuffer[i], cs.bodyFont, textPos, wrappingWidth).h;
					bool textOutside = textPos.y - textHeight > consoleTextRect.max.y || textPos.y < consoleTextRect.min.y;

					if(!textOutside) {
						if(mousePressed && mouseInsideConsole) {
							if(valueBetween(input->mousePosNegative.y, textPos.y - textHeight, textPos.y) && 
							   (input->mousePosNegative.x < consoleTextRect.max.x)) {
								bodySelectionIndex = i;
								bodySelectionMarker1 = textMouseToIndex(mainBuffer[bodySelectionIndex], cs.bodyFont, textPos, input->mousePosNegative, vec2i(-1,1), wrappingWidth);
								bodySelectionMarker2 = bodySelectionMarker1;
								mousePressed = false;
							} 
						}

						if(i == bodySelectionIndex) {
							if(bodySelectionMode) {
								bodySelectionMarker2 = textMouseToIndex(mainBuffer[bodySelectionIndex], cs.bodyFont, textPos, input->mousePosNegative, vec2i(-1,1), wrappingWidth);
							}

							drawTextSelection(mainBuffer[i], cs.bodyFont, textPos, bodySelectionMarker1, bodySelectionMarker2, cs.selectionColor, vec2i(-1,1), wrappingWidth);
						}

						Vec4 color = i%2 == 0 ? cs.bodyFontColor : cs.bodyFontResultColor;
						dcText(mainBuffer[i], cs.bodyFont, textPos, color, vec2i(-1,1), wrappingWidth);
					}

					textPos.y -= textHeight;
				}

				if(cursorIndex != markerIndex) {
					bodySelectionIndex = -1;
				}

				if(bodySelectionIndex != -1) {
					if(input->keysDown[KEYCODE_CTRL] && input->keysPressed[KEYCODE_C]) {
						char* selection = textSelectionToString(mainBuffer[bodySelectionIndex], bodySelectionMarker1, bodySelectionMarker2);
						setClipboard(selection);
					}
				}

				dcDisable(STATE_SCISSOR);
			}
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

	bool charIsLetter(char c) {
		return ((c >= (int)'a') && (c <= (int)'z') || 
		        (c >= (int)'A') && (c <= (int)'Z'));
	}

	bool charIsDigitOrLetter(char c) {
		return charIsDigit(c) || charIsLetter(c);
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

		float height = getTextHeight(newString, cs.bodyFont, vec2(0,0), bodyTextWrapWidth);
		mainBufferTextHeight += height;
	}


	void clearMainBuffer() {
		mainBufferSize = 0;
		mainBufferTextHeight = 0;
	}

	bool strIsType(char* s, int type) {
		switch(type) {
			case ATYPE_INT: {
				s = eatSign(s);
				if(!charIsDigit(s[0])) return false;
				s = eatDigits(s);

				if(s[0] != '\0') return false;

				return true;
			} break;

			case ATYPE_FLOAT: {
				s = eatSign(s);
				if(!charIsDigit(s[0])) return false;
				s = eatDigits(s);
				if(!(s[0] == '.')) return false;
				s++;
				if(!charIsDigit(s[0])) return false;
				s = eatDigits(s);

				if(s[0] != '\0') return false;

				return true;
			} break;

			case ATYPE_BOOL: {
				if(strCompare(s, "true") || strCompare(s, "True") ||
				   strCompare(s, "false") || strCompare(s, "False") ||
				   s[0] == '0' || s[0] == '1') {
					return true;
				} else {
					return false;
				}
			} break;

			case ATYPE_STRING: {
				if(!charIsDigit(s[0])) {
					
					int sl = strLen(s);
					for(int i = 0; i < sl; i++) {
						char c = s[i];
						if(!charIsDigitOrLetter(c)) {
							return false;
							break;
						}
					}

					return true;
				} else {
					return false;
				}
			} break;

			default: return false;
		}
	}

	bool checkTypes(char* str, int* types, int typeCount) {

		int argCount = 0;
		while(true) {
			char* argument = getNextArgument(&str);
			if(argument == 0) break;

			comArgs[argCount] = argument;
			argCount++;

			if(argCount >= CONSOLE_ARGUMENT_COUNT_MAX) break;
		}

		if(argCount != typeCount) {
			char* plural = argCount == 1 ? "" : "s";
			pushToMainBuffer(fillString("Error: Function needs %i argument%s but received %i.", typeCount, plural, argCount));
			return false;
		}

		for(int i = 0; i < typeCount; i++) {
			char* str = comArgs[i];
			int type = types[i];

			bool correctType = strIsType(str, type);
			if(!correctType) {
				char* argString = argumentTypeStrings[type];
				pushToMainBuffer(fillString("Error: Argument %i is not of type \'%s\'.", i+1, argString));
				return false;
			}
		}
		
		return true;
	}

	void evaluateInput() {
		char* com = inputBuffer;

		char* cName = getNextArgument(&com);
		if(cName == 0) return;

		FunctionInfo* fInfo = 0;
		for(int i = 0; i < arrayCount(functionInfo); i++) {
			if(strCompare(functionInfo[i].name, cName)) {
				fInfo = functionInfo + i;
			}
		}

		if(!fInfo) {
			pushToMainBuffer(fillString("Error: Unknown command \"%s\".", cName));
			return;
		}

		bool correctTypes = checkTypes(com, fInfo->types, fInfo->typeCount);
		if(!correctTypes) return;

		comName = cName;
		commandAvailable = true;
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
	float guiAlpha;

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




struct TextEditVars {
	int cursorIndex;
	int markerIndex;
	Vec2 scrollOffset;

	bool cursorChanged;
};

struct TextEditSettings {
	bool wrapping;
	bool singleLine;

	float screenHeight;
	float cursorWidth;

	Vec4 colorBackground;
	Vec4 colorSelection;
	Vec4 colorText;
	Vec4 colorCursor;
};

void textEditBox(char* text, int textMaxSize, Font* font, Rect textRect, Input* input, Vec2i align, TextEditSettings tes, TextEditVars* tev) {

	if(tes.singleLine) tes.wrapping = false;

	Vec2 startPos = rectGetUL(textRect) + tev->scrollOffset;
	int wrapWidth = tes.wrapping ? rectGetDim(textRect).w : 0;

	int cursorIndex = tev->cursorIndex;
	int markerIndex = tev->markerIndex;

	int mouseIndex = textMouseToIndex(text, font, startPos, input->mousePosNegative, align, wrapWidth);

	if(input->mouseButtonPressed[0]) {
		if(pointInRect(input->mousePosNegative, textRect)) {
			markerIndex = mouseIndex;
		}
	}

	if(input->mouseButtonDown[0]) {
		cursorIndex = mouseIndex;
	}



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
	bool tab = input->keysPressed[KEYCODE_TAB];

	bool ctrl = input->keysDown[KEYCODE_CTRL];
	bool shift = input->keysDown[KEYCODE_SHIFT];

	// Main navigation and things.

	int startCursorIndex = cursorIndex;

	if(ctrl && backspace) {
		shift = true;
		left = true;
	}

	if(ctrl && del) {
		shift = true;
		right = true;
	}

	if(!tes.singleLine) {
		if(up || down) {
			float cursorYOffset;
			if(up) cursorYOffset = font->height;
			else if(down) cursorYOffset = -font->height;

			Vec2 cPos = textIndexToPos(text, font, startPos, cursorIndex, align, wrapWidth);
			cPos.y += cursorYOffset;
			int newCursorIndex = textMouseToIndex(text, font, startPos, cPos, align, wrapWidth);
			cursorIndex = newCursorIndex;
		}
	}


	if(left) {
		if(ctrl) {
			if(cursorIndex > 0) {
				while(text[cursorIndex-1] == ' ') cursorIndex--;

				if(cursorIndex > 0)
			 		cursorIndex = strFindBackwards(text, ' ', cursorIndex-1);
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
			while(text[cursorIndex] == ' ') cursorIndex++;
			if(cursorIndex <= strLen(text)) {
				cursorIndex = strFindOrEnd(text, ' ', cursorIndex+1);
				if(cursorIndex != strLen(text)) cursorIndex--;
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
				if(cursorIndex < strLen(text)) cursorIndex++;
			}
		}
	}

	if(tes.singleLine) {
		if(home) {
			cursorIndex = 0;
		}

		if(end) {
			cursorIndex = strLen(text);
		}
	}

	if((startCursorIndex != cursorIndex) && !shift) {
		markerIndex = cursorIndex;
	}

	if(ctrl && a) {
		cursorIndex = 0;
		markerIndex = strLen(text);
	}

	bool isSelected = cursorIndex != markerIndex;

	if((ctrl && x) && isSelected) {
		c = true;
		del = true;
	}

	if((ctrl && c) && isSelected) {
		char* selection = textSelectionToString(text, cursorIndex, markerIndex);
		setClipboard(selection);
	}

	if(enter) {
		if(tes.singleLine) {
			strClear(text);
			cursorIndex = 0;
			markerIndex = 0;
		} else {
			input->inputCharacters[input->inputCharacterCount++] = '\n';
		}
	}

	if(backspace || del || (input->inputCharacterCount > 0) || (ctrl && v)) {
		if(isSelected) {
			int delIndex = min(cursorIndex, markerIndex);
			int delAmount = abs(cursorIndex - markerIndex);
			strRemoveX(text, delIndex, delAmount);
			cursorIndex = delIndex;
		}

		markerIndex = cursorIndex;
	}

	if(ctrl && v) {
		char* clipboard = (char*)getClipboard();
		int clipboardSize = strLen(clipboard);
		if(clipboardSize + strLen(text) < textMaxSize) {
			strInsert(text, cursorIndex, clipboard);
			cursorIndex += clipboardSize;
			markerIndex = cursorIndex;
		}
	}

	// Add input characters to input buffer.
	if(input->inputCharacterCount > 0) {
		if(input->inputCharacterCount + strLen(text) < textMaxSize) {
			strInsert(text, cursorIndex, input->inputCharacters, input->inputCharacterCount);
			cursorIndex += input->inputCharacterCount;
			markerIndex = cursorIndex;
		}
	}

	if(backspace && !isSelected) {
		if(cursorIndex > 0) {
			strRemove(text, cursorIndex);
			cursorIndex--;
		}
		markerIndex = cursorIndex;
	}

	if(del && !isSelected) {
		if(cursorIndex+1 <= strLen(text)) {
			strRemove(text, cursorIndex+1);
		}
		markerIndex = cursorIndex;
	}

	// Scrolling.
	{
		Vec2 cursorPos = textIndexToPos(text, font, startPos, cursorIndex, align, wrapWidth);

		float ch = font->height;
		if(		cursorPos.x 	   < textRect.min.x) tev->scrollOffset.x += textRect.min.x - (cursorPos.x);
		else if(cursorPos.x 	   > textRect.max.x) tev->scrollOffset.x += textRect.max.x - (cursorPos.x);
		if(		cursorPos.y - ch/2 < textRect.min.y) tev->scrollOffset.y += textRect.min.y - (cursorPos.y - ch/2);
		else if(cursorPos.y + ch/2 > textRect.max.y) tev->scrollOffset.y += textRect.max.y - (cursorPos.y + ch/2);

		clampMax(&tev->scrollOffset.x, 0);
		clampMin(&tev->scrollOffset.y, 0);
	}



	// Draw.

	startPos = rectGetUL(textRect) + tev->scrollOffset;

	// Background.

	float g = 0.1f;
	dcRect(textRect, tes.colorBackground);

	// Selection.

	dcEnable(STATE_SCISSOR);
	dcScissor(scissorRectScreenSpace(textRect, tes.screenHeight));

	drawTextSelection(text, font, startPos, cursorIndex, markerIndex, tes.colorSelection, align, wrapWidth);

	// Text.

	dcText(text, font, startPos, tes.colorText, align, wrapWidth);

	dcDisable(STATE_SCISSOR);

	// Cursor.

	// tev->cursorTime += tes.dt * tes.cursorSpeed;
	// Vec4 cmod = vec4(0,cos(tev->cursorTime)*tes.cursorColorMod - tes.cursorColorMod,0,0);

	Vec2 cPos = textIndexToPos(text, font, startPos, cursorIndex, align, wrapWidth);
	Rect cRect = rectCenDim(cPos, vec2(tes.cursorWidth, font->height));
	dcRect(cRect, tes.colorCursor);

	tev->cursorChanged = (tev->cursorIndex != cursorIndex || tev->markerIndex != markerIndex);

	tev->cursorIndex = cursorIndex;
	tev->markerIndex = markerIndex;
}



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


