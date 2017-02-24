

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



struct Console {
	float pos;
	float targetPos;
	int mode;
	bool isActive;

	float scrollMode;
	float scrollPercent;
	float lastDiff;
	float mouseAnchor;

	char* mainBuffer[256];
	int mainBufferSize;

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

	    - Clean this up once it's solid.
	*/

	void init(float windowHeight) {
		*this = {};

		float smallPos = -windowHeight * CONSOLE_SMALL_PERCENT;
		targetPos = smallPos;

		mode = 1;
		// mode = 0;
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
		Vec2 consolePadding = vec2(10,5);
		float cursorWidth = 2;
		float cursorSpeed = 3;
		float cursorColorMod = 0.2f;

		float g = 0.1f;
		Vec4 mouseHoverColorMod 	= vec4(g,g,g,0);
		Vec4 bodyColor 				= vec4(0.3f,0,0.6f,1.0f);
		Vec4 inputColor 			= vec4(0.37f,0,0.6f,1.0f);
		Vec4 bodyFontColor 			= vec4(1,1,1,1);
		Vec4 bodyFontResultColor 	= vec4(0.9f,0,0.1f,1);
		Vec4 inputFontColor 		= vec4(1,1,1,1);
		Vec4 cursorColor 			= vec4(0.2f,0.8f,0.1f,0.9f);
		Vec4 selectionColor 		= vec4(0.1f,0.4f,0.4f,0.9f);
		Vec4 scrollBarBackgroundColor = bodyColor + vec4(0.2,0,0,0);
		Vec4 scrollBarColor = vec4(0,0.5f,0.7f,1);

		float scrollBarWidth = 20;
		float scrollCursorMinHeight = 60;
		float scrollCursorMod = 0.3f;
		float scrollWheelAmount = bodyFontSize;
		float scrollWheelAmountMod = 4;

		float inputScrollMargin = 10;


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


		if(!isActive && pointInRect(input->mousePosNegative, consoleInput)) {
			inputColor += mouseHoverColorMod;

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

		bool visible = true;
		if(pos == closedPos) {
			isActive = false;
			visible = false;
		}

		if(visible) {
			dcRect(consoleBody, bodyColor);
			dcRect(consoleInput, inputColor);


			float scrollOffset = 0;
			if(lastDiff >= 0) {
				scrollOffset = scrollPercent*lastDiff; 
				
				// Scrollbar background.

				Rect scrollRect = consoleBody;
				scrollRect.min.x = scrollRect.max.x - scrollBarWidth;

				// Scrollbar cursor.

				float consoleHeight = rectGetDim(consoleBody).h;
				float scrollCursorHeight = (consoleHeight / (consoleHeight + lastDiff)) * consoleHeight;
				clampMin(&scrollCursorHeight, scrollCursorMinHeight);


				if(input->mouseWheel && pointInRect(input->mousePosNegative, consoleBody)) {
					if(input->keysDown[KEYCODE_CTRL]) scrollWheelAmount *= scrollWheelAmountMod;

					scrollPercent += -input->mouseWheel * (scrollWheelAmount / lastDiff);
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

						scrollPercent += dir * (consoleHeight*0.8f / lastDiff);
					}

					clamp(&scrollPercent, 0, 1);
				}

				// @CodeDuplication.

				float scrollCursorPos = lerp(scrollPercent, scrollRect.max.y - scrollCursorHeight/2, scrollRect.min.y + scrollCursorHeight/2);
				Rect scrollCursorRect = rectCenDim(vec2(scrollRect.max.x - scrollBarWidth/2, scrollCursorPos), vec2(scrollBarWidth,scrollCursorHeight));


				if(input->mouseButtonReleased[0]) {
					scrollMode = false;
				}

				if(pointInRect(input->mousePosNegative, scrollCursorRect)) {
					if(input->mouseButtonPressed[0]) {
						scrollMode = true;
						mouseAnchor = scrollCursorPos - input->mousePosNegative.y;
					}

					if(!scrollMode) {
						scrollBarColor += mouseHoverColorMod;
					}
				}

				if(scrollMode) {
					scrollBarColor += mouseHoverColorMod;

					scrollPercent = mapRangeClamp(input->mousePosNegative.y + mouseAnchor, scrollRect.min.y + scrollCursorHeight/2, scrollRect.max.y - scrollCursorHeight/2, 0, 1);
					scrollPercent = 1-scrollPercent;
				}

				// @CodeDuplication.
				// Recalculate scroll rect to reduce lag.

				scrollCursorPos = lerp(scrollPercent, scrollRect.max.y - scrollCursorHeight/2, scrollRect.min.y + scrollCursorHeight/2);
				scrollCursorRect = rectCenDim(vec2(scrollRect.max.x - scrollBarWidth/2, scrollCursorPos), vec2(scrollBarWidth,scrollCursorHeight));

				// Draw scrollbar.

				dcRect(scrollRect, scrollBarBackgroundColor);
				dcRect(scrollCursorRect, scrollBarColor);
			}

			// Main window.
			{
				dcEnable(STATE_SCISSOR);

				Rect scrollRect = consoleBody;
				scrollRect.min.x = scrollRect.max.x - scrollBarWidth;

				Rect consoleTextRect = consoleBody;
				consoleTextRect = rectExpand(consoleTextRect, -consolePadding*2);
				if(lastDiff >= 0) consoleTextRect.max.x -= scrollBarWidth;

				dcScissor(scissorRectScreenSpace(consoleTextRect, res.h));

				char* pre = "> ";
				float preSize = getTextDim(pre, bodyFont).w;

				Vec2 textPos = vec2(consolePadding.x + preSize, pos + consoleTotalHeight + scrollOffset - consolePadding.y);
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
						dcText(pre, bodyFont, textPos - vec2(preSize,0), bodyFontColor, vec2i(-1,1));
					} else {
						if(strIsEmpty(mainBuffer[i])) continue;
					}

					// Cull texts that are above or below the console body.
					int textHeight = getTextDim(mainBuffer[i], bodyFont, textPos, wrappingWidth).h;
					bool textOutside = textPos.y - textHeight > consoleTextRect.max.y || textPos.y < consoleTextRect.min.y;

					if(!textOutside) {
						if(mousePressed && mouseInsideConsole) {
							if(valueBetween(input->mousePosNegative.y, textPos.y - textHeight, textPos.y) && 
							   (input->mousePosNegative.x < consoleTextRect.max.x)) {
								bodySelectionIndex = i;
								bodySelectionMarker1 = textMouseToIndex(mainBuffer[bodySelectionIndex], bodyFont, textPos, input->mousePosNegative, vec2i(-1,1), wrappingWidth);
								bodySelectionMarker2 = bodySelectionMarker1;
								mousePressed = false;
							} 
						}

						if(i == bodySelectionIndex) {
							if(bodySelectionMode) {
								bodySelectionMarker2 = textMouseToIndex(mainBuffer[bodySelectionIndex], bodyFont, textPos, input->mousePosNegative, vec2i(-1,1), wrappingWidth);
							}

							drawTextSelection(mainBuffer[i], bodyFont, textPos, bodySelectionMarker1, bodySelectionMarker2, selectionColor, vec2i(-1,1), wrappingWidth);
						}

						Vec4 color = i%2 == 0 ? bodyFontColor : bodyFontResultColor;
						dcText(mainBuffer[i], bodyFont, textPos, color, vec2i(-1,1), wrappingWidth);
					}

					textPos.y -= textHeight;
				}

				lastDiff = textStart - textPos.y - rectGetDim(consoleTextRect).h;

				if(cursorIndex != markerIndex) {
					bodySelectionIndex = -1;
				}

				if(bodySelectionIndex != 0) {
					if(input->keysDown[KEYCODE_CTRL] && input->keysPressed[KEYCODE_C]) {
						char* selection = textSelectionToString(mainBuffer[bodySelectionIndex], bodySelectionMarker1, bodySelectionMarker2);
						setClipboard(selection);
					}
				}

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
				Vec2 inputStartPos = (consoleInput.min + vec2(consolePadding.x, 0)) + vec2(-inputOffset, inputHeight/2);

				int mouseIndex = textMouseToIndex(inputBuffer, inputFont, inputStartPos, input->mousePosNegative, vec2i(-1,0));

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

				if(ctrl && x) {
					c = true;
					del = true;
				}

				if(ctrl && c) {
					if(cursorIndex != markerIndex) {
						char* selection = textSelectionToString(inputBuffer, cursorIndex, markerIndex);
						setClipboard(selection);
					}
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
				Rect inputRect = rectExpand(consoleInput, vec2(-consolePadding.x, 0));
				Vec2 inputStartPos = inputRect.min + vec2(-inputOffset, inputHeight/2);

				Vec2 cursorPos = textIndexToPos(inputBuffer, inputFont, inputStartPos, cursorIndex, vec2i(-1,0));

				float cursorDiffLeft = (inputRect.min.x + inputScrollMargin) - cursorPos.x;
				if(cursorDiffLeft > 0) {
					inputOffset = clampMin(inputOffset - cursorDiffLeft, consoleInput.min.x);
				}

				float cursorDiffRight = cursorPos.x - (inputRect.max.x - inputScrollMargin);
				if(cursorDiffRight > 0) {
					inputOffset = inputOffset + cursorDiffRight;
				}
			}

			// 
			// Drawing.
			//
			
			Rect inputRect = rectExpand(consoleInput, vec2(-consolePadding.x, 0));
			Vec2 inputStartPos = inputRect.min + vec2(-inputOffset, inputHeight/2);

			// Selection.

			drawTextSelection(inputBuffer, inputFont, inputStartPos, cursorIndex, markerIndex, selectionColor, vec2i(-1,0));

			// Text.

			dcEnable(STATE_SCISSOR);
			dcScissor(scissorRectScreenSpace(inputRect, res.h));

			dcText(inputBuffer, inputFont, inputStartPos, inputFontColor, vec2i(-1,0));

			dcDisable(STATE_SCISSOR);

			// Cursor.

			cursorTime += dt*cursorSpeed;
			Vec4 cmod = vec4(0,cos(cursorTime)*cursorColorMod - cursorColorMod,0,0);

			Vec2 cursorPos = textIndexToPos(inputBuffer, inputFont, inputStartPos, cursorIndex, vec2i(-1,0));

			bool cursorAtEnd = cursorIndex == strLen(inputBuffer);
			float cWidth = cursorAtEnd ? getTextDim("M", inputFont).w : cursorWidth;
			Rect cursorRect = rectCenDim(cursorPos, vec2(cWidth, inputFont->height));

			if(cursorAtEnd) cursorRect = rectAddOffset(cursorRect, vec2(rectGetDim(cursorRect).w/2, 0));
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


