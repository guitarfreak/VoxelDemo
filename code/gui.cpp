
Vec2 getTextDim(char* text, Font* font) {
	Vec2 textDim = stbtt_GetTextDim(font->cData, font->height, font->glyphStart, text);
	return textDim;
}

float getTextPos(char* text, int index, Font* font) {
	float result = 0;
	Vec2 pos = vec2(0,0);

	// no support for new line
	int length = strLen(text);
	for(int i = 0; i < length+1; i++) {
		char t = text[i];

		if(i == index) {
			result = pos.x;
			break;
		}

		stbtt_aligned_quad q;
		stbtt_GetBakedQuad(font->cData, font->tex.dim.w, font->tex.dim.h, t-font->glyphStart, &pos.x, &pos.y, &q, 1);
		Rect r = rect(q.x0, q.y0, q.x1, q.y1);
	}

	return result;
}


struct GuiInput {
	Vec2 mousePos;
	int mouseWheel;
	bool mouseClick, mouseDown;
	bool escape, enter, space, backSpace, del, home, end;
	bool left, right, up, down;
	bool shift, ctrl;
	char* charList;
	int charListSize;
};

char* guiColorStrings[] = {
	"panelColor",
	"regionColor",
	"sectionColor",
	"selectionColor",
	"resizeButtonColor",
	"textColor",
	"shadowColor",
	"hoverColorAdd",
	"switchColorOn",
	"switchColorOff",
	"sliderColor",
	"sliderHoverColorAdd",
	"cursorColor",
};

char* guiSettingStrings[] = {
	"offsets",
	"border",
	"minSize",
	"sliderSize",
	"sliderResetDistance",
	"fontOffset",
	"sectionOffset",
	"fontHeightOffset",
	"sectionIndent",
	"scrollBarWidth",
	"scrollBarSliderSize",
	"cursorWidth",
	"textShadow",
};

union GuiColors {
	struct {
		Vec4 panelColor;
		Vec4 regionColor;
		Vec4 sectionColor;
		Vec4 selectionColor;
		Vec4 resizeButtonColor;
		Vec4 textColor;
		Vec4 shadowColor;
		Vec4 hoverColorAdd;
		Vec4 switchColorOn;
		Vec4 switchColorOff;
		Vec4 sliderColor;
		Vec4 sliderHoverColorAdd;
		Vec4 cursorColor;
	};
	Vec4 e[13];
};

struct GuiSettings {
	Vec2i border;
	Vec2i offsets;
	Vec2i minSize;
	int sliderSize;
	float sliderResetDistance;
	float fontOffset;
	float sectionOffset;
	float fontHeightOffset;
	float sectionIndent;
	int scrollBarWidth;
	int scrollBarSliderSize;
	int cursorWidth;
	int textShadow;
};

struct Gui;
bool guiCreateSettingsFile();
void guiSave(Gui* gui, int element, int slot);
void guiLoad(Gui* gui, int element, int slot);
void guiSaveAll(Gui* gui, int slot);
void guiLoadAll(Gui* gui, int slot);

struct Font;

struct Gui {
	GuiColors colors;
	GuiSettings settings;
	Vec2 cornerPos;
	Vec2 panelStartDim;

	GuiInput input;
	Font* font;
	Vec2i screenRes;

	Vec2 startPos;
	float panelWidth;
	float fontHeight;

	int currentId;
	int activeId;
	int wantsToBeActiveId;
	int hotId;

	Vec2 currentPos;
	Vec2 currentDim;
	Vec2 lastPos;
	Vec2 lastDim;
	Vec2 lastMousePos;
	Vec2 mouseDelta;

	float mainScrollAmount;
	int mainScrollHeight;

	bool columnMode;
	float columnArray[10];
	float columnSize;
	int columnIndex;

	Rect scissorStack[8];
	int scissorStackSize;

	int scrollStackIndexX;
	int scrollStackIndexY[4];
	float scrollStack[4][4];
	float scrollStartY[4];
	float scrollEndY[4];
	bool showScrollBar[4];
	float scrollAnchor;
	Vec2 scrollMouse;
	float scrollValue;

	void* originalPointer;
	int textBoxIndex;
	char textBoxText[50];
	int textBoxTextSize;
	int textBoxTextCapacity;
	int selectionAnchor;

	bool secondMode;

	float heightStack[4];
	int heightStackIndex;

	void init(Rect panelRect, int saveSlot) {
		*this = {};

		if(guiCreateSettingsFile() || saveSlot == -1) {
			Vec3 mainColor = vec3(280,0.7f,0.3f);

			colors.panelColor 		= vec4(hslToRgb(mainColor), 1);
			colors.regionColor 		= vec4(hslToRgb(mainColor + vec3(0,0,-0.1f)), 1);
			colors.sectionColor 		= vec4(hslToRgb(mainColor + vec3(45,0,0)), 1);
			colors.resizeButtonColor 	= vec4(hslToRgb(mainColor + vec3(-45,-0.1f,0)), 1);
			colors.selectionColor 	= vec4(hslToRgb(mainColor + vec3(-180,0,0)), 1);

			colors.switchColorOn 		= vec4(0,0.3f,0,0);
			colors.switchColorOff 	= vec4(0.3f,0,0,0);

			colors.textColor 			= vec4(0.9f, 0.9f, 0.9f, 1);
			colors.hoverColorAdd 		= vec4(0.2f,0.2f,0.2f,0);
			colors.sliderHoverColorAdd= colors.hoverColorAdd;
			colors.sliderColor 		= vec4(0.6f,0.6f,0.6f,0.8f);
			colors.cursorColor 		= vec4(1,1,1,1);

			settings.sliderSize = 4;
			settings.sliderResetDistance = 100;
			settings.offsets = vec2i(1,1);
			settings.border = vec2i(4,-4);
			settings.minSize = vec2i(100,100);
			settings.fontHeightOffset = 1.0f;
			settings.fontOffset = 0.15f;
			settings.sectionOffset = 1.15f;
			settings.sectionIndent = 0.08f;
			settings.scrollBarWidth = 20;
			settings.scrollBarSliderSize = 30;
			settings.cursorWidth = 1;

			cornerPos = rectGetUL(panelRect);
			panelStartDim = rectGetDim(panelRect);
			panelStartDim.h += settings.border.h*2;
		} else {
			guiLoadAll(this, saveSlot);
		}
	}

	void heightPush(float heightOffset) { heightStack[++heightStackIndex] = heightOffset; }
	void heightPop() { heightStackIndex--; }

	void doScissor(Rect r) {
		Vec2 dim = rectGetDim(r);
		if(dim.w < 0 || dim.h < 0) r = rect(0,0,0,0);
		dcScissor(r);
	}

	void setScissor(bool m) {
		if(m) dcEnable(STATE_SCISSOR);
		else dcDisable(STATE_SCISSOR);
	}

	Rect getCurrentScissor() {
		Rect result = scissorStack[scissorStackSize-1];
		return result;
	}

	Rect scissorPush(Rect newRect) {
		if(scissorStackSize == 0) {
			scissorStack[scissorStackSize++] = newRect;
			doScissor(scissorRect(newRect));

			return newRect;
		}

		if((newRect.max.x < newRect.min.x || newRect.max.y < newRect.min.y) || 
			newRect == rect(0,0,0,0)) {
			doScissor(scissorRect(rect(0,0,0,0)));
			scissorStack[scissorStackSize++] = rect(0,0,0,0);

			return rect(0,0,0,0);
		}

		Rect currentRect = getCurrentScissor();
		Rect intersection;
		rectGetIntersection(&intersection, newRect, currentRect);

		doScissor(scissorRect(intersection));

		scissorStack[scissorStackSize++] = intersection;
		return intersection;
	}

	// Rect scissorPushWidth(float width) {
	// 	if(width <= 0) {
	// 		Rect result = scissorPush(rect(0,0,0,0));
	// 		return result;
	// 	}
	// 	else {
	// 		Rect currentRect = getCurrentScissor();
	// 		currentRect 
	// 		Rect result = scissorPush();
	// 		return result;
	// 	}
	// }

	void scissorPop() {
		scissorStackSize--;
		scissorStackSize = clampMin(scissorStackSize, 0);
		if(scissorStackSize > 0) doScissor(scissorRect(scissorStack[scissorStackSize-1]));
	}

	Rect scissorRect(Rect r) {
		Rect scissorRect = {r.min.x, r.min.y+screenRes.h, r.max.x, r.max.y+screenRes.h};
		return scissorRect;
	}

	void drawText(char* text, int align = 1) {
		Rect region = getCurrentRegion();
		scissorPush(region);

		Vec2 textPos = rectGetCen(region) + vec2(0,fontHeight*settings.fontOffset);
		if(align == 0) textPos.x -= rectGetDim(region).w*0.5f;
		else if(align == 2) textPos.x += rectGetDim(region).w*0.5f;

		dcText(text, font, textPos, colors.textColor, align, 1, settings.textShadow, colors.shadowColor);		

		scissorPop();
	}

	void drawText(char* text, int align, Rect region) {
		scissorPush(region);

		Vec2 textPos = rectGetCen(region) + vec2(0,fontHeight*settings.fontOffset);
		if(align == 0) textPos.x -= rectGetDim(region).w*0.5f;
		else if(align == 2) textPos.x += rectGetDim(region).w*0.5f;

		dcText(text, font, textPos, colors.textColor, align, 1, settings.textShadow, colors.shadowColor);		

		scissorPop();
	}

	int getTextWidth(char* text) {
		Vec2 textDim = getTextDim(text, font);
		return textDim.w;
	}	

	void drawRect(Rect r, Vec4 color, bool scissor = false) {
		if(scissor) scissorPush(getCurrentRegion());
		dcRect(r, rect(0,0,1,1), color, (int)getTexture(TEXTURE_WHITE)->id);
		if(scissor) scissorPop();
	}

	void drawTextBox(Rect region, char* text, Vec4 bgColor, int align = 0) {
		drawRect(region, bgColor, false);
		drawText(text, align, region);
	}

	void start(GuiInput guiInput, Font* font, Vec2i res, bool moveable = true, bool resizeable = true, bool clipToWindow = true) {
		this->font = font;
		fontHeight = font->height;
		lastMousePos = input.mousePos;
		input = guiInput;
		input.mousePos.y *= -1;
		mouseDelta = input.mousePos - lastMousePos;
		screenRes = res;
		currentId = 0;
		scrollStackIndexX = 0;
		for(int i = 0; i < 4; i++) scrollStackIndexY[i] = 0;
		hotId = wantsToBeActiveId;
		wantsToBeActiveId = 0;

		for(int i = 0; i < arrayCount(heightStack); i++) heightStack[i] = 1;
		heightStackIndex = 0;

		Rect background = rectULDim(cornerPos, vec2(panelWidth+settings.border.x*2, -(lastPos.y-cornerPos.y)+lastDim.h - settings.border.y));

		incrementId();
		// drag window
		{
			Vec2 dragDelta = vec2(0,0);
			Vec2 oldPos = cornerPos;
			if(!input.ctrl && drag(background, &dragDelta) && moveable) {
				cornerPos += dragDelta;
			}

			// clamp(&cornerPos, rect(0, -res.y, res.x - rectGetDim(background).w+1, 0.5f));
			if(clipToWindow)
				clamp(&cornerPos, rect(0, -res.y + rectGetDim(background).h, res.x - rectGetDim(background).w+1, 0.5f));
			background = rectAddOffset(background, cornerPos - oldPos);
		}

		// resize window
		Rect resizeRegion;
		{
			resizeRegion = rect(rectGetDR(background)-vec2(settings.border.x*2,0), rectGetDR(background)+vec2(0,-settings.border.y*2));
			Vec2 dragDelta = vec2(0,0);
			int oldMainScrollHeight = panelStartDim.h;
			float oldPanelWidth = panelStartDim.w;

			if(input.ctrl) {
				if(drag(background, &dragDelta) && resizeable) {
					panelStartDim.h += -dragDelta.y;
					panelStartDim.w += dragDelta.x;
				}
			}
			{
				incrementId();
				if(drag(resizeRegion, &dragDelta) && resizeable) {
					panelStartDim.h += -dragDelta.y;
					panelStartDim.w += dragDelta.x;
				}
			}

			// garbage
			if(dragDelta != vec2(0,0) && resizeable) {
				Rect screenRect = rect(0,-screenRes.h,screenRes.w,0);
				dragDelta.x = clampMax(dragDelta.x, screenRect.max.x - background.max.x +1);
				dragDelta.y = clampMin(dragDelta.y, screenRect.min.y - background.min.y);
				panelStartDim.w = oldPanelWidth + dragDelta.x;
				panelStartDim.h = oldMainScrollHeight - dragDelta.y;
			}
		
			panelStartDim.h = clamp(panelStartDim.h, settings.minSize.y, screenRes.h+settings.border.h*2+1);
			panelStartDim.w = clamp(panelStartDim.w, settings.minSize.x, screenRes.w-settings.border.w*2+1);

			Vec2 dragAfterClamp = vec2(panelStartDim.w - oldPanelWidth, oldMainScrollHeight - panelStartDim.h);
			if(showScrollBar[0] == false) dragAfterClamp.y = 0;
			resizeRegion = rectAddOffset(resizeRegion, dragAfterClamp);
			background = rectExpand(background, 0, dragAfterClamp.y, dragAfterClamp.x, 0);

			if(clipToWindow)
				clamp(&cornerPos, rect(0, -res.y + rectGetDim(background).h, res.x - rectGetDim(background).w+1, 0.5f));
		}

		setScissor(true);
		scissorPush(background);

		drawRect(background, colors.panelColor, false);
		drawRect(resizeRegion, colors.resizeButtonColor, false);

		startPos = cornerPos + settings.border;
		// currentPos = startPos + vec2(0, getDefaultYOffset());
		// getDefaultHeight();
		defaultHeight();
		currentPos = startPos + vec2(0, getDefaultYOffset());
		
		panelWidth = panelStartDim.w;
		mainScrollHeight = panelStartDim.h;

		// panel scrollbar
		{
			beginScroll(mainScrollHeight, &mainScrollAmount);
		}
	}


	void end() {
		// terrible hack
		currentPos.y = currentPos.y - (currentDim.h - getDefaultHeightEx());
		currentDim.h = getDefaultHeightEx();

		// panel scrollbar
		{
			endScroll();
		}

		scissorPop();
		setScissor(false);
	}

	float getDefaultHeightEx(){return fontHeight*settings.fontHeightOffset;};
	float getDefaultHeight() {
		float height = fontHeight*settings.fontHeightOffset;
		float stackValue = heightStack[heightStackIndex];

		// switch from multiplier to pixels if over threshhold
		if(stackValue > 10) height = stackValue;
		else height *= stackValue;

		return height;
	};
	float getDefaultWidth(){return panelWidth;};
	float getDefaultYOffset(){return currentDim.h+settings.offsets.h;};
	float getYOffset(){return currentDim.h+settings.offsets.h;};
	// float getDefaultYOffset(){return getDefaultHeight()+settings.offsets.h;};
	Vec2 getDefaultPos(){return vec2(startPos.x, currentPos.y - getDefaultYOffset());};
	void advancePosY() 	{currentPos.y -= getDefaultYOffset();};
	void defaultX() 	{currentPos.x = startPos.x;};
	void defaultHeight(){currentDim.h = getDefaultHeight();};
	void defaultWidth() {currentDim.w = getDefaultWidth();};

	void defaultValues() {
		advancePosY();
		defaultX();
		defaultHeight();
		defaultWidth();
	}

	void setLastPosition() {
		lastPos = currentPos;
		lastDim = currentDim;
	}

	void incrementId() {currentId++;};

	bool pre() {
		incrementId();

		if(columnMode) {
			if(columnIndex == 0) {
				defaultValues();
			} else {
				defaultHeight();
				defaultWidth();
				currentPos.x += lastDim.w + settings.offsets.w;
			}
			currentDim.w = columnArray[columnIndex];

			columnIndex++;
			if(columnIndex == columnSize) columnMode = false;
		} else {
			defaultValues();
		}

		setLastPosition();

		if(scissorStack[scissorStackSize-1] == rect(0,0,0,0)) return false;
		else if(currentDim.w <= 0) return false;
		else return true;
	}


	// no usage yet
	void post() {
	}

	void advanceY(float v) {
		if(v >= 5) currentPos.y -= v;
		else currentPos.y -= getDefaultYOffset()*v;
		// else currentPos.y -= getYOffset();
	}

	void div(float* c, int size) {
		int elementCount = size;
		float dynamicSum = 0;
		int flowCount = 0;
		float staticSum = 0;
		int staticCount = 0;
		for(int i = 0; i < size; i++) {
			float val = c[i];

			if(val == 0) { 			// float element
				flowCount++;
			} else if(val <= 1) { 	// dynamic element
				dynamicSum += val;
			} else { 				// static element
				staticSum += val;
				staticCount++;
			}
		}

		if(flowCount) {
			float flowVal = abs(dynamicSum-1)/(float)flowCount;
			for(int i = 0; i < size; i++)
				if(c[i] == 0) c[i] = flowVal;
		}

		float totalWidth = getDefaultWidth() - staticSum - settings.offsets.w*(elementCount-1);
		for(int i = 0; i < size; i++) {
			float val = c[i];
			if(val <= 1) {
				columnArray[i] = val * totalWidth;
			} else {
				columnArray[i] = val;
			}
		}

		columnSize = size;
		columnMode = true;
		columnIndex = 0;
	}
	void div(Vec2 v) {div(v.e, 2);};
	void div(Vec3 v) {div(v.e, 3);};
	void div(Vec4 v) {div(v.e, 4);};
	void div(float a, float b) {div(vec2(a,b));};
	void div(float a, float b, float c) {div(vec3(a,b,c));};
	void div(float a, float b, float c, float d) {div(vec4(a,b,c,d));};

	Rect getCurrentRegion() {
		Rect result = rectULDim(currentPos, currentDim);
		return result;
	}

	void empty() {
		pre();
		post();
	}

	void label(char* text, int align = 1, Vec4 bgColor = vec4(0,0,0,0)) {
		if(!pre()) return;

		if(bgColor != vec4(0,0,0,0)) drawRect(getCurrentRegion(), bgColor, false);
		drawText(text, align);
		post();
	}

	bool beginSection(char* text, bool* b) {
		heightPush(settings.sectionOffset);
		defaultWidth();
		div(vec2(getDefaultHeight(), 0)); switcher("", b); label(text, 1, colors.sectionColor);
		heightPop();

		float indent = panelWidth*settings.sectionIndent;
		startPos.x += indent*0.5f;
		panelWidth -= indent;

		return *b;
	}

	void endSection() {
		panelWidth = panelWidth*(1/(1-settings.sectionIndent));
		startPos.x -= panelWidth*settings.sectionIndent*0.5f;
	}

	void beginScroll(int scrollHeight, float* scrollAmount) {
		scrollEndY[scrollStackIndexX] = currentPos.y - scrollHeight - settings.offsets.h;

		Rect r = rectULDim(getDefaultPos(), vec2(panelWidth, scrollHeight));

		float scrollValue = scrollStack[scrollStackIndexX][scrollStackIndexY[scrollStackIndexX]];
		float diff = (scrollValue - scrollHeight - settings.offsets.y);
		if(diff > 0) {
			panelWidth -= settings.scrollBarWidth + settings.offsets.w;

			currentPos.y += *scrollAmount * diff;
			Rect scrollRect = r;
			scrollRect.min.x += panelWidth + settings.offsets.w;
			// r.min.x += panelWidth;

			Rect regCen = rectGetCenDim(scrollRect);
			float sliderHeight = regCen.dim.h - diff;
			sliderHeight = clampMin(sliderHeight, settings.scrollBarSliderSize);

			slider(scrollAmount, 0, 1, true, scrollRect, sliderHeight);

			currentPos.y += getDefaultYOffset();

			Rect scissorRect = r;
			scissorRect.max.x -= settings.offsets.w + settings.scrollBarWidth;
			scissorPush(scissorRect);
		} else {
			incrementId();
			scissorPush(r);
		}

		showScrollBar[scrollStackIndexX] = diff > 0 ? true : false;

		scrollStartY[scrollStackIndexX] = currentPos.y;

		scrollStackIndexX++;
	};

	void endScroll() {
		scrollStackIndexX--;

		scissorPop();

		scrollStack[scrollStackIndexX][scrollStackIndexY[scrollStackIndexX]++] = -((currentPos.y) - scrollStartY[scrollStackIndexX]);
		// float h = currentDim.h - getDefaultHeightEx();
		// scrollStack[scrollStackIndexX][scrollStackIndexY[scrollStackIndexX]++] = -((currentPos.y + currentDim.h) - scrollStartY[scrollStackIndexX]);
		// scrollStack[scrollStackIndexX][scrollStackIndexY[scrollStackIndexX]++] = -((currentPos.y + h) - scrollStartY[scrollStackIndexX]);

		if(showScrollBar[scrollStackIndexX]) {
			currentPos.y = scrollEndY[scrollStackIndexX];
			// currentPos.y = scrollEndY[scrollStackIndexX] + currentDim.h;
			panelWidth += settings.scrollBarWidth + settings.offsets.w;
		}

		setLastPosition();
	}

	bool getMouseOver(Vec2 mousePos, Rect region, bool noScissor = false) {
		bool overScissorRegion = noScissor ? true : pointInRect(mousePos, scissorStack[scissorStackSize-1]);
		bool mouseOver = pointInRect(mousePos, region) && overScissorRegion;
		return mouseOver;
	}

	bool getActive() {
		bool active = (activeId == currentId);
		return active;
	}

	bool getHot() {
		bool hot = (hotId == currentId);
		return hot;
	}

	bool setActive(bool mouseOver, int releaseType = 0) {
		if((hotId == currentId) && activeId == 0 && (mouseOver && input.mouseClick)) activeId = currentId;
		else if(activeId == currentId) {
			if( releaseType == 0 || 
				(releaseType == 1 && !input.mouseDown)) {
				activeId = 0;
				secondMode = false;
			}
		}

		if(mouseOver) {
			wantsToBeActiveId = max(wantsToBeActiveId, currentId);
		} 


		bool active = getActive();
		return active;
	}

	Vec4 getColorAdd(bool active, bool mouseOver, int type = 0) {
		if(type == 2) {
			Vec4 color = active || (mouseOver && activeId == 0) ? colors.hoverColorAdd : vec4(0,0,0,0);
			return color;
		}

		Vec4 colorAdd = type == 0 ? colors.hoverColorAdd : colors.sliderHoverColorAdd;
		Vec4 color;
		if(active) color = colorAdd*2;
		else if(mouseOver && activeId == 0) color = colorAdd;
		else color = {};

		return color;
	}

	bool drag(Rect region, Vec2* dragDelta, Vec4 color = vec4(0,0,0,0)) {
		// incrementId();

		bool mouseOver = getMouseOver(input.mousePos, region, true);
		bool active = setActive(mouseOver, 1);

		if(active) {
			*dragDelta = mouseDelta;
		}

		if(color != vec4(0,0,0,0)) drawRect(region, color);

		post();
		return active;
	}

	bool button(char* text, int switcher = 0, Vec4 bgColor = vec4(0,0,0,0)) {
		if(!pre()) return false;

		Rect region = getCurrentRegion();
		bool mouseOver = getMouseOver(input.mousePos, region);
		bool active = setActive(mouseOver);
		Vec4 colorAdd = getColorAdd(active, mouseOver);

		Vec4 finalColor = (bgColor==vec4(0,0,0,0) ? colors.regionColor:bgColor) + colorAdd;

		if(switcher) {
			if(switcher == 2) finalColor += colors.switchColorOn;
			else finalColor += colors.switchColorOff;
		}

		drawRect(region, finalColor);
		drawText(text);

		post();
		return active;
	}

	bool switcher(char* text, bool* value) {
		bool active = false;
		if(button(text, (int)(*value) + 1)) {
			*value = !(*value);
			active = true;
		}

		return active;
	}

	bool slider(void* value, float min, float max, bool typeFloat = true, Rect reg = rect(0,0,0,0), float sliderS = 0) {
		if(!pre()) return false;

		bool verticalSlider = sliderS == 0 ? false : true;
		Vec2 valueRange = verticalSlider ? vec2(max,min) : vec2(min,max);

		Rect region = reg == rect(0,0,0,0) ? getCurrentRegion() : reg;
		scissorPush(region);

		float sliderSize = verticalSlider ? sliderS : settings.sliderSize;
		if(typeFloat) sliderSize = verticalSlider ? sliderS : settings.sliderSize;
		else {
			sliderSize = rectGetDim(region).w * 1/(float)((int)roundInt(max)-(int)roundInt(min) + 1);
			sliderSize = clampMin(sliderSize, settings.sliderSize);
		}

		float calc = typeFloat ? *((float*)value) : *((int*)value);
		Vec2 sliderRange = verticalSlider ? vec2(region.min.y, region.max.y-sliderSize) : 
											vec2(region.min.x, region.max.x-sliderSize);
		float sliderPos = mapRangeClamp(calc, valueRange, sliderRange);
		Rect sliderRegion = verticalSlider ? rect(region.min.x, sliderPos, region.max.x, sliderPos+sliderSize) :
											 rect(sliderPos, region.min.y, sliderPos+sliderSize, region.max.x);
		
		bool activeBefore = getActive();
		bool mouseOverSlider = getMouseOver(input.mousePos, sliderRegion);
		bool active = setActive(mouseOverSlider, 1);
		Vec4 colorAdd = getColorAdd(active, mouseOverSlider, 1);

		if(!activeBefore && active) {
			scrollAnchor = verticalSlider ? input.mousePos.y - sliderPos : input.mousePos.x - sliderPos;
			scrollMouse = input.mousePos;
			scrollValue = typeFloat ? *(float*)value : *(int*)value;
		}

		bool resetSlider = false;
		if(active) {
			if(verticalSlider) {
				calc = mapRange(input.mousePos.y - scrollAnchor, sliderRange, valueRange);
				calc = clamp(calc, valueRange.y, valueRange.x);
			} else {
				calc = mapRangeClamp(input.mousePos.x - scrollAnchor, sliderRange, valueRange);
			}

			float dist = verticalSlider ? abs(scrollMouse.x - input.mousePos.x) : 
										  abs(scrollMouse.y - input.mousePos.y);
			if(dist > settings.sliderResetDistance) calc = scrollValue;

			if(!verticalSlider && typeFloat) {
				if(input.ctrl) calc = roundInt(calc);
				else if(input.shift) calc = roundFloat(calc, 10);
			}

			if(!typeFloat) calc = roundInt(calc);
			
			sliderPos = mapRangeClamp(calc, valueRange, sliderRange);
			sliderRegion = verticalSlider ? rect(region.min.x, sliderPos, region.max.x, sliderPos+sliderSize) :
											rect(sliderPos, region.min.y, sliderPos+sliderSize, region.max.x);
		}

		if(typeFloat) *(float*)value = calc;
		else *(int*)value = calc;

		// draw background
		drawRect(region, colors.regionColor);

		// draw slider
		drawRect(sliderRegion, colors.sliderColor + colorAdd);

		// draw text
		char* text;
		if(verticalSlider) text = "";
		if(typeFloat) text = fillString("%f", *((float*)value));
		else text = fillString("%i", *((int*)value));
		drawText(text);

		scissorPop();

		post();
		return active;
	}

	bool slider(int* value, float min, float max) {
		return slider(value, min, max, false);
	}

	bool slider2d(float v[2], float min, float max) {
		bool active1 = slider(&v[0], min, max);
		bool active2 = slider(&v[1], min, max);
		return (active1 || active2);
	}

	// really?
	bool slider2d(float v[2], float min, float max, float min2, float max2) {
		bool active1 = slider(&v[0], min, max);
		bool active2 = slider(&v[1], min2, max2);
		return (active1 || active2);
	}


	bool textBox(void* value, int type, int textSize, int textCapacity) {
		if(!pre()) return false;

		bool activeBefore = getActive();

		Rect region = getCurrentRegion();
		bool mouseOver = getMouseOver(input.mousePos, region);
		bool active = setActive(mouseOver, 2);
		Vec4 colorAdd = getColorAdd(active, mouseOver, 2);

		if(!activeBefore && active) {
			int textLength;
			if(type == 0) {
				textLength = textSize == 0 ? strLen((char*)value) : textSize;
				strCpy(textBoxText, (char*)value, textLength);
			} else if(type == 1) {
				intToStr(textBoxText, *((int*)value));
				textLength = strLen(textBoxText);
			} else {
				floatToStr(textBoxText, *((float*)value));
				textLength = strLen(textBoxText);
			}

			originalPointer = value;
			textBoxIndex = textLength;
			textBoxTextSize = textLength;
			textBoxTextCapacity = textCapacity;

			selectionAnchor = -1;
		}

		drawRect(region, colors.regionColor + colorAdd);

		if(active) {
			int oldTextBoxIndex = textBoxIndex;
			if(input.left) textBoxIndex--;
			if(input.right) textBoxIndex++;
			clampInt(&textBoxIndex, 0, textBoxTextSize);

			if(input.ctrl) {
				if(input.left) textBoxIndex = strFindBackwards(textBoxText, ' ', textBoxIndex);
				else if(input.right) {
					int foundIndex = strFind(textBoxText, ' ', textBoxIndex);
					if(foundIndex == 0) textBoxIndex = textBoxTextSize;
					else textBoxIndex = foundIndex - 1;
				}
			}
			if(input.home) textBoxIndex = 0;
			if(input.end) textBoxIndex = textBoxTextSize;

			if(selectionAnchor == -1) {
				if(input.shift && (oldTextBoxIndex != textBoxIndex)) selectionAnchor = oldTextBoxIndex;
			} else {
				if(input.shift) {
					if(textBoxIndex == selectionAnchor) selectionAnchor = -1;
				} else {
					if(input.left || input.right || input.home || input.end) {
						if(input.left) textBoxIndex = min(selectionAnchor, oldTextBoxIndex);
						else if(input.right) textBoxIndex = max(selectionAnchor, oldTextBoxIndex);
						selectionAnchor = -1;
					}
				}
			}

			for(int i = 0; i < input.charListSize; i++) {
				if(textBoxTextSize >= textBoxTextCapacity) break;
				char c = input.charList[i];
				if(type == 1 && ((c < '0' || c > '9') && c != '-')) break;
				if(type == 2 && ((c < '0' || c > '9') && c != '.' && c != '-')) break; 

				if(selectionAnchor != -1) {
					int index = min(selectionAnchor, textBoxIndex);
					int size = max(selectionAnchor, textBoxIndex) - index;
					strErase(textBoxText, index, size);
					textBoxTextSize -= size;
					textBoxIndex = index;
					selectionAnchor = -1;
				}

				strInsert(textBoxText, textBoxIndex, input.charList[i]);
				textBoxTextSize++;
				textBoxIndex++;
			}

			if(selectionAnchor != -1) {
				if(input.backSpace || input.del) {
					// code duplication
					int index = min(selectionAnchor, textBoxIndex);
					int size = max(selectionAnchor, textBoxIndex) - index;
					strErase(textBoxText, index, size);
					textBoxTextSize -= size;
					textBoxIndex = index;
					selectionAnchor = -1;
				}
			} else {
				if(input.backSpace && textBoxIndex > 0) {
					strRemove(textBoxText, textBoxIndex, textBoxTextSize);
					textBoxTextSize--;
					textBoxIndex--;
				}
				
				if(input.del && textBoxIndex < textBoxTextSize) {
					strRemove(textBoxText, textBoxIndex+1, textBoxTextSize);
					textBoxTextSize--;
				}
			}

			Rect regCen = rectGetCenDim(region);
			Vec2 textStart = regCen.cen - vec2(regCen.dim.w*0.5f,0);
			Vec2 cursorPos = textStart + vec2(getTextPos(textBoxText, textBoxIndex, font), 0);
			if(textBoxIndex == 0) cursorPos += vec2(1,0); // avoid scissoring cursor on far left

			Rect cursorRect = rectCenDim(cursorPos, vec2(settings.cursorWidth,font->height));

			if(selectionAnchor != -1) {
				int start = min(textBoxIndex, selectionAnchor);
				int end = max(textBoxIndex, selectionAnchor);

				Vec2 selectionStartPos = textStart + vec2(getTextPos(textBoxText, start, font), 0);
				Vec2 selectionEndPos = textStart + vec2(getTextPos(textBoxText, end, font), 0);
				float selectionWidth = selectionEndPos.x - selectionStartPos.x;

				Vec2 center = selectionStartPos+selectionWidth*0.5;
				Rect selectionRect = rectCenDim(selectionStartPos + vec2(selectionWidth*0.5f,0), vec2(selectionWidth,font->height));
				drawRect(selectionRect, colors.selectionColor, true);
			} 

			drawText(textBoxText, 0);
			drawRect(cursorRect, colors.cursorColor, true);

			if(input.enter) {
				if(type == 0) strCpy((char*)originalPointer, textBoxText, textBoxTextSize);
				else if(type == 1) *((int*)originalPointer) = strToInt(textBoxText);
				else *((float*)originalPointer) = strToFloat(textBoxText);

				activeId = 0;
			}

			if(input.mouseClick && !getMouseOver(input.mousePos, region)) activeId = 0;

			if(input.escape) activeId = 0;

		} else {
			if(type == 0) drawText((char*)value, 0);
			else { 
				char* buffer = getTString(50); // unfortunate
				strClear(buffer);
				if(type == 1) {
					intToStr(buffer, *((int*)value));
					drawText(buffer, 0);
				} else {
					floatToStr(buffer, *((float*)value));
					drawText(buffer, 0);
				}
			}
		}

		post();
		return active;
	}

	bool textBoxChar(char* text, int textSize = 0, int textCapacity = 0) {
		return textBox(text, 0, textSize, textCapacity);
	}
	bool textBoxInt(int* number) {
		return textBox(number, 1, 0, arrayCount(textBoxText));
	}
	bool textBoxFloat(float* number) {
		return textBox(number, 2, 0, arrayCount(textBoxText));
	}
};


struct SerializeData {
	char* name;
	int offset;
	int size;
};

void saveData(void* data, SerializeData sd, char* fileName, int fileOffset) {
	writeBufferSectionToFile((char*)(data) + sd.offset, fileName, fileOffset + sd.offset, sd.size);
}

void loadData(void* data, SerializeData sd, char* fileName, int fileOffset) {
	readFileSectionToBuffer((char*)(data) + sd.offset, fileName, fileOffset + sd.offset, sd.size);
}

const SerializeData settingData[] = {
	{"Colors", offsetof(Gui, colors), memberSize(Gui, colors)}, 
	{"Settings", offsetof(Gui, settings), memberSize(Gui, settings)}, 
	{"Layout", offsetof(Gui, cornerPos), memberSize(Gui, cornerPos) + memberSize(Gui, panelStartDim)}, 
};

const char* settingFile = "C:\\Projects\\Hmm\\data\\guiSettings.txt";
const int settingSlotSize = settingData[0].size + settingData[1].size + settingData[2].size;
const int settingSlots = 4;
const int settingFileSize = settingSlotSize*settingSlots;

bool guiCreateSettingsFile() {
	if(!fileExists((char*)settingFile)) {
		createFileAndOverwrite((char*)settingFile, settingFileSize);
		return true;
	}

	return false;
}

void guiSave(Gui* gui, int element, int slot) {
	saveData(gui, settingData[element], (char*)settingFile, settingSlotSize*slot);
}

void guiLoad(Gui* gui, int element, int slot) {
	loadData(gui, settingData[element], (char*)settingFile, settingSlotSize*slot);
}

void guiSaveAll(Gui* gui, int slot) {
	for(int i = 0; i < arrayCount(settingData); i++) guiSave(gui, i, slot);
}

void guiLoadAll(Gui* gui, int slot) {
	for(int i = 0; i < arrayCount(settingData); i++) guiLoad(gui, i, slot);
}


void guiSettings(Gui* gui) {
	static int maxStringWidth = 0;
	static bool staticInit = true;
	if(staticInit) {
		// guiCreateSettingsFile();

		staticInit = false;
		for(int i = 0; i < arrayCount(guiSettingStrings); i++) maxStringWidth = max(maxStringWidth, gui->getTextWidth(guiSettingStrings[i]));
		for(int i = 0; i < arrayCount(guiColorStrings); i++) maxStringWidth = max(maxStringWidth, gui->getTextWidth(guiColorStrings[i]));
	}


	int colorCount = arrayCount(gui->colors.e);

	static int slot = 0;
	gui->div(vec2(gui->getTextWidth("Slot: "),0)); gui->label("Slot: "); gui->slider(&slot, 0, settingSlots-1);


	for(int i = 0; i < arrayCount(settingData); i++) {
		char* string = settingData[i].name;
		gui->div(gui->getTextWidth(string),0,0); gui->label(string, 0); 
		if(gui->button("Save")) guiSave(gui, i, slot);
		if(gui->button("Load")) guiLoad(gui, i, slot);
	}

	gui->heightPush(2); gui->div(vec3(0,0,0)); 
		if(gui->button("SaveAll")) {
			guiSaveAll(gui, slot);
		}
		if(gui->button("LoadAll")) {
			guiLoadAll(gui, slot);
		} 
		if(gui->button("ResetColors")) {
			for(int i = 0; i < colorCount; i++) {
				gui->colors.e[i] = vec4(0,0,0,1);
			}
		}
	gui->heightPop();


	static int tw = maxStringWidth + 2;
	bool mainUpdated = false;
	Vec3 colorUpdate = vec3(0,0,0);
	for(int i = 0; i < colorCount; i++) {
		bool main = i == 0 ? true : false;

		gui->advanceY(5);

		gui->heightPush(2.5f);
			int colorBoxWidth = gui->getDefaultHeight();
			float divWidths[] = {tw, colorBoxWidth, 0, 0, 0};
			gui->div(divWidths, arrayCount(divWidths)); 

			gui->label(guiColorStrings[i],0); 
			if(gui->button("", 0, gui->colors.e[i]) && !main) {
				gui->colors.e[i] = vec4(0,0,0,1);
			}
		gui->heightPop();

		gui->heightPush(1.5f);
		Vec3 hslColor = rgbToHsl(gui->colors.e[i].xyz);

		if(mainUpdated) {
			Vec3 preColor = hslColor + colorUpdate;
			preColor.x = modFloat(preColor.x, 360);
			preColor.y = clamp(preColor.y, 0, 1);
			preColor.z = clamp(preColor.z, 0, 1);
			hslColor = preColor;
		}

		Vec3 oldHslColor = hslColor;
		bool hue = gui->slider(&hslColor.x,0,360);
		bool sat = gui->slider(&hslColor.y,0,1);
		bool lig = gui->slider(&hslColor.z,0,1);

		if(main && (hue || sat || lig)) {
			mainUpdated = true;
			colorUpdate = hslColor - oldHslColor;
		}

		gui->colors.e[i].xyz = hslToRgb(hslColor);
		gui->colors.e[i].w = 1;
		gui->heightPop();

		gui->div(divWidths, arrayCount(divWidths)); gui->label("",0); gui->label("",0); 
		bool r = gui->slider(&gui->colors.e[i].x, 0,1); 
		bool g = gui->slider(&gui->colors.e[i].y, 0,1); 
		bool b = gui->slider(&gui->colors.e[i].z, 0,1);
	}

	gui->advanceY(1);

	char** ss = &guiSettingStrings[0];
	int i = 0;
	GuiSettings* gs = &gui->settings;
	gui->div(tw,0,0); gui->label(ss[i++],0); gui->slider(&gs->offsets.x, 0, 20); gui->slider(&gs->offsets.y, 0, 20);
	gui->div(tw,0,0); gui->label(ss[i++],0); gui->slider(&gs->border.x, 0, 20), gui->slider(&gs->border.y, -20, 0);
	gui->div(tw,0,0); gui->label(ss[i++],0); gui->slider(&gs->minSize.x, 1, 400), gui->slider(&gs->minSize.y, 1, 400);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->sliderSize, 		1, 20);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->sliderResetDistance,20, 300);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->fontOffset, 		0, 1);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->sectionOffset, 		0, 4);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->fontHeightOffset, 	0.5, 2);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->sectionIndent, 		0, 0.5f);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->scrollBarWidth, 	2, 30);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->scrollBarSliderSize,1, 50);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->cursorWidth, 		1, 10);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->textShadow, 		0, 10);
}





