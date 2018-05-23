
Font* fontInit(Font* fontSlot, char* file, float height, bool enableHinting = false) {
	char* fontFolder = 0;
	for(int i = 0; i < theGraphicsState->fontFolderCount; i++) {
		if(fileExists(fillString("%s%s", theGraphicsState->fontFolders[i], file))) {
			fontFolder = theGraphicsState->fontFolders[i];
			break;
		}
	}
	if(!fontFolder) return 0;

	char* path = fillString("%s%s", fontFolder, file);



	Font font;
	font.isSubPixel = false;

	// Settings.
	
	bool stemDarkening = true;
	bool pixelAlign = true;
	
	int target;
	if(height <= 14.0f)   target = FT_LOAD_TARGET_MONO | FT_LOAD_FORCE_AUTOHINT;
	else if(height <= 25) target = FT_LOAD_TARGET_NORMAL | FT_LOAD_FORCE_AUTOHINT;
	else                  target = FT_LOAD_TARGET_NORMAL;

	int loadFlags = FT_LOAD_DEFAULT | target;

	// FT_RENDER_MODE_NORMAL, FT_RENDER_MODE_LIGHT, FT_RENDER_MODE_MONO, FT_RENDER_MODE_LCD, FT_RENDER_MODE_LCD_V,
	FT_Render_Mode renderFlags = FT_RENDER_MODE_NORMAL;


	font.glyphRangeCount = 0;
	#define setupRange(a,b) vec2i(a, b - a + 1)
	font.glyphRanges[font.glyphRangeCount++] = setupRange(0x20, 0x7F);
	font.glyphRanges[font.glyphRangeCount++] = setupRange(0xA1, 0xFF);
	font.glyphRanges[font.glyphRangeCount++] = setupRange(0x25BA, 0x25C4);
	// font.glyphRanges[font.glyphRangeCount++] = setupRange(0x48, 0x49);
	#undef setupRange

	font.totalGlyphCount = 0;
	for(int i = 0; i < font.glyphRangeCount; i++) font.totalGlyphCount += font.glyphRanges[i].y;



	font.file = getPString(strLen(file)+1);
	strCpy(font.file, file);
	font.heightIndex = height;

	int error;
	error = FT_Init_FreeType(&font.library); assert(error == 0);
	error = FT_New_Face(font.library, path, 0, &font.face); assert(error == 0);
	FT_Face face = font.face;

	FT_Parameter parameter;
	FT_Bool darkenBool = stemDarkening;
	parameter.tag = FT_PARAM_TAG_STEM_DARKENING;
	parameter.data = &darkenBool;
	error = FT_Face_Properties(face, 1, &parameter); assert(error == 0);

	int pointFraction = 64;
	font.pixelScale = (float)1/pointFraction;
	float fullHeightToAscend = (float)face->ascender / (float)(face->ascender + abs(face->descender));

	// Height < 0 means use point size instead of pixel size
	if(height > 0) {
		error = FT_Set_Pixel_Sizes(font.face, 0, roundIntf(height) * fullHeightToAscend); assert(error == 0);
	} else {
		error = FT_Set_Char_Size(font.face, 0, (roundIntf(-height) * fullHeightToAscend) * pointFraction, 0, 0); assert(error == 0);
	}

	// Get true height from freetype.
	font.height = (face->size->metrics.ascender + abs(face->size->metrics.descender)) / pointFraction;
	font.baseOffset = (face->size->metrics.ascender / pointFraction);

	// We calculate the scaling ourselves because Freetype doesn't offer it??
	float scale = (float)face->size->metrics.ascender / (float)face->ascender;
	font.lineSpacing = roundIntf(((face->height * scale) / pointFraction));
	font.pixelAlign = pixelAlign;



	int gridSize = (sqrt(font.totalGlyphCount) + 1);
	Vec2i texSize = vec2i(gridSize * font.height);
	uchar* fontBitmapBuffer = mallocArray(unsigned char, texSize.x*texSize.y);
	memSet(fontBitmapBuffer, 0, texSize.x*texSize.y);

	{
		font.cData = mallocArray(PackedChar, font.totalGlyphCount);
		int glyphIndex = 0;
		for(int rangeIndex = 0; rangeIndex < font.glyphRangeCount; rangeIndex++) {
			for(int i = 0; i < font.glyphRanges[rangeIndex].y; i++) {
				int unicode = font.glyphRanges[rangeIndex].x + i;

				FT_Load_Char(face, unicode, loadFlags);
				FT_Render_Glyph(face->glyph, renderFlags);

				FT_Bitmap* bitmap = &face->glyph->bitmap;
				Vec2i coordinate = vec2i(glyphIndex%gridSize, glyphIndex/gridSize);
				Vec2i startPixel = coordinate * font.height;

				font.cData[glyphIndex].x0 = startPixel.x;
				font.cData[glyphIndex].x1 = startPixel.x + bitmap->width;
				font.cData[glyphIndex].y1 = startPixel.y + bitmap->rows;
				font.cData[glyphIndex].y0 = startPixel.y;

				font.cData[glyphIndex].xBearing = face->glyph->metrics.horiBearingX / pointFraction;
				font.cData[glyphIndex].yBearing = face->glyph->metrics.horiBearingY / pointFraction;
				font.cData[glyphIndex].width =    face->glyph->metrics.width        / pointFraction;
				font.cData[glyphIndex].height =   face->glyph->metrics.height       / pointFraction;

				font.cData[glyphIndex].xadvance = face->glyph->metrics.horiAdvance / pointFraction;

				for(int y = 0; y < bitmap->rows; y++) {
					for(int x = 0; x < bitmap->width; x++) {
						Vec2i coord = startPixel + vec2i(x,y);
						fontBitmapBuffer[coord.y*texSize.w + coord.x] = bitmap->buffer[y*bitmap->width + x];
					}
				}

				glyphIndex++;
			}
		}
	}


	Texture tex;
	uchar* fontBitmap = mallocArray(unsigned char, texSize.x*texSize.y*4);
	memSet(fontBitmap, 255, texSize.w*texSize.h*4);
	for(int i = 0; i < texSize.w*texSize.h; i++) fontBitmap[i*4+3] = fontBitmapBuffer[i];

	// loadTexture(&tex, fontBitmap, texSize.w, texSize.h, 1, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE);
	loadTexture(&tex, fontBitmap, texSize.w, texSize.h, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

	font.tex = tex;


	free(fontBitmapBuffer);
	free(fontBitmap);

	*fontSlot = font;
	return fontSlot;
}

#if 0
Font* fontInitSubpixel(Font* fontSlot, char* file, float height) {
	char* fontFolder = 0;
	for(int i = 0; i < theGraphicsState->fontFolderCount; i++) {
		if(fileExists(fillString("%s%s", theGraphicsState->fontFolders[i], file))) {
			fontFolder = theGraphicsState->fontFolders[i];
			break;
		}
	}
	if(!fontFolder) return 0;

	char* path = fillString("%s%s", fontFolder, file);



	Font font;
	font.isSubPixel = true;

	// Settings.
	
	bool stemDarkening = true;
	bool pixelAlign = true;
	
	int target;
	if(height <= 14.0f)   target = FT_LOAD_TARGET_MONO | FT_LOAD_FORCE_AUTOHINT;
	else if(height <= 25) target = FT_LOAD_TARGET_NORMAL | FT_LOAD_FORCE_AUTOHINT;
	else                  target = FT_LOAD_TARGET_NORMAL;

	target = FT_LOAD_TARGET_LCD;

	int loadFlags = FT_LOAD_DEFAULT | target;

	// FT_RENDER_MODE_NORMAL, FT_RENDER_MODE_LIGHT, FT_RENDER_MODE_MONO, FT_RENDER_MODE_LCD, FT_RENDER_MODE_LCD_V,
	// FT_Render_Mode renderFlags = FT_RENDER_MODE_NORMAL;
	FT_Render_Mode renderFlags = FT_RENDER_MODE_LCD;
	

	font.glyphRangeCount = 0;
	#define setupRange(a,b) vec2i(a, b - a + 1)
	font.glyphRanges[font.glyphRangeCount++] = setupRange(0x20, 0x7F);
	font.glyphRanges[font.glyphRangeCount++] = setupRange(0xA1, 0xFF);
	font.glyphRanges[font.glyphRangeCount++] = setupRange(0x25BA, 0x25C4);
	// font.glyphRanges[font.glyphRangeCount++] = setupRange(0x48, 0x49);
	#undef setupRange

	font.totalGlyphCount = 0;
	for(int i = 0; i < font.glyphRangeCount; i++) font.totalGlyphCount += font.glyphRanges[i].y;



	font.file = getPString(strLen(file)+1);
	strCpy(font.file, file);
	font.heightIndex = height;

	int error;
	error = FT_Init_FreeType(&font.library); assert(error == 0);
	error = FT_New_Face(font.library, path, 0, &font.face); assert(error == 0);
	FT_Face face = font.face;

	FT_Parameter parameter;
	FT_Bool darkenBool = stemDarkening;
	parameter.tag = FT_PARAM_TAG_STEM_DARKENING;
	parameter.data = &darkenBool;
	error = FT_Face_Properties(face, 1, &parameter); assert(error == 0);

	int pointFraction = 64;
	font.pixelScale = (float)1/pointFraction;
	float fullHeightToAscend = (float)face->ascender / (float)(face->ascender + abs(face->descender));

	// Height < 0 means use point size instead of pixel size
	if(height > 0) {
		error = FT_Set_Pixel_Sizes(font.face, 0, roundIntf(height) * fullHeightToAscend); assert(error == 0);
	} else {
		error = FT_Set_Char_Size(font.face, 0, (roundIntf(-height) * fullHeightToAscend) * pointFraction, 0, 0); assert(error == 0);
	}

	// Get true height from freetype.
	font.height = (face->size->metrics.ascender + abs(face->size->metrics.descender)) / pointFraction;
	font.baseOffset = (face->size->metrics.ascender / pointFraction);

	// We calculate the scaling ourselves because Freetype doesn't offer it??
	float scale = (float)face->size->metrics.ascender / (float)face->ascender;
	font.lineSpacing = roundIntf(((face->height * scale) / pointFraction));
	font.pixelAlign = pixelAlign;



	int gridSize = (sqrt(font.totalGlyphCount) + 1);
	Vec2i texSize = vec2i(gridSize * font.height);
	uchar* fontBitmapBuffer = (uchar*)malloc(sizeof(uchar) * 3 * texSize.x*texSize.y);
	memSet(fontBitmapBuffer, 0, texSize.x*texSize.y*3);

	{
		font.cData = mallocArray(PackedChar, font.totalGlyphCount);
		int glyphIndex = 0;
		for(int rangeIndex = 0; rangeIndex < font.glyphRangeCount; rangeIndex++) {
			for(int i = 0; i < font.glyphRanges[rangeIndex].y; i++) {
				int unicode = font.glyphRanges[rangeIndex].x + i;

				FT_Load_Char(face, unicode, loadFlags);
				FT_Render_Glyph(face->glyph, renderFlags);

				FT_Bitmap* bitmap = &face->glyph->bitmap;
				Vec2i coordinate = vec2i(glyphIndex%gridSize * 3, glyphIndex/gridSize);
				Vec2i startPixel = coordinate * font.height;

				font.cData[glyphIndex].x0 = startPixel.x/3;
				font.cData[glyphIndex].x1 = startPixel.x/3 + bitmap->width/3;
				font.cData[glyphIndex].y1 = startPixel.y + bitmap->rows;
				font.cData[glyphIndex].y0 = startPixel.y;

				font.cData[glyphIndex].xBearing = face->glyph->metrics.horiBearingX / pointFraction;
				font.cData[glyphIndex].yBearing = face->glyph->metrics.horiBearingY / pointFraction;
				font.cData[glyphIndex].width =    bitmap->width/3;
				font.cData[glyphIndex].height =   bitmap->rows;

				font.cData[glyphIndex].xadvance = face->glyph->metrics.horiAdvance / pointFraction;

				for(int y = 0; y < bitmap->rows; y++) {
					for(int x = 0; x < bitmap->width; x++) {
						Vec2i coord = startPixel + vec2i(x,y);
						fontBitmapBuffer[coord.y*texSize.w*3 + coord.x] = bitmap->buffer[y*bitmap->pitch + x];
					}
				}

				glyphIndex++;
			}
		}
	}


	Texture tex;
	uchar* fontBitmap = mallocArray(unsigned char, texSize.x*texSize.y*4);
	memSet(fontBitmap, 255, texSize.w*texSize.h*4);
	for(int i = 0; i < texSize.w*texSize.h; i++) {
		fontBitmap[(i*4)+0] = fontBitmapBuffer[i*3+0];
		fontBitmap[(i*4)+1] = fontBitmapBuffer[i*3+1];
		fontBitmap[(i*4)+2] = fontBitmapBuffer[i*3+2];
		fontBitmap[(i*4)+3] = 255;
	}

	// loadTexture(&tex, fontBitmap, texSize.w, texSize.h, 1, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE);
	loadTexture(&tex, fontBitmap, texSize.w, texSize.h, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

	font.tex = tex;

	// stbi_write_png("C:\\Projects\\VoxelGame\\test.png", texSize.x, texSize.y, 4, fontBitmap, 0);

	free(fontBitmapBuffer);
	free(fontBitmap);

	*fontSlot = font;
	return fontSlot;
}
#endif

void freeFont(Font* font) {
	freeZero(font->cData);
	// FT_Done_Face(font->face);
	// FT_Done_Library(font->library);
	glDeleteTextures(1, &font->tex.id);
	font->heightIndex = 0;
}

Font* getFont(char* fontFile, float heightIndex, char* boldFontFile = 0, char* italicFontFile = 0) {

	int fontCount = arrayCount(theGraphicsState->fonts);
	int fontSlotCount = arrayCount(theGraphicsState->fonts[0]);
	Font* fontSlot = 0;
	for(int i = 0; i < fontCount; i++) {
		if(theGraphicsState->fonts[i][0].heightIndex == 0) {
			fontSlot = &theGraphicsState->fonts[i][0];
			break;
		} else {
			if(strCompare(fontFile, theGraphicsState->fonts[i][0].file)) {
				for(int j = 0; j < fontSlotCount; j++) {
					float h = theGraphicsState->fonts[i][j].heightIndex;
					if(h == 0 || h == heightIndex) {
						fontSlot = &theGraphicsState->fonts[i][j];
						goto forEnd;
					}
				}
			}
		}
	}
	forEnd:

	// We are going to assume for now that a font size of 0 means it is uninitialized.
	if(fontSlot->heightIndex == 0) {
		Font* font = fontInit(fontSlot, fontFile, heightIndex);
		if(!font) {
			printf("Could not initialize font!\n");
			exit(0);
		}

		if(boldFontFile) {
			fontSlot->boldFont = getPStruct(Font);
			fontInit(fontSlot->boldFont, boldFontFile, heightIndex);
		} else font->boldFont = 0;

		if(italicFontFile) {
			fontSlot->italicFont = getPStruct(Font);
			fontInit(fontSlot->italicFont, italicFontFile, heightIndex);
		} else font->italicFont = 0;
	}

	return fontSlot;
}

//

inline char getRightBits(char n, int count) {
	int bitMask = 0;
	for(int i = 0; i < count; i++) bitMask += (1 << i);
	return n&bitMask;
}

int unicodeDecode(uchar* s, int* byteCount) {
	if(s[0] <= 127) {
		*byteCount = 1;
		return s[0];
	}

	int bitCount = 1;
	for(;;) {
		char bit = (1 << 8-bitCount-1);
		if(s[0]&bit) bitCount++;
		else break;
	}

	(*byteCount) = bitCount;

	int unicodeChar = 0;
	for(int i = 0; i < bitCount; i++) {
		char byte = i==0 ? getRightBits(s[i], 8-(bitCount+1)) : getRightBits(s[i], 6);

		unicodeChar += ((int)byte) << (6*((bitCount-1)-i));
	}

	return unicodeChar;
}

int unicodeGetSize(uchar* s) {
	if(s[0] <= 127) return 1;

	int bitCount = 1;
	for(;;) {
		char bit = (1 << 8-bitCount-1);
		if(s[0]&bit) bitCount++;
		else break;
	}

	return bitCount;
}

int getUnicodeRangeOffset(int c, Font* font) {
	int unicodeOffset = -1;

	bool found = false;
	for(int i = 0; i < font->glyphRangeCount; i++) {
		if(between(c, font->glyphRanges[i].x, font->glyphRanges[i].x+font->glyphRanges[i].y)) {
			unicodeOffset += c - font->glyphRanges[i].x + 1;
			found = true;
			break;
		}
		unicodeOffset += font->glyphRanges[i].y;
	}

	if(!found) {
		if(c == Font_Error_Glyph) return 0;
		unicodeOffset = getUnicodeRangeOffset(Font_Error_Glyph, font);
	}

	return unicodeOffset;
}

// Taken from stbtt_truetype.
void getPackedQuad(PackedChar *chardata, Vec2i texDim, int char_index, Vec2 pos, Rect* r, Rect* uv, int alignToInteger)
{
   PackedChar *b = chardata + char_index;

   if (alignToInteger) {
   	  // *r = rectBLDim(roundf(pos.x + b->xBearing), roundf(pos.y - (b->height - b->yBearing)), b->width, b->height);

   	  (*r).left = roundf(pos.x + b->xBearing);
   	  (*r).bottom = roundf(pos.y - (b->height - b->yBearing));
   	  (*r).right = (*r).left + b->width;
   	  (*r).top = (*r).bottom + b->height;

   } else {
   	  // *r = rectBLDim(pos.x + b->xBearing, pos.y - (b->height - b->yBearing), b->width, b->height);

   	  (*r).left = pos.x + b->xBearing;
   	  (*r).bottom = pos.y - (b->height - b->yBearing);
   	  (*r).right = (*r).left + b->width;
   	  (*r).top = (*r).bottom + b->height;
   }

   Vec2 ip = vec2(1.0f / texDim.w, 1.0f / texDim.h);
   *uv = rect(b->x0*ip.x, b->y0*ip.y, b->x1*ip.x, b->y1*ip.y);
}

void getTextQuad(int c, Font* font, Vec2 pos, Rect* r, Rect* uv) {

	int unicodeOffset = getUnicodeRangeOffset(c, font);
	getPackedQuad(font->cData, font->tex.dim, unicodeOffset, pos, r, uv, font->pixelAlign);

	float off = font->baseOffset;
	if(font->pixelAlign) off = roundIntf(off);
	r->bottom -= off;
	r->top -= off;
}

float getCharAdvance(int c, Font* font) {
	int unicodeOffset = getUnicodeRangeOffset(c, font);
	float result = font->cData[unicodeOffset].xadvance;
	return result;
}

float getCharAdvance(int c, int c2, Font* font) {
	int unicodeOffset = getUnicodeRangeOffset(c, font);
	float result = font->cData[unicodeOffset].xadvance;

	if(FT_HAS_KERNING(font->face)) {
		FT_Vector kerning;

		uint charIndex1 = FT_Get_Char_Index(font->face, c);
		uint charIndex2 = FT_Get_Char_Index(font->face, c2);

		FT_Get_Kerning(font->face, charIndex1, charIndex2, FT_KERNING_DEFAULT, &kerning);
		float kernAdvance = kerning.x * font->pixelScale;

		result += kernAdvance;
	}

	return result;
}

TextSimInfo initTextSimInfo(Vec2 startPos) {
	TextSimInfo tsi = {};
	tsi.pos = startPos;
	tsi.index = 0;
	tsi.wrapIndex = 0;
	tsi.lineBreak = false;
	tsi.breakPos = vec2(0,0);
	return tsi;
}

// The marker system is badly designed and has all kinds of edge cases where it breaks if you
// don't pay attention, so... put markers in sparingly.

int parseTextMarkers(char* text, TextSimInfo* tsi, int* type = 0) {
	// Return how many characters to skip.

	if(text[0] == '<') {
		if(text[1] != '\0' && text[2] != '\0' && text[2] == '>') {
			switch(text[1]) {
				case 'b': if(type) { *type = TEXT_MARKER_BOLD; } return Marker_Size;
				case 'i': if(type) { *type = TEXT_MARKER_ITALIC; } return Marker_Size;
				case 'c': if(type) { *type = TEXT_MARKER_COLOR; } 
					if(tsi->colorMode) return Marker_Size;
					else return Marker_Size + 6; // FFFFFF
			}
		}
	}

	return 0;
}

void updateMarkers(char* text, TextSimInfo* tsi, Font* font, bool skip = false) {

	int type;
	int length = 0;
	while(length = parseTextMarkers(text + tsi->index, tsi, &type)) {
		switch(type) {
			case TEXT_MARKER_BOLD: {
				tsi->index += length;
				tsi->wrapIndex += length;

				if(!font->boldFont) return;
				if(!skip) {
					if(!tsi->bold) {
						glEnd();
						glBindTexture(GL_TEXTURE_2D, font->boldFont->tex.id);
						glBegin(GL_QUADS);
					} else {
						glEnd();
						glBindTexture(GL_TEXTURE_2D, font->tex.id);
						glBegin(GL_QUADS);
					}
				}

				tsi->bold = !tsi->bold;
			} break;

			case TEXT_MARKER_ITALIC: {
				tsi->index += length;
				tsi->wrapIndex += length;

				if(!font->italicFont) return;
				if(!skip) {
					if(!tsi->italic && font->italicFont) {
						glEnd();
						glBindTexture(GL_TEXTURE_2D, font->italicFont->tex.id);
						glBegin(GL_QUADS);
					} else {
						glEnd();
						glBindTexture(GL_TEXTURE_2D, font->tex.id);
						glBegin(GL_QUADS);
					}
				}

				tsi->italic = !tsi->italic;
			} break;

			case TEXT_MARKER_COLOR: {
				if(skip) {
					tsi->index += length;
					tsi->wrapIndex += length;
					if(type == TEXT_MARKER_COLOR) tsi->colorMode = !tsi->colorMode;
					continue;
				} 
				tsi->index += Marker_Size;
				tsi->wrapIndex += Marker_Size;
				Vec3 c;
				if(!tsi->colorMode) {
					c.r = colorIntToFloat(strHexToInt(getTStringCpy(&text[tsi->index], 2))); tsi->index += 2;
					c.g = colorIntToFloat(strHexToInt(getTStringCpy(&text[tsi->index], 2))); tsi->index += 2;
					c.b = colorIntToFloat(strHexToInt(getTStringCpy(&text[tsi->index], 2))); tsi->index += 2;
					tsi->colorOverwrite = COLOR_SRGB(c);
					
					tsi->wrapIndex += 6;
				}

				tsi->colorMode = !tsi->colorMode;
			} break;
		}
	}
}

int textSim(char* text, Font* font, TextSimInfo* tsi, TextInfo* ti, Vec2 startPos = vec2(0,0), int wrapWidth = 0) {
	ti->lineBreak = false;

	if(tsi->lineBreak) {
		ti->lineBreak = true;
		ti->breakPos = tsi->breakPos;
		tsi->lineBreak = false;
	}

	if(text[tsi->index] == '\0') {
		ti->pos = tsi->pos;
		ti->index = tsi->index;
		return 0;
	}

	Vec2 oldPos = tsi->pos;

	int i = tsi->index;
	int tSize;
	int t = unicodeDecode((uchar*)(&text[i]), &tSize);

	bool wrapped = false;

	if(wrapWidth != 0 && i == tsi->wrapIndex) {
		int size;
		int c = unicodeDecode((uchar*)(&text[i]), &size);
		float wordWidth = 0;
		if(c == '\n') wordWidth = getCharAdvance(c, font);

		char* tempText = text;
		int it = i;
		while(c != '\n' && c != '\0' && c != ' ') {

			// Awkward.
			bool hadMarker = false;
			int markerLength = 0;
			while(markerLength = parseTextMarkers(tempText + it, tsi)) {
				// Pretend markers aren't there by moving text pointer.
				tempText += markerLength;
				hadMarker = true;
			}
			if(hadMarker) {
				c = unicodeDecode((uchar*)(&tempText[it]), &size);
				continue;
			}

			wordWidth += getCharAdvance(c, font);
			it += size;
			c = unicodeDecode((uchar*)(&tempText[it]), &size);
		}

		if(tsi->pos.x + wordWidth > startPos.x + wrapWidth) {
			wrapped = true;
		}

		if(it != i) tsi->wrapIndex = it;
		else tsi->wrapIndex++;
	}

	if(t == '\n' || wrapped) {
		tsi->lineBreak = true;
		if(t == '\n') tsi->breakPos = tsi->pos + vec2(getCharAdvance(t, font),0);
		if(wrapped) tsi->breakPos = tsi->pos;

		tsi->pos.x = startPos.x;
		// tsi->pos.y -= font->height;
		tsi->pos.y -= font->lineSpacing;

		if(wrapped) {
			return textSim(text, font, tsi, ti, startPos, wrapWidth);
		}
	} else {
		getTextQuad(t, font, tsi->pos, &ti->r, &ti->uv);

		if(text[i+1] != '\0') {
			int tSize2;
			int t2 = unicodeDecode((uchar*)(&text[i+tSize]), &tSize2);
			tsi->pos.x += getCharAdvance(t, t2, font);
		} else tsi->pos.x += getCharAdvance(t, font);
	}

	if(ti) {
		ti->pos = oldPos;
		ti->index = tsi->index;
		ti->posAdvance = tsi->pos - oldPos;
	}

	tsi->index += tSize;

	return 1;
}


TextSettings textSettings(Font* font, Vec4 color, int shadowMode, Vec2 shadowDir, float shadowSize, Vec4 shadowColor) {
	return {font, color, shadowMode, shadowDir, shadowSize, shadowColor};
}
TextSettings textSettings(Font* font, Vec4 color, int shadowMode, float shadowSize, Vec4 shadowColor) {
	return {font, color, shadowMode, vec2(-1,-1), shadowSize, shadowColor};
}
TextSettings textSettings(Font* font, Vec4 color) {
	return {font, color};
}

Vec2 getTextDim(char* text, Font* font, Vec2 startPos = vec2(0,0), int wrapWidth = 0) {
	float maxX = startPos.x;

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		Font* f = font;
		updateMarkers(text, &tsi, font, true);
		if(tsi.bold) f = font->boldFont;
		else if(tsi.italic) f = font->italicFont;

		TextInfo ti;
		if(!textSim(text, f, &tsi, &ti, startPos, wrapWidth)) break;

		maxX = max(maxX, ti.pos.x + ti.posAdvance.x);
	}

	Vec2 dim = vec2(maxX - startPos.x, startPos.y - (tsi.pos.y - font->height));

	return dim;
}

Vec2 testgetTextStartPos(char* text, Font* font, Vec2 startPos, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	Vec2 dim = getTextDim(text, font, startPos, wrapWidth);
	startPos.x -= (align.x+1)*0.5f*dim.w;
	startPos.y -= (align.y-1)*0.5f*dim.h;

	return startPos;
}

Rect getTextLineRect(char* text, Font* font, Vec2 startPos, Vec2i align = vec2i(-1,1)) {
	startPos = testgetTextStartPos(text, font, startPos, align, 0);

	Vec2 textDim = getTextDim(text, font);
	Rect r = rectTLDim(startPos, textDim);

	return r;
}

void drawText(char* text, Vec2 startPos, Vec2i align, int wrapWidth, TextSettings settings) {
	float z = theGraphicsState->zOrder;
	Font* font = settings.font;

	int cullWidth = wrapWidth;
	if(settings.cull) wrapWidth = 0;

	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	// if(!settings.srgb) setSRGB(false);

	Vec4 c = COLOR_SRGB(settings.color);
	Vec4 sc = COLOR_SRGB(settings.shadowColor);

	// pushColor(c);

	int texId = font->tex.id;
	// glBindTexture(GL_TEXTURE_2D, texId);
	// glBegin(GL_QUADS);

	// if(font->isSubPixel) bindShader(SHADER_FONT);

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		
		Font* f = font;
		updateMarkers(text, &tsi, font);
		if(tsi.bold) f = font->boldFont;
		else if(tsi.italic) f = font->italicFont;

		TextInfo ti;
		if(!textSim(text, f, &tsi, &ti, startPos, wrapWidth)) break;
		if(text[ti.index] == '\n') continue;

		if(settings.cull && (ti.pos.x > startPos.x + cullWidth)) break;

		if(settings.shadowMode != TEXTSHADOW_MODE_NOSHADOW) {
			// pushColor(sc);

			if(settings.shadowMode == TEXTSHADOW_MODE_SHADOW) {
				Vec2 p = ti.r.min + norm(settings.shadowDir) * settings.shadowSize;
				Rect sr = rectBLDim(vec2(roundf(p.x), roundf(p.y)), rectDim(ti.r));

				// pushRect(sr, ti.uv, z);
				drawRect(sr, sc, ti.uv, texId);

			} else if(settings.shadowMode == TEXTSHADOW_MODE_OUTLINE) {
				for(int i = 0; i < 8; i++) {
					
					// Not sure if we should align to pixels on an outline.

					Vec2 dir = rotateVec2(vec2(1,0), (M_2PI/8)*i);
					Rect r = rectTrans(ti.r, dir*settings.shadowSize);
					// pushRect(r, ti.uv, z);
					drawRect(r, sc, ti.uv, texId);


					// Vec2 dir = rotateVec2(vec2(1,0), (M_2PI/8)*i);
					// Vec2 p = ti.r.min + dir * settings.shadowSize;
					// Rect sr = rectBLDim(vec2(roundf(p.x), roundf(p.y)), rectDim(ti.r));
					// pushRect(sr, ti.uv, z);
				}
			}
		}

		// if(tsi.colorMode) pushColor(vec4(tsi.colorOverwrite, 1));
		// else pushColor(c);

		// pushRect(ti.r, ti.uv, z);

		if(!font->isSubPixel)
			drawRect(ti.r, c, ti.uv, texId);
		else 
			drawFont(ti.r, c, ti.uv, texId);

	}

	if(font->isSubPixel)
		bindShader(SHADER_QUAD);
	
	// glEnd();

	// if(!settings.srgb) setSRGB();
}
void drawText(char* text, Vec2 startPos, TextSettings settings) {
	return drawText(text, startPos, vec2i(-1,1), 0, settings);
}
void drawText(char* text, Vec2 startPos, Vec2i align, TextSettings settings) {
	return drawText(text, startPos, align, 0, settings);
}

// // @CodeDuplication.
// void drawTextLineCulled(char* text, Vec2 startPos, Vec2i align, int width, TextSettings settings) {
// 	float z = theGraphicsState->zOrder;
// 	Font* font = settings.font;

// 	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

// 	Vec4 c = COLOR_SRGB(settings.color);
// 	Vec4 sc = COLOR_SRGB(settings.shadowColor);

// 	TextSimInfo tsi = initTextSimInfo(startPos);
// 	while(true) {
		
// 		Font* f = font;
// 		updateMarkers(text, &tsi, font);
// 		if(tsi.bold) f = font->boldFont;
// 		else if(tsi.italic) f = font->italicFont;

// 		TextInfo ti;
// 		if(!textSim(text, f, &tsi, &ti, startPos, wrapWidth)) break;
// 		if(text[ti.index] == '\n') continue;

// 		if(ti.pos.x > startPos.x + width) break;

// 		if(settings.shadowMode != TEXTSHADOW_MODE_NOSHADOW) {

// 			if(settings.shadowMode == TEXTSHADOW_MODE_SHADOW) {
// 				Vec2 p = ti.r.min + norm(settings.shadowDir) * settings.shadowSize;
// 				Rect sr = rectBLDim(vec2(roundf(p.x), roundf(p.y)), rectDim(ti.r));

// 				drawRect(sr, sc, ti.uv, font->tex.id);

// 			} else if(settings.shadowMode == TEXTSHADOW_MODE_OUTLINE) {
// 				for(int i = 0; i < 8; i++) {
					
// 					Vec2 dir = rotateVec2(vec2(1,0), (M_2PI/8)*i);
// 					Rect r = rectTrans(ti.r, dir*settings.shadowSize);
// 					drawRect(r, sc, ti.uv, font->tex.id);
// 				}
// 			}
// 		}

// 		drawRect(ti.r, c, ti.uv, font->tex.id);
// 	}
// }

// void drawTextLineCulled(char* text, Font* font, Vec2 startPos, float width, Vec4 color, Vec2i align = vec2i(-1,1)) {
// 	startPos = testgetTextStartPos(text, font, startPos, align, 0);
// 	startPos = vec2(roundInt((int)startPos.x), roundInt((int)startPos.y));

// 	TextSimInfo tsi = initTextSimInfo(startPos);
// 	while(true) {
// 		Font* f = font;
// 		updateMarkers(text, &tsi, font, true);
// 		if(tsi.bold) f = font->boldFont;
// 		else if(tsi.italic) f = font->italicFont;

// 		TextInfo ti;
// 		if(!textSim(text, f, &tsi, &ti, startPos, 0)) break;
// 		if(text[ti.index] == '\n') continue;

// 		if(ti.pos.x > startPos.x + width) break;

// 		drawRect(ti.r, color, ti.uv, f->tex.id);
// 	}
// }

Vec2 textIndexToPos(char* text, Font* font, Vec2 startPos, int index, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		Font* f = font;
		updateMarkers(text, &tsi, font, true);
		if(tsi.bold) f = font->boldFont;
		else if(tsi.italic) f = font->italicFont;

		TextInfo ti;
		int result = textSim(text, f, &tsi, &ti, startPos, wrapWidth);

		if(ti.index == index) {
			Vec2 pos = ti.pos - vec2(0, f->height/2);
			return pos;
		}

		if(!result) break;
	}

	return vec2(0,0);
}

void drawTextSelection(char* text, Font* font, Vec2 startPos, int index1, int index2, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	if(index1 == index2) return;
	if(index1 > index2) swap(&index1, &index2);

	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	Vec2 lineStart;
	bool drawSelection = false;

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		Font* f = font;
		updateMarkers(text, &tsi, font, true);
		if(tsi.bold) f = font->boldFont;
		else if(tsi.italic) f = font->italicFont;

		TextInfo ti;
		int result = textSim(text, f, &tsi, &ti, startPos, wrapWidth);

		bool endReached = ti.index == index2;

		if(drawSelection) {
			if(ti.lineBreak || endReached) {

				Vec2 lineEnd;
				if(ti.lineBreak) lineEnd = ti.breakPos;
				else if(!result) lineEnd = tsi.pos;
				else lineEnd = ti.pos;

				Rect r = rect(lineStart - vec2(0,f->height), lineEnd);
				drawRect(r, color);

				lineStart = ti.pos;

				if(endReached) break;
			}
		}

		if(!drawSelection && (ti.index >= index1)) {
			drawSelection = true;
			lineStart = ti.pos;
		}

		if(!result) break;
	}
}

int textMouseToIndex(char* text, Font* font, Vec2 startPos, Vec2 mousePos, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	if(mousePos.y > startPos.y) return 0;
	
	bool foundLine = false;
	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		Font* f = font;
		updateMarkers(text, &tsi, font, true);
		if(tsi.bold) f = font->boldFont;
		else if(tsi.italic) f = font->italicFont;

		TextInfo ti;
		int result = textSim(text, f, &tsi, &ti, startPos, wrapWidth);
		
		bool fLine = between(mousePos.y, ti.pos.y - f->height, ti.pos.y);
		if(fLine) foundLine = true;
		else if(foundLine) return ti.index-1;

	    if(foundLine) {
	    	float charMid = ti.pos.x + ti.posAdvance.x*0.5f;
			if(mousePos.x < charMid) return ti.index;
		}

		if(!result) break;
	}

	return tsi.index;
}

// char* textSelectionToString(char* text, int index1, int index2) {
// 	myAssert(index1 >= 0 && index2 >= 0);

// 	int range = abs(index1 - index2);
// 	char* str = getTStringDebug(range + 1); // We assume text selection will only be used for debug things.
// 	strCpy(str, text + minInt(index1, index2), range);
// 	return str;
// }

char* textSelectionToString(char* text, int index1, int index2) {
	assert(index1 >= 0 && index2 >= 0);

	int range = abs(index1 - index2);
	char* str = getTStringDebug(range + 1); // We assume text selection will only be used for debug things.
	strCpy(str, text + min(index1, index2), range);
	return str;
}
