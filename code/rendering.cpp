
struct DrawCommandList;
extern DrawCommandList* globalCommandList;
struct GraphicsState;
extern GraphicsState* theGraphicsState;

// 
// Misc.
// 

char* fillString(char* text, ...) {
	va_list vl;
	va_start(vl, text);

	int length = strLen(text);
	char* buffer = getTStringX(length+1);

	char valueBuffer[20] = {};

	int ti = 0;
	int bi = 0;
	while(true) {
		char t = text[ti];

		if(text[ti] == '%' && text[ti+1] == '.') {
			float v = va_arg(vl, double);
			floatToStr(valueBuffer, v, charToInt(text[ti+2]));
			int sLen = strLen(valueBuffer);
			memCpy(buffer + bi, valueBuffer, sLen);

			ti += 4;
			bi += sLen;
			getTStringX(sLen);
		} else if(text[ti] == '%' && text[ti+1] == 'f') {
			float v = va_arg(vl, double);
			floatToStr(valueBuffer, v, 2);
			int sLen = strLen(valueBuffer);
			memCpy(buffer + bi, valueBuffer, sLen);

			ti += 2;
			bi += sLen;
			getTStringX(sLen);
		} else if(text[ti] == '%' && text[ti+1] == 'i') {
			if(text[ti+2] == '6') {
				// 64 bit signed integer.

				assert(text[ti+3] == '4');

				i64 v = va_arg(vl, i64);
				intToStr(valueBuffer, v);
				int sLen = strLen(valueBuffer);

				if(text[ti+4] == '.') {
					ti += 1;

					int digitCount = intDigits(v);
					int commaCount = digitCount/3;
					if(digitCount%3 == 0) commaCount--;
					for(int i = 0; i < commaCount; i++) {
						strInsert(valueBuffer, sLen - (i+1)*3 - i, ',');
						sLen++;
					}
				}

				memCpy(buffer + bi, valueBuffer, sLen);
				ti += 4;
				bi += sLen;
				getTStringX(sLen);
			} else {
				// 32 bit signed integer.
				int v = va_arg(vl, int);
				intToStr(valueBuffer, v);
				int sLen = strLen(valueBuffer);

				if(text[ti+2] == '.') {
					ti += 1;

					int digitCount = intDigits(v);
					int commaCount = digitCount/3;
					if(digitCount%3 == 0) commaCount--;
					for(int i = 0; i < commaCount; i++) {
						strInsert(valueBuffer, sLen - (i+1)*3 - i, ',');
						sLen++;
					}
				}

				memCpy(buffer + bi, valueBuffer, sLen);

				ti += 2;
				bi += sLen;
				getTStringX(sLen);
			}
		} else if(text[ti] == '%' && text[ti+1] == 's') {
			char* str = va_arg(vl, char*);
			int sLen = strLen(str);
			memCpy(buffer + bi, str, sLen);

			ti += 2;
			bi += sLen;
			getTStringX(sLen);
		} else if(text[ti] == '%' && text[ti+1] == 'b') {
			bool str = va_arg(vl, bool);
			char* s = str == 1 ? "true" : "false";
			int sLen = strLen(s);
			memCpy(buffer + bi, s, sLen);

			ti += 2;
			bi += sLen;
			getTStringX(sLen);
		} else if(text[ti] == '%' && text[ti+1] == '%') {
			buffer[bi++] = '%';
			ti += 2;
			getTStringX(1);
		} else {
			buffer[bi++] = text[ti++];
			getTStringX(1);

			if(buffer[bi-1] == '\0') break;
		}
	}

	return buffer;
}

//
// CommandList.
//

enum CommandState {
	STATE_SCISSOR,
	STATE_POLYGONMODE, 
	STATE_POLYGON_OFFSET, 
	STATE_LINEWIDTH,
	STATE_CULL,
	STATE_DEPTH_TEST,
	STATE_DEPTH_FUNC,
};

enum Polygon_Mode {
	POLYGON_MODE_FILL = 0,
	POLYGON_MODE_LINE,
	POLYGON_MODE_POINT,
};

enum DrawListCommand {
	Draw_Command_State_Type,
	Draw_Command_Enable_Type,
	Draw_Command_Disable_Type,
	Draw_Command_Cube_Type,
	Draw_Command_Line_Type,
	Draw_Command_Line2d_Type,
	Draw_Command_Quad_Type,
	Draw_Command_Quad2d_Type,
	Draw_Command_Rect_Type,
	Draw_Command_RoundedRect_Type,
	Draw_Command_Text_Type,
	Draw_Command_Scissor_Type,
	Draw_Command_Blend_Type,
	Draw_Command_PolygonOffset_Type,
};

struct DrawCommandList {
	void* data;
	int count;
	int bytes;
	int maxBytes;
};

void drawCommandListInit(DrawCommandList* cl, char* data, int maxBytes) {
	cl->data = data;
	cl->count = 0;
	cl->bytes = 0;
	cl->maxBytes = maxBytes;
}

#pragma pack(push,1)
struct Draw_Command_Cube {
	Vec3 trans;
	Vec3 scale;
	Vec4 color;
	float degrees;
	Vec3 rot;
};

struct Draw_Command_Line {
	Vec3 p0, p1;
	Vec4 color;
};

struct Draw_Command_Line2d {
	Vec2 p0, p1;
	Vec4 color;
};

struct Draw_Command_Quad {
	Vec3 p0, p1, p2, p3;
	Vec4 color;
	int textureId;
	Rect uv;
	int texZ;
};

struct Draw_Command_Quad2d {
	Vec2 p0, p1, p2, p3;
	Vec4 color;
	int textureId;
	Rect uv;
	int texZ;
};

struct Draw_Command_Rect {
	Rect r, uv;
	Vec4 color;
	int texture;
	int texZ;
};

struct Draw_Command_RoundedRect {
	Rect r;
	Vec4 color;

	float steps;
	float size;
};

struct Font;
struct Draw_Command_Text {
	char* text;
	Font* font;
	Vec2 pos;
	Vec4 color;

	int vAlign;
	int hAlign;
	int shadow;
	Vec4 shadowColor;

	int wrapWidth;
	float cullWidth;
};

struct Draw_Command_Scissor {
	Rect rect;
};

struct Draw_Command_Int {
	int state;
};

struct Draw_Command_State {
	int state;
	int value;
};

struct Draw_Command_Blend {
	int sourceColor;
	int destinationColor;
	int sourceAlpha;
	int destinationAlpha;
	int functionColor;
	int functionAlpha;
};

struct Draw_Command_PolygonOffset {
	float factor;
	float units;
};
#pragma pack(pop)


#define PUSH_DRAW_COMMAND(commandType, structType) \
	if(!drawList) drawList = globalCommandList; \
	char* list = (char*)drawList->data + drawList->bytes; \
	*((int*)list) = Draw_Command_##commandType##_Type; \
	list += sizeof(int); \
	Draw_Command_##structType* command = (Draw_Command_##structType*)list; \
	drawList->count++; \
	drawList->bytes += sizeof(Draw_Command_##structType) + sizeof(int); \
	assert(sizeof(Draw_Command_##structType) + drawList->bytes < drawList->maxBytes);


void dcCube(Vec3 trans, Vec3 scale, Vec4 color, float degrees = 0, Vec3 rot = vec3(0,0,0), DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Cube, Cube);

	command->trans = trans;
	command->scale = scale;
	command->color = color;
	command->degrees = degrees;
	command->rot = rot;
}

void dcLine(Vec3 p0, Vec3 p1, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Line, Line);

	command->p0 = p0;
	command->p1 = p1;
	command->color = color;
}

void dcLine2d(Vec2 p0, Vec2 p1, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Line2d, Line2d);

	command->p0 = p0;
	command->p1 = p1;
	command->color = color;
}

void dcQuad2d(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Vec4 color, int textureId = 0, Rect uv = rect(0,0,1,1), int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Quad2d, Quad2d);

	command->p0 = p0;
	command->p1 = p1;
	command->p2 = p2;
	command->p3 = p3;
	command->color = color;
	command->textureId = textureId;
	command->uv = uv;
	command->texZ = texZ;
}

void dcQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, int textureId = 0, Rect uv = rect(0,0,1,1), int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Quad, Quad);

	command->p0 = p0;
	command->p1 = p1;
	command->p2 = p2;
	command->p3 = p3;
	command->color = color;
	command->textureId = textureId;
	command->uv = uv;
	command->texZ = texZ;
}

void dcRect(Rect r, Rect uv, Vec4 color, int texture = -1, int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Rect, Rect);

	command->r = r;
	command->uv = uv;
	command->color = color;
	command->texture = texture;
	command->texZ = texZ;
}
void dcRect(Rect r, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Rect, Rect);

	command->r = r;
	command->uv = rect(0,0,1,1);
	command->color = color;
	command->texture = -1;
	command->texZ = -1;
}

void dcRoundedRect(Rect r, Vec4 color, float size, float steps = 0, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(RoundedRect, RoundedRect);

	command->r = r;
	command->color = color;
	command->steps = steps;
	command->size = size;
}

void dcText(char* text, Font* font, Vec2 pos, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1), DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Text, Text);

	command->text = text;
	command->font = font;
	command->pos = pos;
	command->color = color;
	command->vAlign = align.x;
	command->hAlign = align.y;
	command->shadow = shadow;
	command->shadowColor = shadowColor;
	command->wrapWidth = wrapWidth;
	command->cullWidth = -1;
}

void dcTextLine(char* text, Font* font, Vec2 pos, Vec4 color, Vec2i align = vec2i(-1,1), int cullWidth = -1, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1), DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Text, Text);

	command->text = text;
	command->font = font;
	command->pos = pos;
	command->color = color;
	command->vAlign = align.x;
	command->hAlign = align.y;
	command->shadow = shadow;
	command->shadowColor = shadowColor;
	command->wrapWidth = 0;
	command->cullWidth = cullWidth;
}

void dcScissor(Rect rect, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Scissor, Scissor);

	command->rect = rect;
}

void dcState(int state, int value, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(State, State);

	command->state = state;
	command->value = value;
}

void dcEnable(int state, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Enable, Int);

	command->state = state;
}

void dcDisable(int state, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Disable, Int);

	command->state = state;
}

void dcBlend(int sourceColor, int destinationColor, int sourceAlpha, int destinationAlpha, int functionColor, int functionAlpha, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Blend, Blend);

	command->sourceColor = sourceColor;
	command->destinationColor = destinationColor;
	command->sourceAlpha = sourceAlpha;
	command->destinationAlpha = destinationAlpha;

	command->functionColor = functionColor;
	command->functionAlpha = functionAlpha;
}

void dcBlend(int source, int destination, int function, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Blend, Blend);

	command->sourceColor = source;
	command->destinationColor = destination;
	command->sourceAlpha = source;
	command->destinationAlpha = destination;

	command->functionColor = function;
	command->functionAlpha = function;
}

void dcPolygonOffset(float factor, float units, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(PolygonOffset, PolygonOffset);

	command->factor = factor;
	command->units = units;
}

//
// Shaders.
//

struct ShaderUniform {
	int type;
	int vertexLocation;
	int fragmentLocation;
};

struct Shader {
	uint program;
	uint vertex;
	uint fragment;
	int uniformCount;
	ShaderUniform* uniforms;
};

//
// Textures.
// 

struct Texture {
	// char* name;
	uint id;
	Vec2i dim;
	int channels;
	int levels;
	int internalFormat;
	int channelType;
	int channelFormat;

	bool isRenderBuffer;
	int msaa;
};

int getMaximumMipmapsFromSize(int size) {
	int mipLevels = 1;
	while(size >= 2) {
		size /= 2;
		mipLevels++;
	}

	return mipLevels;
}

void loadTexture(Texture* texture, unsigned char* buffer, int w, int h, int mipLevels, int internalFormat, int channelType, int channelFormat, bool reload = false) {

	if(!reload) {
		glCreateTextures(GL_TEXTURE_2D, 1, &texture->id);
		glTextureStorage2D(texture->id, mipLevels, internalFormat, w, h);

		texture->dim = vec2i(w,h);
		texture->channels = 4;
		texture->levels = mipLevels;
	}	

	glTextureSubImage2D(texture->id, 0, 0, 0, w, h, channelType, channelFormat, buffer);
	glGenerateTextureMipmap(texture->id);
}

void loadTextureFromFile(Texture* texture, char* path, int mipLevels, int internalFormat, int channelType, int channelFormat, bool reload = false) {
	int x,y,n;
	unsigned char* stbData = stbi_load(path, &x, &y, &n, 0);

	if(mipLevels == -1) mipLevels = getMaximumMipmapsFromSize(min(x,y));
	
	loadTexture(texture, stbData, x, y, mipLevels, internalFormat, channelType, channelFormat, reload);

	stbi_image_free(stbData);
}

void loadCubeMapFromFile(Texture* texture, char* filePath, int mipLevels, int internalFormat, int channelType, int channelFormat, bool reload = false) {
	int texWidth, texHeight, n;
	uint* stbData = (uint*)stbi_load(filePath, &texWidth, &texHeight, &n, 4);

	int skySize = texWidth/(float)4;

	if(!reload) {
		texture->dim = vec2i(skySize, skySize);
		texture->channels = 4;
		texture->levels = 6;

		glCreateTextures(GL_TEXTURE_CUBE_MAP_ARRAY, CUBEMAP_SIZE, &texture->id);
		glTextureStorage3D(texture->id, mipLevels, internalFormat, skySize, skySize, 6);
	}

	uint* skyTex = getTArray(uint, skySize*skySize);
	Vec2i texOffsets[] = {{2,1}, {0,1}, {1,0}, {1,2}, {1,1}, {3,1}};
	for(int i = 0; i < 6; i++) {
		Vec2i offset = texOffsets[i] * skySize;

		for(int x = 0; x < skySize; x++) {
			for(int y = 0; y < skySize; y++) {
				skyTex[y*skySize + x] = stbData[(offset.y+y)*texWidth + (offset.x+x)];
			}
		}

		glTextureSubImage3D(texture->id, 0, 0, 0, i, skySize, skySize, 1, channelType, channelFormat, skyTex);
	}
	glTextureParameteri(texture->id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureParameteri(texture->id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(texture->id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(texture->id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureParameteri(texture->id, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);  

	glGenerateTextureMipmap(texture->id);

	stbi_image_free(stbData);
}

void createTexture(Texture* texture, bool isRenderBuffer = false) {	
	if(!isRenderBuffer) glCreateTextures(GL_TEXTURE_2D, 1, &texture->id);
	else glCreateRenderbuffers(1, &texture->id);
}

void recreateTexture(Texture* t) {
	glDeleteTextures(1, &t->id);
	glCreateTextures(GL_TEXTURE_2D, 1, &t->id);

	glTextureStorage2D(t->id, 1, t->internalFormat, t->dim.w, t->dim.h);
}

//
// Fonts.
//

struct PackedChar {
   unsigned short x0,y0,x1,y1;
   float xBearing, yBearing;
   float width, height;
   float xadvance; // yBearing + h-yBearing
};

struct Font {
	char* file;
	int id;
	float heightIndex;

	FT_Library library;
	FT_Face face;

	float pixelScale;

	char* fileBuffer;
	Texture tex;
	Vec2i glyphRanges[5];
	int glyphRangeCount;
	int totalGlyphCount;

	PackedChar* cData;
	int height;
	float baseOffset;
	float lineSpacing;

	Font* boldFont;
	Font* italicFont;

	bool pixelAlign;
};


// 
// Meshes.
//

struct Mesh {
	uint bufferId;
	uint elementBufferId;

	// char* buffer;
	// char* elementBuffer;
	int vertCount;
	int elementCount;
};

// 
// Samplers.
//


//
// Framebuffers.
//

enum FrameBufferSlot {
	FRAMEBUFFER_SLOT_COLOR,
	FRAMEBUFFER_SLOT_DEPTH,
	FRAMEBUFFER_SLOT_STENCIL,
	FRAMEBUFFER_SLOT_DEPTH_STENCIL,
};

struct FrameBuffer {
	uint id;

	union {
		struct {
			Texture* colorSlot[4];
			Texture* depthSlot[4];
			Texture* stencilSlot[4];
			Texture* depthStencilSlot[4];
		};

		struct {
			Texture* slots[16];
		};
	};
};

FrameBuffer* getFrameBuffer(int id);
Texture* addTexture(Texture tex);

void initFrameBuffer(FrameBuffer* fb) {
	glCreateFramebuffers(1, &fb->id);

	for(int i = 0; i < arrayCount(fb->slots); i++) {
		fb->slots[i] = 0;
	}
}

void attachToFrameBuffer(int id, int slot, int internalFormat, int w, int h, int msaa = 0) {
	FrameBuffer* fb = getFrameBuffer(id);

	bool isRenderBuffer = msaa > 0;

	Texture t;
	createTexture(&t, isRenderBuffer);
	t.internalFormat = internalFormat;
	t.dim.w = w;
	t.dim.h = h;
	t.isRenderBuffer = isRenderBuffer;
	t.msaa = msaa;

	Texture* tex = addTexture(t);

	Vec2i indexRange;
	if(slot == FRAMEBUFFER_SLOT_COLOR) indexRange = vec2i(0,4);
	else if(slot == FRAMEBUFFER_SLOT_DEPTH) indexRange = vec2i(4,8);
	else if(slot == FRAMEBUFFER_SLOT_STENCIL) indexRange = vec2i(8,12);
	else if(slot == FRAMEBUFFER_SLOT_DEPTH_STENCIL) indexRange = vec2i(12,16);

	for(int i = indexRange.x; i <= indexRange.y; i++) {
		if(fb->slots[i] == 0) {
			fb->slots[i] = tex;
			break;
		}
	}

}

void reloadFrameBuffer(int id) {
	FrameBuffer* fb = getFrameBuffer(id);

	for(int i = 0; i < arrayCount(fb->slots); i++) {
		if(!fb->slots[i]) continue;
		Texture* t = fb->slots[i];

		int slot;
		if(valueBetween(i, 0, 3)) slot = GL_COLOR_ATTACHMENT0 + i;
		else if(valueBetween(i, 4, 7)) slot = GL_DEPTH_ATTACHMENT;
		else if(valueBetween(i, 8, 11)) slot = GL_STENCIL_ATTACHMENT;
		else if(valueBetween(i, 12, 15)) slot = GL_DEPTH_STENCIL_ATTACHMENT;

		if(t->isRenderBuffer) {
			glNamedRenderbufferStorageMultisample(t->id, t->msaa, t->internalFormat, t->dim.w, t->dim.h);
			glNamedFramebufferRenderbuffer(fb->id, slot, GL_RENDERBUFFER, t->id);
		} else {
			glDeleteTextures(1, &t->id);
			glCreateTextures(GL_TEXTURE_2D, 1, &t->id);
			glTextureStorage2D(t->id, 1, t->internalFormat, t->dim.w, t->dim.h);
			glNamedFramebufferTexture(fb->id, slot, t->id, 0);
		}
	}

}

void blitFrameBuffers(int id1, int id2, Vec2i dim, int bufferBit, int filter) {
	FrameBuffer* fb1 = getFrameBuffer(id1);
	FrameBuffer* fb2 = getFrameBuffer(id2);

	glBlitNamedFramebuffer (fb1->id, fb2->id, 0,0, dim.x, dim.y, 0,0, dim.x, dim.y, bufferBit, filter);
}

void bindFrameBuffer(int id, int slot = GL_FRAMEBUFFER) {
	FrameBuffer* fb = getFrameBuffer(id);
	glBindFramebuffer(slot, fb->id);
}

void setDimForFrameBufferAttachmentsAndUpdate(int id, int w, int h) {
	FrameBuffer* fb = getFrameBuffer(id);

	for(int i = 0; i < arrayCount(fb->slots); i++) {
		if(!fb->slots[i]) continue;
		Texture* t = fb->slots[i];

		t->dim = vec2i(w, h);
	}

	reloadFrameBuffer(id);
}

uint checkStatusFrameBuffer(int id) {
	FrameBuffer* fb = getFrameBuffer(id);
	GLenum result = glCheckNamedFramebufferStatus(fb->id, GL_FRAMEBUFFER);
	return result;
}

//
// Data.
//

struct GraphicsState {
	Shader shaders[SHADER_SIZE];
	Texture textures[32]; //TEXTURE_SIZE
	int textureCount;
	Texture cubeMaps[CUBEMAP_SIZE];
	Texture textures3d[2];
	GLuint samplers[SAMPLER_SIZE];
	Mesh meshs[MESH_SIZE];

	Font fonts[10][20];
	int fontsCount;
	char* fontFolders[10];
	int fontFolderCount;

	GLuint textureUnits[16];
	GLuint samplerUnits[16];

	FrameBuffer frameBuffers[FRAMEBUFFER_SIZE];

	Vec2i screenRes;

	float zOrder;
	bool useSRGB;
};

Shader* getShader(int shaderId) {
	Shader* s = theGraphicsState->shaders + shaderId;
	return s;
}

Mesh* getMesh(int meshId) {
	Mesh* m = theGraphicsState->meshs + meshId;
	return m;
}

Texture* getTexture(int textureId) {
	Texture* t = theGraphicsState->textures + textureId;
	return t;
}

Texture* getTextureX(int textureId) {
	GraphicsState* gs = theGraphicsState;
	for(int i = 0; i < arrayCount(gs->textures); i++) {
		if(gs->textures[i].id == textureId) {
			return gs->textures + i;
		}
	}

	return 0;
}

Texture* addTexture(Texture tex) {
	GraphicsState* gs = theGraphicsState;
	gs->textures[gs->textureCount++] = tex;
	return gs->textures + (gs->textureCount - 1);
}

Texture* getCubemap(int textureId) {
	Texture* t = theGraphicsState->cubeMaps + textureId;
	return t;
}

FrameBuffer* getFrameBuffer(int id) {
	FrameBuffer* fb = theGraphicsState->frameBuffers + id;
	return fb;
}



#define Font_Error_Glyph (int)0x20-1

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
		error = FT_Set_Pixel_Sizes(font.face, 0, roundInt(height) * fullHeightToAscend); assert(error == 0);
	} else {
		error = FT_Set_Char_Size(font.face, 0, (roundInt(-height) * fullHeightToAscend) * pointFraction, 0, 0); assert(error == 0);
	}

	// Get true height from freetype.
	font.height = (face->size->metrics.ascender + abs(face->size->metrics.descender)) / pointFraction;
	font.baseOffset = (face->size->metrics.ascender / pointFraction);

	// We calculate the scaling ourselves because Freetype doesn't offer it??
	float scale = (float)face->size->metrics.ascender / (float)face->ascender;
	font.lineSpacing = roundInt(((face->height * scale) / pointFraction));
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


void bindShader(int shaderId) {
	int shader = theGraphicsState->shaders[shaderId].program;
	glBindProgramPipeline(shader);
}

uint createShader(const char* vertexShaderString, const char* fragmentShaderString, uint* vId, uint* fId) {
	*vId = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &vertexShaderString);
	*fId = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fragmentShaderString);

	uint shaderId;
	glCreateProgramPipelines(1, &shaderId);
	glUseProgramStages(shaderId, GL_VERTEX_SHADER_BIT, *vId);
	glUseProgramStages(shaderId, GL_FRAGMENT_SHADER_BIT, *fId);

	return shaderId;
}

void loadShaders() {
	GraphicsState* gs = theGraphicsState;

	for(int i = 0; i < SHADER_SIZE; i++) {
		MakeShaderInfo* info = makeShaderInfo + i; 
		Shader* s = gs->shaders + i;

		s->program = createShader(info->vertexString, info->fragmentString, &s->vertex, &s->fragment);
		s->uniformCount = info->uniformCount;
		s->uniforms = getPArray(ShaderUniform, s->uniformCount);

		for(int i = 0; i < s->uniformCount; i++) {
			ShaderUniform* uni = s->uniforms + i;
			uni->type = info->uniformNameMap[i].type;	
			uni->vertexLocation = glGetUniformLocation(s->vertex, info->uniformNameMap[i].name);
			uni->fragmentLocation = glGetUniformLocation(s->fragment, info->uniformNameMap[i].name);
		}
	}
}

void pushUniform(uint shaderId, int shaderStage, uint uniformId, void* data, int count = 1) {
	Shader* s = theGraphicsState->shaders + shaderId;
	ShaderUniform* uni = s->uniforms + uniformId;

	int i = shaderStage;
	bool setBothStages = false;
	if(shaderStage == 2) {
		i = 0;
		setBothStages = true;
	}

	for(; i < 2; i++) {
		uint stage = i == 0 ? s->vertex : s->fragment;
		uint location = i == 0 ? uni->vertexLocation : uni->fragmentLocation;

		switch(uni->type) {
			case UNIFORM_TYPE_MAT4: glProgramUniformMatrix4fv(stage, location, count, 1, (float*)data); break;
			case UNIFORM_TYPE_VEC4: glProgramUniform4fv(stage, location, count, (float*)data); break;
			case UNIFORM_TYPE_VEC3: glProgramUniform3fv(stage, location, count, (float*)data); break;
			case UNIFORM_TYPE_VEC2: glProgramUniform2fv(stage, location, count, (float*)data); break;
			case UNIFORM_TYPE_INT:  glProgramUniform1iv(stage, location, count, (int*)data); break;
			case UNIFORM_TYPE_FLOAT: glProgramUniform1fv(stage, location, count, (float*)data); break;
		}

		if(!setBothStages) break;
	}
};

void pushUniform(uint shaderId, int shaderStage, uint uniformId, float data) {
	pushUniform(shaderId, shaderStage, uniformId, &data);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, int data) {
	pushUniform(shaderId, shaderStage, uniformId, &data);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, float f0, float f1, float f2, float f3) {
	Vec4 d = vec4(f0, f1, f2, f3);
	pushUniform(shaderId, shaderStage, uniformId, &d);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, Vec4 v) {
	pushUniform(shaderId, shaderStage, uniformId, v.e);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, Vec3 v) {
	pushUniform(shaderId, shaderStage, uniformId, v.e);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, Mat4 m) {
	pushUniform(shaderId, shaderStage, uniformId, m.e);
};

void getUniform(uint shaderId, int shaderStage, uint uniformId, float* data) {
	Shader* s = theGraphicsState->shaders + shaderId;
	ShaderUniform* uni = s->uniforms + uniformId;

	uint stage = shaderStage == 0 ? s->vertex : s->fragment;
	uint location = shaderStage == 0 ? uni->vertexLocation : uni->fragmentLocation;

	glGetUniformfv(stage, location, data);
}



void drawRect(Rect r, Vec4 color, Rect uv = rect(0,0,1,1), int texture = -1, float texZ = -1) {	
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_PRIMITIVE_MODE, 0);

	Rect cd = rectCenDim(r);

	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_MOD, cd.e);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_UV, uv.min.x, uv.max.x, uv.max.y, uv.min.y);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_TEXZ, texZ);

	if(texture == -1) texture = getTexture(TEXTURE_WHITE)->id;

	uint tex[2] = {texture, texture};
	glBindTextures(0,2,tex);
	glBindSamplers(0, 1, theGraphicsState->samplers);

	glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP, 0, 4, 1, 0);
}

void drawQuad(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Vec4 color, int textureId, Rect uv, float texZ) {

	// Disabling these arrays is very important.

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	
	Vec2 verts[] = {p0, p1, p2, p3};

	if(texZ == -1) textureId = getTexture(textureId)->id;
	uint tex[2] = {textureId, textureId};

	glBindTextures(0,2,tex);

	glBindSamplers(0,1,&theGraphicsState->samplers[SAMPLER_NORMAL]);
	glBindSamplers(1,1,&theGraphicsState->samplers[SAMPLER_NORMAL]);

	Vec2 quadUVs[] = { rectBL(uv), rectTL(uv), rectTR(uv), rectBR(uv) };

	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_PRIMITIVE_MODE, true);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_UVS, quadUVs[0].e, arrayCount(quadUVs));
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_TEXZ, texZ);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_VERTS, verts[0].e, arrayCount(verts));

	glDrawArrays(GL_QUADS, 0, arrayCount(verts));
}

void ortho(Rect r) {
	r = rectCenDim(r);

	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_CAMERA, r.e);
}

void lookAt(Vec3 pos, Vec3 look, Vec3 up, Vec3 right) {
	Mat4 view;
	viewMatrix(&view, pos, look, up, right);

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VIEW, view.e);
}

void perspective(float fov, float aspect, float n, float f) {
	Mat4 proj;
	projMatrix(&proj, fov, aspect, n, f);

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_PROJ, proj.e);
}

//

enum TextStatus {
	TEXTSTATUS_END = 0, 
	TEXTSTATUS_NEWLINE, 
	TEXTSTATUS_WRAPPED, 
	TEXTSTATUS_DEFAULT, 
	TEXTSTATUS_SIZE, 
};

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
		if(valueBetweenInt(c, font->glyphRanges[i].x, font->glyphRanges[i].x+font->glyphRanges[i].y)) {
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

static int counter = 0;
static __int64 total = 0;

// Taken from stbtt_truetype.
void getPackedQuad(PackedChar *chardata, Vec2i texDim, int char_index, Vec2 pos, Rect* r, Rect* uv, int alignToInteger)
{
   PackedChar *b = chardata + char_index;

   if (alignToInteger) {
   	  *r = rectBLDim(roundFloat(pos.x + b->xBearing), roundFloat(pos.y - (b->height - b->yBearing)), b->width, b->height);

   	  (*r).left = roundFloat(pos.x + b->xBearing);
   	  (*r).bottom = roundFloat(pos.y - (b->height - b->yBearing));
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
	if(font->pixelAlign) off = roundInt(off);
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



struct TextInfo {
	Vec2 pos;
	int index;
	Vec2 posAdvance;
	Rect r;
	Rect uv;

	bool lineBreak;
	Vec2 breakPos;
};

struct TextSimInfo {
	Vec2 pos;
	int index;
	int wrapIndex;

	bool lineBreak;
	Vec2 breakPos;

	bool bold;
	bool italic;

	bool colorMode;
	Vec3 colorOverwrite;
};

TextSimInfo initTextSimInfo(Vec2 startPos) {
	TextSimInfo tsi = {};
	tsi.pos = startPos;
	tsi.index = 0;
	tsi.wrapIndex = 0;
	tsi.lineBreak = false;
	tsi.breakPos = vec2(0,0);
	return tsi;
}

enum {
	TEXT_MARKER_BOLD = 0,
	TEXT_MARKER_ITALIC,
	TEXT_MARKER_COLOR,
};

#define Marker_Size 3
#define Bold_Marker "<b>"
#define Italic_Marker "<i>"
#define Color_Marker "<c>"

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

struct TextSettings {
	Font* font;
	Vec4 color;

	int shadowMode;
	Vec2 shadowDir;
	float shadowSize;
	Vec4 shadowColor;

	bool srgb;

	bool cull;
};

TextSettings textSettings(Font* font, Vec4 color, int shadowMode, Vec2 shadowDir, float shadowSize, Vec4 shadowColor) {
	return {font, color, shadowMode, shadowDir, shadowSize, shadowColor};
}
TextSettings textSettings(Font* font, Vec4 color, int shadowMode, float shadowSize, Vec4 shadowColor) {
	return {font, color, shadowMode, vec2(-1,-1), shadowSize, shadowColor};
}
TextSettings textSettings(Font* font, Vec4 color) {
	return {font, color};
}


enum {
	TEXTSHADOW_MODE_NOSHADOW = 0,
	TEXTSHADOW_MODE_SHADOW,
	TEXTSHADOW_MODE_OUTLINE,
};

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
				Vec2 p = ti.r.min + normVec2(settings.shadowDir) * settings.shadowSize;
				Rect sr = rectBLDim(vec2(roundFloat(p.x), roundFloat(p.y)), rectDim(ti.r));

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
					// Rect sr = rectBLDim(vec2(roundFloat(p.x), roundFloat(p.y)), rectDim(ti.r));
					// pushRect(sr, ti.uv, z);
				}
			}
		}

		// if(tsi.colorMode) pushColor(vec4(tsi.colorOverwrite, 1));
		// else pushColor(c);

		// pushRect(ti.r, ti.uv, z);
		drawRect(ti.r, c, ti.uv, texId);
	}
	
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
// 				Vec2 p = ti.r.min + normVec2(settings.shadowDir) * settings.shadowSize;
// 				Rect sr = rectBLDim(vec2(roundFloat(p.x), roundFloat(p.y)), rectDim(ti.r));

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
		
		bool fLine = valueBetween(mousePos.y, ti.pos.y - f->height, ti.pos.y);
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
	strCpy(str, text + minInt(index1, index2), range);
	return str;
}




void drawCube(Vec3 trans, Vec3 scale, Vec4 color, float degrees, Vec3 rot) {
	glBindTextures(0,1,&getTexture(TEXTURE_WHITE)->id);

	Mesh* cubeMesh = getMesh(MESH_CUBE);
	glBindBuffer(GL_ARRAY_BUFFER, cubeMesh->bufferId);

	glVertexAttribPointer(0, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3) + sizeof(Vec2)));
	glEnableVertexAttribArray(2);

	Mat4 model = modelMatrix(trans, scale, degrees, rot);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODEL, model.e);
	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, color.e);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, colorSRGB(color).e);

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, false);
	pushUniform(SHADER_CUBE, 1, CUBE_UNIFORM_TEXZ, -1.0f);

	glDrawArrays(GL_QUADS, 0, cubeMesh->vertCount);
	// glDrawElements(GL_QUADS, cubeMesh->elementCount, GL_UNSIGNED_INT, (void*)0);
}

void drawLine(Vec3 p0, Vec3 p1, Vec4 color) {

	// Disabling these arrays is very important.

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

	glBindTextures(0,1,&getTexture(TEXTURE_WHITE)->id);

	Vec3 verts[] = {p0, p1};
	Vec2 quadUVs[] = {{0,0}, {0,1}};
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs[0].e, arrayCount(quadUVs));

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, true);
	pushUniform(SHADER_CUBE, 1, CUBE_UNIFORM_TEXZ, -1.0f);

	glDrawArrays(GL_LINES, 0, arrayCount(verts));
}

void drawQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, int textureId = TEXTURE_WHITE, Rect uv = rect(0,0,1,1), float texZ = -1) {

	// Disabling these arrays is very important.

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	
	Vec3 verts[] = {p0, p1, p2, p3};

	if(texZ == -1) textureId = getTexture(textureId)->id;
	uint tex[2] = {textureId, textureId};

	glBindTextures(0,2,tex);
	// glBindSamplers(0, 1, theGraphicsState->samplers);

	// Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
	Vec2 quadUVs[] = { rectBL(uv), rectTL(uv), rectTR(uv), rectBR(uv) };

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs[0].e, arrayCount(quadUVs));
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_CUBE, 1, CUBE_UNIFORM_TEXZ, texZ);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, true);

	glDrawArrays(GL_QUADS, 0, arrayCount(verts));
}

void drawQuad(Vec3 p, Vec3 normal, float size, Vec4 color) {
	Vec3 base = p;

	int sAxis[2];
	int biggestAxis = getBiggestAxis(normal, sAxis);

	float s2 = size*0.5f;

	Vec3 verts[4] = {};
	for(int i = 0; i < 4; i++) {
		Vec3 d = base;
		if(i == 0) { d.e[sAxis[0]] += -s2; d.e[sAxis[1]] += -s2; }
		else if(i == 1) { d.e[sAxis[0]] += -s2; d.e[sAxis[1]] +=  s2; }
		else if(i == 2) { d.e[sAxis[0]] +=  s2; d.e[sAxis[1]] +=  s2; }
		else if(i == 3) { d.e[sAxis[0]] +=  s2; d.e[sAxis[1]] += -s2; }
		verts[i] = d;
	}

	drawQuad(verts[0], verts[1], verts[2], verts[3], color);
}

uint createSampler(float ani, int wrapS, int wrapT, int magF, int minF, int wrapR = GL_CLAMP_TO_EDGE) {
	uint result;
	glCreateSamplers(1, &result);

	glSamplerParameteri(result, GL_TEXTURE_MAX_ANISOTROPY_EXT, ani);
	glSamplerParameteri(result, GL_TEXTURE_WRAP_S, wrapS);
	glSamplerParameteri(result, GL_TEXTURE_WRAP_T, wrapT);
	glSamplerParameteri(result, GL_TEXTURE_MAG_FILTER, magF);
	glSamplerParameteri(result, GL_TEXTURE_MIN_FILTER, minF);

	glSamplerParameteri(result, GL_TEXTURE_WRAP_R, wrapR);

	return result;
}



int stateSwitch(int state) {
	switch(state) {
		case STATE_CULL: return GL_CULL_FACE;
		case STATE_SCISSOR: return GL_SCISSOR_TEST;
		case STATE_DEPTH_TEST: return GL_DEPTH_TEST;
		case STATE_POLYGON_OFFSET: return GL_POLYGON_OFFSET_FILL;
	}
	return 0;
}

#define dcGetStructAndIncrement(structType) \
	Draw_Command_##structType dc = *((Draw_Command_##structType*)drawListIndex); \
	drawListIndex += sizeof(Draw_Command_##structType); \

void executeCommandList(DrawCommandList* list, bool print = false, bool skipStrings = false) {
	// TIMER_BLOCK();

	if(print) {
		printf("\nDrawCommands: %i \n", list->count);
	}

	char* drawListIndex = (char*)list->data;
	for(int i = 0; i < list->count; i++) {
		int command = *((int*)drawListIndex);
		drawListIndex += sizeof(int);

		if(print) {
			printf("%i ", command);
		}

		switch(command) {
			case Draw_Command_Cube_Type: {
				dcGetStructAndIncrement(Cube);
				drawCube(dc.trans, dc.scale, dc.color, dc.degrees, dc.rot);
			} break;

			case Draw_Command_Line_Type: {
				dcGetStructAndIncrement(Line);
				drawLine(dc.p0, dc.p1, dc.color);
			} break;

			case Draw_Command_Line2d_Type: {
				dcGetStructAndIncrement(Line2d);

				// Vec2 verts[] = {dc.p0, dc.p1};
				Vec2 verts[] = {roundInt(dc.p0.x)-0.5f, roundInt(dc.p0.y)-0.5f, roundInt(dc.p1.x)-0.5f, roundInt(dc.p1.y)-0.5f};
				pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_VERTS, verts, 2);
				pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_COLOR, colorSRGB(dc.color).e);
				pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_PRIMITIVE_MODE, 1);

				uint tex[1] = {getTexture(TEXTURE_WHITE)->id};
				glBindTextures(0,1,tex);

				glDrawArrays(GL_LINES, 0, 2);
			} break;

			case Draw_Command_Quad2d_Type: {
				dcGetStructAndIncrement(Quad2d);

				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color, dc.textureId, dc.uv, dc.texZ);
			} break;

			case Draw_Command_Quad_Type: {
				dcGetStructAndIncrement(Quad);

				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color, dc.textureId, dc.uv, dc.texZ);
			} break;

			case Draw_Command_State_Type: {
				dcGetStructAndIncrement(State);

				switch(dc.state) {
					case STATE_POLYGONMODE: {
						int m;
						switch(dc.value) {
							case POLYGON_MODE_FILL: m = GL_FILL; break;
							case POLYGON_MODE_LINE: m = GL_LINE; break;
							case POLYGON_MODE_POINT: m = GL_POINT; break;
						}
						glPolygonMode(GL_FRONT_AND_BACK, m);
					} break;

					case STATE_LINEWIDTH: glLineWidth(dc.value); break;

					case STATE_DEPTH_FUNC: glDepthFunc(dc.value); break;

					default: {} break;
				}
			} break;

			case Draw_Command_Enable_Type: {
				dcGetStructAndIncrement(Int);

				int m = stateSwitch(dc.state);
				glEnable(m);
			} break;			

			case Draw_Command_Disable_Type: {
				dcGetStructAndIncrement(Int);

				int m = stateSwitch(dc.state);
				glDisable(m);
			} break;	

			case Draw_Command_Rect_Type: {
				dcGetStructAndIncrement(Rect);
				int texture = dc.texture == -1 ? getTexture(TEXTURE_WHITE)->id : dc.texture;
				drawRect(dc.r, dc.color, dc.uv, texture, dc.texZ-1);
			} break;

			case Draw_Command_RoundedRect_Type: {
				dcGetStructAndIncrement(RoundedRect);

				if(dc.steps == 0) dc.steps = 6;

				float s = dc.size;
				Rect r = dc.r;
				drawRect(rect(r.min.x+s, r.min.y, r.max.x-s, r.max.y), dc.color);
				drawRect(rect(r.min.x, r.min.y+s, r.max.x, r.max.y-s), dc.color);

				Vec2 verts[10];

				pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_PRIMITIVE_MODE, 1);
				pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_COLOR, colorSRGB(dc.color).e);
				uint tex[1] = {getTexture(TEXTURE_WHITE)->id};
				glBindTextures(0,1,tex);

				Rect rc = rectExpand(r, -vec2(s,s)*2);
				Vec2 corners[] = {rc.max, vec2(rc.max.x, rc.min.y), rc.min, vec2(rc.min.x, rc.max.y)};
				for(int cornerIndex = 0; cornerIndex < 4; cornerIndex++) {

					Vec2 corner = corners[cornerIndex];
					float round = s;
					float start = M_PI_2*cornerIndex;

					verts[0] = corner;

					for(int i = 0; i < dc.steps; i++) {
						float angle = start + i*(M_PI_2/(dc.steps-1));
						Vec2 v = vec2(sin(angle), cos(angle));

						verts[i+1] = corner + v*round;
					}

					pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_VERTS, verts, dc.steps+1);
					glDrawArraysInstancedBaseInstance(GL_TRIANGLE_FAN, 0, dc.steps+1, 1, 0);
				}

			} break;

			case Draw_Command_Text_Type: {
				dcGetStructAndIncrement(Text);

				if(skipStrings) break;

				if(dc.cullWidth == -1) {
					TextSettings ts = textSettings(dc.font, dc.color, TEXTSHADOW_MODE_SHADOW, vec2(dc.shadow,-dc.shadow), lenVec2(vec2(dc.shadow,-dc.shadow)), dc.shadowColor);

					drawText(dc.text, dc.pos, vec2i(dc.vAlign, dc.hAlign), dc.wrapWidth, ts);

				} else {
					TextSettings ts = textSettings(dc.font, dc.color, TEXTSHADOW_MODE_SHADOW, vec2(dc.shadow,-dc.shadow), lenVec2(vec2(dc.shadow,-dc.shadow)), dc.shadowColor);
					ts.cull = true;

					drawText(dc.text, dc.pos, vec2i(dc.vAlign, dc.hAlign), dc.cullWidth, ts);
				}

			} break;

			case Draw_Command_Scissor_Type: {
				dcGetStructAndIncrement(Scissor);
				Rect r = dc.rect;
				Vec2 dim = rectDim(r);
				assert(dim.w >= 0 && dim.h >= 0);
				glScissor(r.min.x, r.min.y, dim.x, dim.y);
			} break;

			case Draw_Command_Blend_Type: {
				dcGetStructAndIncrement(Blend);
				glBlendFuncSeparate(dc.sourceColor, dc.destinationColor, 
				                    dc.sourceAlpha, dc.destinationAlpha);
				glBlendEquationSeparate(dc.functionColor, dc.functionAlpha);
			} break;

			case Draw_Command_PolygonOffset_Type: {
				dcGetStructAndIncrement(PolygonOffset);
				glPolygonOffset(dc.factor, dc.units);
			} break;

			default: {} break;
		}
	}

	if(print) {
		printf("\n\n");
	}
}


void scissorTest(Rect r) {
	int left   = roundInt(r.left);
	int bottom = roundInt(r.bottom);
	int right  = roundInt(r.right);
	int top    = roundInt(r.top);

	// glScissor(left, bottom, right-left, top-bottom);
	dcScissor(rect(left, bottom, right, top));
}

Rect scissorRectScreenSpace(Rect r, float screenHeight) {
	Rect scissorRect = {r.min.x, r.min.y+screenHeight, r.max.x, r.max.y+screenHeight};
	return scissorRect;
}

void scissorTestScreen(Rect r) {
	Rect sr = scissorRectScreenSpace(r, theGraphicsState->screenRes.h);
	if(rectW(sr) < 0 || rectH(sr) < 0) sr = rect(0,0,0,0);

	scissorTest(sr);
}

