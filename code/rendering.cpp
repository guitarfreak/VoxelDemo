
struct DrawCommandList;
extern DrawCommandList* globalCommandList;
struct GraphicsState;
extern GraphicsState* globalGraphicsState;


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
			int v = va_arg(vl, int);
			intToStr(valueBuffer, v);
			int sLen = strLen(valueBuffer);
			memCpy(buffer + bi, valueBuffer, sLen);

			ti += 2;
			bi += sLen;
			getTStringX(sLen);
		} if(text[ti] == '%' && text[ti+1] == 's') {
			char* str = va_arg(vl, char*);
			int sLen = strLen(str);
			memCpy(buffer + bi, str, sLen);

			ti += 2;
			bi += sLen;
			getTStringX(sLen);
		} if(text[ti] == '%' && text[ti+1] == '%') {
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
	STATE_LINEWIDTH,
	STATE_CULL,
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
	Draw_Command_Rect_Type,
	Draw_Command_RoundedRect_Type,
	Draw_Command_Text_Type,
	Draw_Command_Scissor_Type,
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

void dcQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Quad, Quad);

	command->p0 = p0;
	command->p1 = p1;
	command->p2 = p2;
	command->p3 = p3;
	command->color = color;
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
	// glGenerateTextureMipmap(ad->cubemapTextureId);

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

struct Font {
	int id;
	char* fileBuffer;
	Texture tex;
	int glyphStart, glyphCount;
	stbtt_bakedchar* cData;
	int height;
	float baseOffset;
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
	Font fonts[FONT_SIZE][20];
	Mesh meshs[MESH_SIZE];

	GLuint textureUnits[16];
	GLuint samplerUnits[16];

	FrameBuffer frameBuffers[FRAMEBUFFER_SIZE];
};

Shader* getShader(int shaderId) {
	Shader* s = globalGraphicsState->shaders + shaderId;
	return s;
}

Mesh* getMesh(int meshId) {
	Mesh* m = globalGraphicsState->meshs + meshId;
	return m;
}

Texture* getTexture(int textureId) {
	Texture* t = globalGraphicsState->textures + textureId;
	return t;
}

Texture* getTextureX(int textureId) {
	GraphicsState* gs = globalGraphicsState;
	for(int i = 0; i < arrayCount(gs->textures); i++) {
		if(gs->textures[i].id == textureId) {
			return gs->textures + i;
		}
	}

	return 0;
}

Texture* addTexture(Texture tex) {
	GraphicsState* gs = globalGraphicsState;
	gs->textures[gs->textureCount++] = tex;
	return gs->textures + (gs->textureCount - 1);
}

Texture* getCubemap(int textureId) {
	Texture* t = globalGraphicsState->cubeMaps + textureId;
	return t;
}

FrameBuffer* getFrameBuffer(int id) {
	FrameBuffer* fb = globalGraphicsState->frameBuffers + id;
	return fb;
}

Font* getFont(int fontId, int height) {

	// Search if Font in this size exists, if not create it.

	int fontSlots = arrayCount(globalGraphicsState->fonts[0]);
	Font* fontArray = globalGraphicsState->fonts[fontId];
	Font* fontSlot = 0;
	for(int i = 0; i < fontSlots; i++) {
		int fontHeight = fontArray[i].height;
		if(fontHeight == height || fontHeight == 0) {
			fontSlot = fontArray + i;
			break;
		}
	}

	// We are going to assume for now that a font size of 0 means it is uninitialized.

	if(fontSlot->height == 0) {
		Font font;
		char* path = fontPaths[fontId];

		// font.fileBuffer = (char*)getPMemory(fileSize(path) + 1);
		char* fileBuffer = getTArray(char, fileSize(path) + 1);

		readFileToBuffer(fileBuffer, path);
		Vec2i size = vec2i(512,512);
		unsigned char* fontBitmapBuffer = (unsigned char*)getTMemory(size.x*size.y);
		unsigned char* fontBitmap = (unsigned char*)getTMemory(size.x*size.y*4);
		
		font.id = fontId;
		font.height = height;
		font.baseOffset = 0.8f;
		font.glyphStart = 32;
		font.glyphCount = 95;
		font.cData = (stbtt_bakedchar*)getPMemory(sizeof(stbtt_bakedchar)*font.glyphCount);
		stbtt_BakeFontBitmap((unsigned char*)fileBuffer, 0, font.height, fontBitmapBuffer, size.w, size.h, font.glyphStart, font.glyphCount, font.cData);
		for(int i = 0; i < size.w*size.h; i++) {
			fontBitmap[i*4] = fontBitmapBuffer[i];
			fontBitmap[i*4+1] = fontBitmapBuffer[i];
			fontBitmap[i*4+2] = fontBitmapBuffer[i];
			fontBitmap[i*4+3] = fontBitmapBuffer[i];
		}

		Texture tex;
		loadTexture(&tex, fontBitmap, size.w, size.h, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		font.tex = tex;

		*fontSlot = font;
	}

	return fontSlot;
}

void bindShader(int shaderId) {
	int shader = globalGraphicsState->shaders[shaderId].program;
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
	GraphicsState* gs = globalGraphicsState;

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
	Shader* s = globalGraphicsState->shaders + shaderId;
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
	Shader* s = globalGraphicsState->shaders + shaderId;
	ShaderUniform* uni = s->uniforms + uniformId;

	uint stage = shaderStage == 0 ? s->vertex : s->fragment;
	uint location = shaderStage == 0 ? uni->vertexLocation : uni->fragmentLocation;

	glGetUniformfv(stage, location, data);
}



void drawRect(Rect r, Vec4 color, Rect uv = rect(0,0,1,1), int texture = -1, float texZ = -1) {	
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_PRIMITIVE_MODE, 0);

	Rect cd = rectGetCenDim(r);

	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_MOD, cd.e);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_UV, uv.min.x, uv.max.x, uv.max.y, uv.min.y);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_TEXZ, texZ);

	if(texture == -1) texture = getTexture(TEXTURE_WHITE)->id;

	uint tex[2] = {texture, texture};
	glBindTextures(0,2,tex);
	glBindSamplers(0, 1, globalGraphicsState->samplers);

	glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP, 0, 4, 1, 0);
}

void ortho(Rect r) {
	r = rectGetCenDim(r);

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




enum TextStatus {
	TEXTSTATUS_END = 0, 
	TEXTSTATUS_NEWLINE, 
	TEXTSTATUS_WRAPPED, 
	TEXTSTATUS_DEFAULT, 
	TEXTSTATUS_SIZE, 
};

void getTextQuad(char c, Font* font, Vec2 pos, Rect* r, Rect* uv) {
	stbtt_aligned_quad q;
	stbtt_GetBakedQuad(font->cData, font->tex.dim.w, font->tex.dim.h, c-font->glyphStart, pos.x, pos.y, &q);

	float baseLine = 0.8f;
	float off = baseLine * font->height;

	*r = rect(q.x0, q.y0 - off, q.x1, q.y1 - off);
	*uv = rect(q.s0, q.t0, q.s1, q.t1);
}

float getCharAdvance(char c, Font* font) {
	float result = stbtt_GetCharAdvance(font->cData, c-font->glyphStart);
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
};

TextSimInfo initTextSimInfo(Vec2 startPos) {
	TextSimInfo tsi;
	tsi.pos = startPos;
	tsi.index = 0;
	tsi.wrapIndex = 0;
	tsi.lineBreak = false;
	tsi.breakPos = vec2(0,0);
	return tsi;
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
	char t = text[i];

	bool wrapped = false;

	if(wrapWidth != 0 && i == tsi->wrapIndex) {
		char c = text[i];
		float wordWidth = 0;
		if(c == '\n') wordWidth = getCharAdvance(c, font);

		int it = i;
		while(c != '\n' && c != '\0' && c != ' ') {
			wordWidth += getCharAdvance(c, font);
			it++;
			c = text[it];
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
		tsi->pos.y -= font->height;

		if(wrapped) {
			return textSim(text, font, tsi, ti, startPos, wrapWidth);
		}
	} else {
		getTextQuad(t, font, tsi->pos, &ti->r, &ti->uv);
		tsi->pos.x += getCharAdvance(t, font);
	}

	if(ti) {
		ti->pos = oldPos;
		ti->index = tsi->index;
		ti->posAdvance = tsi->pos - oldPos;
	}

	tsi->index++;

	return 1;
}

float getTextHeight(char* text, Font* font, Vec2 startPos = vec2(0,0), int wrapWidth = 0) {

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		if(!textSim(text, font, &tsi, &ti, startPos, wrapWidth)) break;
	}

	float height = startPos.y - (tsi.pos.y - font->height);
	return height;
}

Vec2 getTextDim(char* text, Font* font, Vec2 startPos = vec2(0,0), int wrapWidth = 0) {
	float maxX = startPos.x;

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		if(!textSim(text, font, &tsi, &ti, startPos, wrapWidth)) break;

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

void drawText(char* text, Font* font, Vec2 startPos, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);
	startPos = vec2(roundInt((int)startPos.x), roundInt((int)startPos.y));

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		if(!textSim(text, font, &tsi, &ti, startPos, wrapWidth)) break;
		if(text[ti.index] == '\n') continue;

		drawRect(ti.r, color, ti.uv, font->tex.id);
	}
}

Vec2 textIndexToPos(char* text, Font* font, Vec2 startPos, int index, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		int result = textSim(text, font, &tsi, &ti, startPos, wrapWidth);

		if(ti.index == index) {
			Vec2 pos = ti.pos - vec2(0, font->height/2);
			return pos;
		}

		if(!result) break;
	}

	return vec2(0,0);
}

void drawTextSelection(char* text, Font* font, Vec2 startPos, int index1, int index2, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0, DrawCommandList* drawList = 0) {
	if(index1 == index2) return;
	if(index1 > index2) swap(&index1, &index2);

	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	Vec2 lineStart;
	bool drawSelection = false;

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		int result = textSim(text, font, &tsi, &ti, startPos, wrapWidth);

		bool endReached = ti.index == index2;

		if(drawSelection) {
			if(ti.lineBreak || endReached) {

				Vec2 lineEnd;
				if(ti.lineBreak) lineEnd = ti.breakPos;
				else if(!result) lineEnd = tsi.pos;
				else lineEnd = ti.pos;

				Rect r = rect(lineStart - vec2(0,font->height), lineEnd);
				dcRect(r, color, drawList);

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
		TextInfo ti;
		int result = textSim(text, font, &tsi, &ti, startPos, wrapWidth);
		
		bool fLine = valueBetween(mousePos.y, ti.pos.y - font->height, ti.pos.y);
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

	glDrawArrays(GL_LINES, 0, arrayCount(verts));
}

void drawQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color) {

	// Disabling these arrays is very important.

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	
	Vec3 verts[] = {p0, p1, p2, p3};

	uint tex[2] = {getTexture(TEXTURE_WHITE)->id, 0};
	glBindTextures(0,2,tex);

	Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs[0].e, arrayCount(quadUVs));

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, colorSRGB(color).e);
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

				Vec2 verts[] = {dc.p0, dc.p1};
				// Vec2 verts[] = {(int)dc.p0.x, (int)dc.p0.y, (int)dc.p1.x, (int)dc.p1.y};
				pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_VERTS, verts, 2);
				pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_PRIMITIVE_MODE, 1);
				pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_COLOR, colorSRGB(dc.color).e);

				uint tex[1] = {getTexture(TEXTURE_WHITE)->id};
				glBindTextures(0,1,tex);

				glDrawArraysInstancedBaseInstance(GL_LINES, 0, 2, 1, 0);
			} break;

			case Draw_Command_Quad_Type: {
				dcGetStructAndIncrement(Quad);

				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color);
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

				if(dc.shadow != 0) 
					drawText(dc.text, dc.font, dc.pos + vec2(dc.shadow,-dc.shadow), dc.shadowColor, vec2i(dc.vAlign, dc.hAlign), dc.wrapWidth);
				drawText(dc.text, dc.font, dc.pos, dc.color, vec2i(dc.vAlign, dc.hAlign), dc.wrapWidth);

			} break;

			case Draw_Command_Scissor_Type: {
				dcGetStructAndIncrement(Scissor);
				Rect r = dc.rect;
				Vec2 dim = rectGetDim(r);
				assert(dim.w >= 0 && dim.h >= 0);
				glScissor(r.min.x, r.min.y, dim.x, dim.y);
			} break;

			default: {} break;
		}
	}

	if(print) {
		printf("\n\n");
	}
}
