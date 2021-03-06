
extern GraphicsState* theGraphicsState;

struct GraphicsState {
	Shader shaders[SHADER_SIZE];
	GLuint samplers[SAMPLER_SIZE];
	Mesh meshs[MESH_SIZE];

	Texture* textures;
	int texturesCount;
	int texturesCountMax;
	Texture* textureWhite;

	Font fonts[10][20];
	int fontsCount;
	char* fontFolders[10];
	int fontFolderCount;

	FrameBuffer* frameBuffers;
	int frameBufferCount;
	int frameBufferCountMax;
	Texture frameBufferTextures[32];
	int frameBufferTextureCount;

	//

	Vec2i screenRes;

	float zOrder;
	bool useSRGB;
};

//
// @Textures.
// 

Texture* getTexture(char* name) {
	GraphicsState* gs = theGraphicsState;
	for(int i = 0; i < gs->texturesCount; i++) {
		if(strCompare(gs->textures[i].name, name)) {
			return gs->textures + i;
		}
	}

	return 0;
}

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
		texture->type = TEXTURE_TYPE_2D;
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

	int skySize = texWidth/4;

	if(!reload) {
		texture->dim = vec2i(skySize, skySize);
		texture->channels = 4;
		texture->levels = 6;
		texture->type = TEXTURE_TYPE_CUBEMAP;

		glCreateTextures(GL_TEXTURE_CUBE_MAP_ARRAY, 1, &texture->id);
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

	// glGenerateTextureMipmap(texture->id);

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

#ifdef STB_IMAGE_WRITE_IMPLEMENTATION
void writeTextureToFile(Texture* tex, char* path) {
	int w = tex->dim.w;
	int h = tex->dim.h;

	int bufSize = w*h*4;
	char* buffer = mallocArray(char, bufSize); 
	defer {free(buffer);};

	int texId = tex->id;
	glGetTextureImage(texId, 0, GL_RGBA, GL_UNSIGNED_BYTE, bufSize, buffer);

	int result = stbi_write_png(path, w, h, 4, buffer, w*4);
}
#endif

//
// @Sampler.
//

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

//
// @Framebuffers.
//

FrameBuffer* getFrameBuffer(char* name) {
	GraphicsState* gs = theGraphicsState;

	for(int i = 0; i < gs->frameBufferCount; i++) {
		if(strCompare(gs->frameBuffers[i].name, name)) {
			return gs->frameBuffers + i;
		}
	}

	return 0;
}

void initFrameBuffer(FrameBuffer* fb) {
	glCreateFramebuffers(1, &fb->id);
	for(int i = 0; i < arrayCount(fb->slots); i++) {
		fb->slots[i] = 0;
	}
}

FrameBuffer* addFrameBuffer(char* name) {
	FrameBuffer* fb = theGraphicsState->frameBuffers + theGraphicsState->frameBufferCount;
	theGraphicsState->frameBufferCount++;

	assert(theGraphicsState->frameBufferCount < theGraphicsState->frameBufferCountMax);

	fb->name = getPStringCpy(name);

	initFrameBuffer(fb);

	return fb;
}

void attachToFrameBuffer(FrameBuffer* fb, int slot, int internalFormat, int w, int h, int msaa = 0) {

	bool isRenderBuffer = msaa > 0;

	GraphicsState* gs = theGraphicsState;
	Texture* tex = &gs->frameBufferTextures[gs->frameBufferTextureCount++];

	createTexture(tex, isRenderBuffer);
	tex->internalFormat = internalFormat;
	tex->dim.w = w;
	tex->dim.h = h;
	tex->isRenderBuffer = isRenderBuffer;
	tex->msaa = msaa;

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

void reloadFrameBuffer(FrameBuffer* fb) {
	for(int i = 0; i < arrayCount(fb->slots); i++) {
		if(!fb->slots[i]) continue;
		Texture* t = fb->slots[i];

		int slot;
		if(between(i, 0, 3)) slot = GL_COLOR_ATTACHMENT0 + i;
		else if(between(i, 4, 7)) slot = GL_DEPTH_ATTACHMENT;
		else if(between(i, 8, 11)) slot = GL_STENCIL_ATTACHMENT;
		else if(between(i, 12, 15)) slot = GL_DEPTH_STENCIL_ATTACHMENT;

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

void blitFrameBuffers(char* name1, char* name2, Vec2i dim, int bufferBit, int filter) {
	FrameBuffer* fb1 = getFrameBuffer(name1);
	FrameBuffer* fb2 = getFrameBuffer(name2);

	glBlitNamedFramebuffer(fb1->id, fb2->id, 0,0, dim.x, dim.y, 0,0, dim.x, dim.y, bufferBit, filter);
}

void bindFrameBuffer(char* name, int slot = GL_FRAMEBUFFER) {
	FrameBuffer* fb = getFrameBuffer(name);
	glBindFramebuffer(slot, fb->id);
}

void setDimForFrameBufferAttachmentsAndUpdate(char* name, int w, int h) {
	FrameBuffer* fb = getFrameBuffer(name);

	for(int i = 0; i < arrayCount(fb->slots); i++) {
		if(!fb->slots[i]) continue;
		Texture* t = fb->slots[i];

		t->dim = vec2i(w, h);
	}

	reloadFrameBuffer(fb);
}

uint checkStatusFrameBuffer(char* name) {
	FrameBuffer* fb = getFrameBuffer(name);
	GLenum result = glCheckNamedFramebufferStatus(fb->id, GL_FRAMEBUFFER);
	return result;
}

void clearFrameBuffer(char* name, Vec4 c, int bits) {
	bindFrameBuffer(name);
	glClearColor(c.r, c.g, c.b, c.a);
	glClear(bits);
}

//
// @Mesh.
//

Mesh* getMesh(int meshId) {
	Mesh* m = theGraphicsState->meshs + meshId;
	return m;
}

//
// @Shader.
//

Shader* getShader(int shaderId) {
	Shader* s = theGraphicsState->shaders + shaderId;
	return s;
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

		for(int stage = 0; stage < 2; stage++) {
			int program = stage == 0 ? s->vertex : s->fragment;

			int count;
			glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &count);

			s->uniforms[stage] = getPArray(ShaderUniform, count);
			s->uniformCount[stage] = count;

			ShaderUniform* uniforms = s->uniforms[stage];

			for(int i = 0; i < count; i++) {
				int size;
				GLenum type;
				char name[20];
			    glGetActiveUniform(program, i, arrayCount(name), 0, &size, &type, name);

			    // Remove "[0]" at end of array uniform names.
			    int len = strLen(name);
			    if(name[len-1] == ']') name[len-3] = '\0';

				uint location = glGetUniformLocation(program, name);

			    uniforms[i].name = getPStringCpy(name);
			    uniforms[i].location = location;
			}
		}

	}
}

void pushUniform(uint shaderId, int shaderStage, char* name, int type, void* data, int count = 1) {
	Shader* s = theGraphicsState->shaders + shaderId;

	int stage = shaderStage;
	bool setBothStages = false;
	if(shaderStage == 2) {
		stage = 0;
		setBothStages = true;
	}

	for(; stage < 2; stage++) {
		uint program = stage == 0 ? s->vertex : s->fragment;

		ShaderUniform* uniforms = s->uniforms[stage];
		int uniformCount = s->uniformCount[stage];

		// Get uniform location.
		uint location = -1;
		for(int i = 0; i < uniformCount; i++) {
			if(strCompare(uniforms[i].name, name)) {
				location = uniforms[i].location;
				break;
			}
		}

		if(location == -1) continue;

		switch(type) {
			case UNIFORM_TYPE_MAT4: glProgramUniformMatrix4fv(program, location, count, 1, (float*)data); break;
			case UNIFORM_TYPE_VEC4: glProgramUniform4fv(program, location, count, (float*)data); break;
			case UNIFORM_TYPE_VEC3: glProgramUniform3fv(program, location, count, (float*)data); break;
			case UNIFORM_TYPE_VEC2: glProgramUniform2fv(program, location, count, (float*)data); break;
			case UNIFORM_TYPE_INT:  glProgramUniform1iv(program, location, count,   (int*)data); break;
			case UNIFORM_TYPE_FLOAT:glProgramUniform1fv(program, location, count, (float*)data); break;
		}

		if(!setBothStages) break;
	}
};

void pushUniform(uint shaderId, int shaderStage, char* name, float f0, float f1, float f2, float f3) {
	Vec4 d = vec4(f0, f1, f2, f3);
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_VEC4, &d);
};
void pushUniform(uint shaderId, int shaderStage, char* name, float data) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_FLOAT, &data);
};
void pushUniform(uint shaderId, int shaderStage, char* name, int data) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_INT, &data);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Vec4 v) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_VEC4, v.e);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Vec3 v) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_VEC3, v.e);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Vec2 v) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_VEC2, v.e);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Mat4 m) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_MAT4, m.e);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Rect v) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_VEC4, v.e);
};

void pushUniform(uint shaderId, int shaderStage, char* name, float* data, int size) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_FLOAT, &data, size);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Vec4* v, int size) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_VEC4, v, size);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Vec3* v, int size) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_VEC3, v, size);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Vec2* v, int size) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_VEC2, v, size);
};
void pushUniform(uint shaderId, int shaderStage, char* name, Mat4* m) {
	pushUniform(shaderId, shaderStage, name, UNIFORM_TYPE_MAT4, m);
};

//
// @Drawing.
//

void drawRect(Rect r, Vec4 color, Rect uv = rect(0,0,1,1), int texture = -1, float texZ = -1) {	
	pushUniform(SHADER_Quad, 0, "primitiveMode", 0);

	Rect cd = r.cenDim();

	pushUniform(SHADER_Quad, 0, "mod", cd);
	pushUniform(SHADER_Quad, 0, "setUV", uv.min.x, uv.max.x, uv.max.y, uv.min.y);
	pushUniform(SHADER_Quad, 0, "setColor", linearToGamma(color));
	pushUniform(SHADER_Quad, 0, "texZ", texZ);

	if(texture == -1) texture = theGraphicsState->textureWhite->id;

	uint tex[2] = {texture, texture};
	glBindTextures(0,2,tex);
	glBindSamplers(0, 1, theGraphicsState->samplers);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void drawLine(Vec2 p0, Vec2 p1, Vec4 color) {
	Vec2 verts[] = { roundIntf(p0.x)-0.5f, roundIntf(p0.y)-0.5f, 
		             roundIntf(p1.x)-0.5f, roundIntf(p1.y)-0.5f };
	Vec4 colors[] = { vec4(1), vec4(1), };

	glBindTextures(0,1,&theGraphicsState->textureWhite->id);

	pushUniform(SHADER_Quad, 0, "primitiveMode", true);
	pushUniform(SHADER_Quad, 0, "verts", verts, arrayCount(verts));
	pushUniform(SHADER_Quad, 0, "colors", colors, arrayCount(colors));
	pushUniform(SHADER_Quad, 0, "setColor", linearToGamma(color));

	glDrawArrays(GL_LINES, 0, 2);
}

void drawRectRounded(Rect r, Vec4 color, int size, int steps = 0) {
	if(steps == 0) steps = 6;

	float s = size;
	drawRect(rect(r.min.x+s, r.min.y, r.max.x-s, r.max.y), color);
	drawRect(rect(r.min.x, r.min.y+s, r.max.x, r.max.y-s), color);

	Vec2 verts[10];
	Vec4 colors[10];
	for(int i = 0; i < arrayCount(colors); i++) colors[i] = vec4(1);

	glBindTextures(0,1,&theGraphicsState->textureWhite->id);

	pushUniform(SHADER_Quad, 0, "primitiveMode", true);
	pushUniform(SHADER_Quad, 0, "colors", colors, arrayCount(colors));
	pushUniform(SHADER_Quad, 0, "setColor", linearToGamma(color));

	Rect rc = r.expand(-vec2(s,s)*2);
	Vec2 corners[] = {rc.max, vec2(rc.max.x, rc.min.y), rc.min, vec2(rc.min.x, rc.max.y)};
	for(int cornerIndex = 0; cornerIndex < 4; cornerIndex++) {

		Vec2 corner = corners[cornerIndex];
		float round = s;
		float start = M_PI_2*cornerIndex;

		verts[0] = corner;

		for(int i = 0; i < steps; i++) {
			float angle = start + i*(M_PI_2/(steps-1));
			Vec2 v = vec2(sin(angle), cos(angle));

			verts[i+1] = corner + v*round;
		}

		pushUniform(SHADER_Quad, 0, "verts", verts, steps+1);
		glDrawArrays(GL_TRIANGLE_FAN, 0, steps+1);
	}
}

void drawRectRoundedOutline(Rect r, Vec4 color, int size, float offset = -1, int steps = 0) {
	// if(size == 0) {
	// 	drawRectOutline(r, dc.color);
	// 	return;
	// }

	#define Rounding_Mod 1/2
	steps = round(M_PI_2 * size * Rounding_Mod);

	if(steps == 0) steps = 6;

	float s = size;
	s = min(s, min(r.w()/2, r.h()/2));

	//

	Vec2 verts[32];
	Vec4 colors[32];
	for(int i = 0; i < arrayCount(colors); i++) colors[i] = vec4(1);

	glBindTextures(0,1,&theGraphicsState->textureWhite->id);

	pushUniform(SHADER_Quad, 0, "primitiveMode", 1);
	pushUniform(SHADER_Quad, 0, "colors", colors, arrayCount(colors));
	pushUniform(SHADER_Quad, 0, "setColor", linearToGamma(color));

	float off = offset*0.5f;
	Rect rc = r.expand(-vec2(s-off,s-off)*2);
	Vec2 corners[] = { rc.tr(), rc.br(), rc.bl(), rc.tl() };
	int index = 0;
	for(int cornerIndex = 0; cornerIndex < 4; cornerIndex++) {

		Vec2 corner = corners[cornerIndex];
		float round = s;
		float start = M_PI_2*cornerIndex;

		for(int i = 0; i < steps; i++) {
			float angle = start + i*(M_PI_2/(steps-1));
			Vec2 v = vec2(sin(angle), cos(angle));

			verts[index++] = corner + v*round;
		}
	}

	int size = index+1;
	verts[index] = verts[0];
	pushUniform(SHADER_Quad, 0, "verts", verts, size);
	glDrawArrays(GL_LINE_STRIP, 0, size);
}

void drawRectNewColored(Rect r, Vec4 c0, Vec4 c1, Vec4 c2, Vec4 c3) {	
	Vec2 verts[] = { r.bl(), r.tl(), r.tr(), r.br() };
	Vec4 colors[] = { linearToGamma(c0), linearToGamma(c1), 
		              linearToGamma(c2), linearToGamma(c3) };

	glBindTextures(0,1,&theGraphicsState->textureWhite->id);

	pushUniform(SHADER_Quad, 0, "primitiveMode", true);
	pushUniform(SHADER_Quad, 0, "verts", verts, arrayCount(verts));
	pushUniform(SHADER_Quad, 0, "colors", colors, arrayCount(colors));
	pushUniform(SHADER_Quad, 0, "setColor", vec4(1));

	glDrawArrays(GL_QUADS, 0, arrayCount(verts));
}

void drawRectNewColoredH(Rect r, Vec4 c0, Vec4 c1) {	
	drawRectNewColored(r, c0, c1, c1, c0);
}
void drawRectNewColoredW(Rect r, Vec4 c0, Vec4 c1) {	
	drawRectNewColored(r, c0, c0, c1, c1);
}

void drawRectRoundedGradient(Rect r, Vec4 color, int size, Vec4 off, int steps = 0) {	
	glEnable(GL_STENCIL_TEST);

	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
	glStencilFunc(GL_ALWAYS, 1, 0xFF);
	glStencilMask(0xFF);

	drawRectRounded(r, color, size);

	glStencilFunc(GL_EQUAL, 1, 0xFF);
	glStencilMask(0x00); 

	// setSRGB(false); @fix
	drawRectNewColoredH(r, color - off, color + off);
	// setSRGB(true); @fix

	glStencilMask(0xFF); 
	glDisable(GL_STENCIL_TEST);
}

void drawTriangle(Vec2 p, float size, Vec2 dir, Vec4 color) {
	Vec2 verts[3];

	dir = norm(dir) * size;
	verts[0] = p + dir;
	verts[1] = p + rotateVec2(dir, degreeToRadian(360/3));
	verts[2] = p + rotateVec2(dir, degreeToRadian(360/3*2));

	Vec4 colors[] = { vec4(1), vec4(1), vec4(1), };

	glBindTextures(0,1,&theGraphicsState->textureWhite->id);

	pushUniform(SHADER_Quad, 0, "primitiveMode", true);
	pushUniform(SHADER_Quad, 0, "verts", verts, arrayCount(verts));
	pushUniform(SHADER_Quad, 0, "colors", colors, arrayCount(colors));
	pushUniform(SHADER_Quad, 0, "setColor", linearToGamma(color));

	glDrawArrays(GL_TRIANGLES, 0, arrayCount(verts));
}

void drawQuad(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Vec4 color, int textureId, Rect uv, float texZ) {

	Vec2 verts[] = { p0, p1, p2, p3 };

	// if(texZ == -1) textureId = getTexture(textureId)->id;
	uint tex[2] = {textureId, textureId};
	glBindTextures(0,2,tex);

	glBindSamplers(0,1,&theGraphicsState->samplers[SAMPLER_NORMAL]);
	glBindSamplers(1,1,&theGraphicsState->samplers[SAMPLER_NORMAL]);

	Vec2 quadUVs[] = { uv.bl(), uv.tl(), uv.tr(), uv.br() };
	Vec4 colors[] = { vec4(1), vec4(1), vec4(1), vec4(1), };

	pushUniform(SHADER_Quad, 0, "primitiveMode", true);
	pushUniform(SHADER_Quad, 0, "verts", verts, arrayCount(verts));
	pushUniform(SHADER_Quad, 0, "uvs", quadUVs, arrayCount(quadUVs));
	pushUniform(SHADER_Quad, 0, "colors", colors, arrayCount(colors));
	pushUniform(SHADER_Quad, 0, "texZ", texZ);
	pushUniform(SHADER_Quad, 0, "setColor", linearToGamma(color));

	glDrawArrays(GL_QUADS, 0, arrayCount(verts));
}

void ortho(Rect r) {
	r = r.cenDim();

	pushUniform(SHADER_Quad, 0, "camera", r);
}

void lookAt(Vec3 pos, Vec3 look, Vec3 up, Vec3 right) {
	Mat4 view;
	viewMatrix(&view, pos, look, up, right);

	pushUniform(SHADER_Cube, 0, "view", &view);
}

void perspective(float fov, float aspect, float n, float f) {
	Mat4 proj;
	projMatrix(&proj, fov, aspect, n, f);

	pushUniform(SHADER_Cube, 0, "proj", &proj);
}

//

void drawCube(Vec3 trans, Vec3 scale, Vec4 color, float degrees, Vec3 rot) {
	glBindTextures(0,1,&theGraphicsState->textureWhite->id);

	Mesh* cubeMesh = getMesh(MESH_Cube);
	glBindBuffer(GL_ARRAY_BUFFER, cubeMesh->bufferId);

	glVertexAttribPointer(0, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3) + sizeof(Vec2)));
	glEnableVertexAttribArray(2);

	Mat4 model = modelMatrix(trans, scale, degrees, rot);
	pushUniform(SHADER_Cube, 0, "model", &model);
	pushUniform(SHADER_Cube, 0, "setColor", linearToGamma(color));
	pushUniform(SHADER_Cube, 0, "mode", false);
	pushUniform(SHADER_Cube, 1, "texZ", -1.0f);


	glDrawArrays(GL_QUADS, 0, cubeMesh->vertCount);
	// glDrawElements(GL_QUADS, cubeMesh->elementCount, GL_UNSIGNED_INT, (void*)0);
}

void drawLine(Vec3 p0, Vec3 p1, Vec4 color) {

	// Disabling these arrays is very important.

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

	glBindTextures(0,1,&theGraphicsState->textureWhite->id);

	Vec3 verts[] = {p0, p1};
	Vec2 quadUVs[] = {{0,0}, {0,1}};
	pushUniform(SHADER_Cube, 0, "setUV", quadUVs, arrayCount(quadUVs));
	pushUniform(SHADER_Cube, 0, "vertices", verts, arrayCount(verts));
	pushUniform(SHADER_Cube, 0, "setColor", linearToGamma(color));
	pushUniform(SHADER_Cube, 0, "mode", true);
	pushUniform(SHADER_Cube, 1, "texZ", -1.0f);

	glDrawArrays(GL_LINES, 0, arrayCount(verts));
}

void drawQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, int textureId = 0, Rect uv = rect(0,0,1,1), float texZ = -1) {

	// Disabling these arrays is very important.

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	
	Vec3 verts[] = {p0, p1, p2, p3};

	if(texZ == -1 && textureId == 0) textureId = theGraphicsState->textureWhite->id;
	uint tex[2] = {textureId, textureId};

	glBindTextures(0,2,tex);
	// glBindSamplers(0, 1, theGraphicsState->samplers);

	// Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
	Vec2 quadUVs[] = { uv.bl(), uv.tl(), uv.tr(), uv.br() };
	pushUniform(SHADER_Cube, 0, "setUV", quadUVs, arrayCount(quadUVs));
	pushUniform(SHADER_Cube, 0, "vertices", verts, arrayCount(verts));
	pushUniform(SHADER_Cube, 0, "setColor", linearToGamma(color));
	pushUniform(SHADER_Cube, 1, "texZ", texZ);
	pushUniform(SHADER_Cube, 0, "mode", true);

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