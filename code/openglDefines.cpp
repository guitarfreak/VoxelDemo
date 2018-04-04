
#define GL_TEXTURE_CUBE_MAP_SEAMLESS      0x884F
#define GL_FRAMEBUFFER_SRGB               0x8DB9
#define GL_FRAMEBUFFER_SRGB               0x8DB9
#define GL_TEXTURE_BUFFER                 0x8C2A
#define GL_MAP_WRITE_BIT                  0x0002
#define GL_MAP_PERSISTENT_BIT             0x0040
#define GL_MAP_COHERENT_BIT               0x0080
#define GL_VERTEX_SHADER                  0x8B31
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER_BIT              0x00000001
#define GL_FRAGMENT_SHADER_BIT            0x00000002
#define GL_DEBUG_OUTPUT                   0x92E0
#define WGL_CONTEXT_FLAGS_ARB             0x2094
#define WGL_CONTEXT_DEBUG_BIT_ARB         0x0001
#define WGL_CONTEXT_MAJOR_VERSION_ARB     0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB     0x2092
#define GL_MAJOR_VERSION                  0x821B
#define GL_MINOR_VERSION                  0x821C
#define GL_RGB32F                         0x8815
#define GL_RGBA8I                         0x8D8E
#define GL_RGBA8UI                        0x8D7C
#define GL_R8                             0x8229
#define GL_ARRAY_BUFFER                   0x8892

#define GL_NUM_EXTENSIONS                 0x821D

#define GL_CLAMP_TO_EDGE                  0x812F
#define GL_TEXTURE_MAX_ANISOTROPY_EXT     0x84FE
#define GL_TEXTURE_WRAP_R                 0x8072
#define GL_STATIC_DRAW                    0x88E4
#define GL_SRGB8_ALPHA8                   0x8C43
#define GL_TEXTURE_CUBE_MAP_ARRAY         0x9009
#define GL_STREAM_DRAW                    0x88E0
#define GL_TEXTURE_2D_ARRAY               0x8C1A
#define GL_FRAMEBUFFER                    0x8D40
#define GL_COLOR_ATTACHMENT0              0x8CE0
#define GL_STENCIL_ATTACHMENT             0x8D20
#define GL_DEPTH_ATTACHMENT               0x8D00
#define GL_DEPTH_STENCIL_ATTACHMENT       0x821A
#define GL_RENDERBUFFER                   0x8D41
#define GL_DEPTH_STENCIL                  0x84F9
#define GL_DEPTH24_STENCIL8               0x88F0
#define GL_DEBUG_OUTPUT_SYNCHRONOUS       0x8242
#define GL_DEBUG_SEVERITY_HIGH            0x9146
#define GL_DEBUG_SEVERITY_MEDIUM          0x9147
#define GL_DEBUG_SEVERITY_LOW             0x9148
#define GL_DEBUG_SEVERITY_NOTIFICATION    0x826B
#define GL_MULTISAMPLE                    0x809D
#define GL_FUNC_ADD                       0x8006
#define GL_CLIP_DISTANCE0                 0x3000
#define GL_CLIP_DISTANCE1                 0x3001
#define GL_FUNC_ADD                       0x8006
#define GL_MAX                            0x8008
typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;


#define makeGLFunction(returnType, name, ...) \
	typedef returnType WINAPI name##Function(__VA_ARGS__); \
	name##Function* gl##name; 
#define loadGLFunction(name) \
	gl##name = (name##Function*)wglGetProcAddress("gl" #name);

#define GL_FUNCTION_LIST \
	GLOP(void, CreateVertexArrays, GLsizei n, GLuint *arrays) \
	GLOP(void, BindVertexArray, GLuint array) \
	GLOP(void, CreateBuffers, GLsizei n, GLuint *buffers) \
	GLOP(void, CreateTextures, GLenum target, GLsizei n, GLuint *textures) \
	GLOP(void, NamedBufferStorage, GLuint buffer, GLsizei size, const void *data, GLbitfield flags) \
	GLOP(void*, MapNamedBufferRange, GLuint buffer, GLint* offset, GLsizei length, GLbitfield access) \
	GLOP(void, TextureBuffer, GLuint texture, GLenum internalformat, GLuint buffer) \
	GLOP(uint, CreateShaderProgramv, GLenum type, GLsizei count, const char **strings) \
	GLOP(void, CreateProgramPipelines, GLsizei n, GLuint *pipelines) \
	GLOP(void, UseProgramStages, GLuint pipeline, GLbitfield stages, GLuint program) \
	GLOP(void, BindTextures, GLuint first, GLsizei count, const GLuint *textures) \
	GLOP(void, BindProgramPipeline, GLuint pipeline) \
	GLOP(void, DrawArraysInstancedBaseInstance, GLenum mode, GLint first, GLsizei count, GLsizei primcount, GLuint baseinstance) \
	GLOP(GLuint, GetDebugMessageLog, GLuint counter, GLsizei bufSize, GLenum *source, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, char *messageLog) \
	GLOP(void, TextureStorage3D, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth) \
	GLOP(void, TextureSubImage3D, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels) \
	GLOP(void, GenerateTextureMipmap, GLuint texture) \
	GLOP(void, TextureStorage2D, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) \
	GLOP(void, TextureSubImage2D, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) \
	GLOP(void, CreateSamplers, GLsizei n, GLuint *samplers) \
	GLOP(void, SamplerParameteri, GLuint sampler, GLenum pname, GLint param) \
	GLOP(void, BindSamplers, GLuint first, GLsizei count, const GLuint *samplers) \
	GLOP(void, TextureParameteri, GLuint texture, GLenum pname, GLint param) \
	GLOP(GLint, GetUniformLocation, GLuint program, const GLchar *name) \
	GLOP(void, ProgramUniform4f, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2,GLfloat v3) \
	GLOP(void, ProgramUniform3f, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2) \
	GLOP(void, ProgramUniform2f, GLuint program, GLint location, GLfloat v0, GLfloat v1) \
	GLOP(void, ProgramUniform1f, GLuint program, GLint location, GLfloat v0) \
	GLOP(void, ProgramUniform1i, GLuint program, GLint location, GLuint v0) \
	GLOP(void, ProgramUniform1fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
	GLOP(void, ProgramUniform2fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
	GLOP(void, ProgramUniform1iv, GLuint program, GLint location, GLsizei count, const GLint *value) \
	GLOP(void, ProgramUniformMatrix4fv, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) \
	GLOP(void, ProgramUniform4fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
	GLOP(void, UseProgram, GLuint program) \
	GLOP(void, ActiveTexture, GLenum texture) \
	GLOP(void, EnableVertexAttribArray, GLuint index) \
	GLOP(void, DisableVertexAttribArray, GLuint index) \
	GLOP(void, TexImage3DEXT,  GLenum	target, GLint	level, GLenum	internalformat, GLsizei	width, GLsizei	height, GLsizei	depth, GLint	border, GLenum	format, GLenum	type, const	GLvoid *pixels) \
	GLOP(void, ActiveTextureARB, GLenum texture) \
	GLOP(void, VertexAttribIPointer, GLuint index, GLint size, GLenum type, GLsizei stride, const GLvoid * pointer) \
	GLOP(void, BindBuffer, GLenum target, GLuint buffer) \
	GLOP(void, BindBufferARB, GLenum target, GLuint buffer) \
	GLOP(void, GenBuffersARB, GLsizei n, GLuint * buffers) \
	GLOP(void, NamedBufferData, GLuint buffer, GLsizeiptr size, const GLvoid * data, GLenum usage) \
	GLOP(void, BufferDataARB, GLenum target, GLsizeiptr size, const GLvoid * data, GLenum usage) \
	GLOP(void, TexBufferARB, GLenum target, GLenum internalFormat, GLuint buffer) \
	GLOP(void, UniformMatrix4fv, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) \
	GLOP(void, VertexAttribPointer, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid * pointer) \
	GLOP(void, ProgramUniform3fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
	GLOP(GLint, GetAttribLocation, GLuint program, const GLchar *name) \
	GLOP(void, GenSamplers, GLsizei n, GLuint *samplers) \
	GLOP(void, BindSampler, GLuint unit, GLuint sampler) \
	GLOP(void, BindTextureUnit, GLuint unit, GLuint texture) \
	GLOP(void, NamedBufferSubData, GLuint buffer, GLintptr offset, GLsizei size, const void *data) \
	GLOP(void, NamedBufferSubDataEXT, GLuint buffer, GLintptr offset, GLsizei size, const void *data) \
	GLOP(void, GetUniformiv, GLuint program, GLint location, GLint * params) \
	GLOP(void, GetUniformfv, GLuint program, GLint location, GLfloat * params) \
	GLOP(void, CreateFramebuffers, GLsizei n, GLuint *framebuffers) \
	GLOP(void, NamedFramebufferParameteri, GLuint framebuffer, GLenum pname, GLint param) \
	GLOP(void, NamedRenderbufferStorageMultisample, GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height) \
	GLOP(void, CreateRenderbuffers, GLsizei n, GLuint *renderbuffers) \
	GLOP(void, NamedFramebufferRenderbuffer, GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) \
	GLOP(void, BindFramebuffer, GLenum target, GLuint framebuffer) \
	GLOP(void, NamedFramebufferDrawBuffer, GLuint framebuffer, GLenum buf) \
	GLOP(void, BlitFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) \
	GLOP(void, BlitNamedFramebuffer, GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) \
	GLOP(void, NamedFramebufferTexture, GLuint framebuffer, GLenum attachment, GLuint texture, GLint level) \
	GLOP(void, TextureStorage2DMultisample, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) \
	GLOP(void, TexImage2DMultisample, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) \
	GLOP(void, NamedRenderbufferStorage, GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height) \
	GLOP(GLenum, CheckNamedFramebufferStatus, GLuint framebuffer, GLenum target) \
	GLOP(void, GenFramebuffers, GLsizei n, GLuint *ids) \
	GLOP(void, FramebufferTexture, GLenum target, GLenum attachment, GLuint texture, GLint level) \
	GLOP(void, BlendFuncSeparate, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) \
	GLOP(void, BlendEquation, GLenum mode) \
	GLOP(void, BlendEquationSeparate, GLenum modeRGB, GLenum modeAlpha) \
	GLOP(void, GetTextureSubImage, uint texture, int level, int xoffset, int yoffset, int zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void *pixels) \
	GLOP(GLubyte*, GetStringi, GLenum name, GLuint index) \
	GLOP(void, DrawArraysInstanced, GLenum mode, GLint first, GLsizei count, GLsizei primcount) \
	GLOP(void, VertexAttribDivisor, GLuint index, GLuint divisor)

#define GLOP(returnType, name, ...) makeGLFunction(returnType, name, __VA_ARGS__) 
	GL_FUNCTION_LIST
#undef GLOP



// typedef HGLRC wglCreateContextAttribsARBFunction(HDC hDC, HGLRC hshareContext, const int *attribList);
// wglCreateContextAttribsARBFunction* wglCreateContextAttribsARB;
typedef int WINAPI wglGetSwapIntervalEXTFunction(void);
wglGetSwapIntervalEXTFunction* wglGetSwapIntervalEXT;
typedef int WINAPI wglSwapIntervalEXTFunction(int);
wglSwapIntervalEXTFunction* wglSwapIntervalEXT;



void loadFunctions() {
#define GLOP(returnType, name, ...) loadGLFunction(name)
	GL_FUNCTION_LIST

	wglGetSwapIntervalEXT = (wglGetSwapIntervalEXTFunction*)wglGetProcAddress("wglGetSwapIntervalEXT");
	wglSwapIntervalEXT = (wglSwapIntervalEXTFunction*)wglGetProcAddress("wglSwapIntervalEXT");
#undef GLOP
}

void printGlExtensions() {
	for(int i = 0; i < GL_NUM_EXTENSIONS; i++) {
		char* s = (char*)glGetStringi(GL_EXTENSIONS, i);
		printf("%s\n", s);
	}
}