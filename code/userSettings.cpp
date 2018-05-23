
// #define USE_DIRECT3D

#define USE_SRGB 1
const int INTERNAL_TEXTURE_FORMAT = USE_SRGB ? GL_SRGB8_ALPHA8 : GL_RGBA8;

#define COLOR_SRGB(color) \
	(theGraphicsState->useSRGB ? linearToGamma(color) : color);

#define APP_NAME "VoxelGame"

#define editor_executable_path "C:\\Program Files\\Sublime Text 3\\sublime_text.exe"

#define HOTRELOAD_SHADERS 1

#define App_Session_File ".\\session.tmp"
#define Game_Settings_File ".\\settings.tmp"

#ifdef SHIPPING_MODE
#define DATA_FOLDER(str) ".\\data\\" str
#else 
#define DATA_FOLDER(str) "..\\data\\" str
#endif

#define SAVES_FOLDER ".\\saves\\"
#define SAVE_STATE1 "saveState1.sav"

#define GUI_SETTINGS_FILE DATA_FOLDER("guiSettings.txt")

#define App_Font_Folder    DATA_FOLDER("Fonts\\")
#define App_Texture_Folder DATA_FOLDER("Textures\\")
#define App_Audio_Folder   DATA_FOLDER("Audio\\")

#define Windows_Font_Folder "\\Fonts\\"
#define Windows_Font_Path_Variable "windir"

#define MINECRAFT_TEXTURE_FOLDER DATA_FOLDER("Textures\\minecraft\\")

//

const char* watchFolders[] = {
	DATA_FOLDER("Textures\\misc\\"),
	DATA_FOLDER("Textures\\skyboxes\\"),
	DATA_FOLDER("Textures\\minecraft\\"),
};

struct AppSessionSettings {
	Rect windowRect;
};

void appWriteSessionSettings(char* filePath, AppSessionSettings* at) {
	writeDataToFile((char*)at, sizeof(AppSessionSettings), filePath);
}

void appReadSessionSettings(char* filePath, AppSessionSettings* at) {
	readDataFile((char*)at, filePath);
}

void saveAppSettings(AppSessionSettings at) {
	if(fileExists(App_Session_File)) {
		appWriteSessionSettings(App_Session_File, &at);
	}
}

//

enum TextureId {
	TEXTURE_WHITE = 0,
	TEXTURE_RECT,
	TEXTURE_CIRCLE,
	TEXTURE_TEST,
	TEXTURE_DESTROY_STAGES,
	TEXTURE_SIZE,
};

char* texturePaths[] = {
	DATA_FOLDER("Textures\\misc\\white.png"),
	DATA_FOLDER("Textures\\misc\\rect.png"),
	DATA_FOLDER("Textures\\misc\\circle.png"),
	DATA_FOLDER("Textures\\misc\\test.png"),
	DATA_FOLDER("Textures\\minecraft\\destroyStages.png"),
};

//

enum CubeMapIds {
	CUBEMAP_5 = 0,
	CUBEMAP_SIZE,
};

char* cubeMapPaths[] = {
	DATA_FOLDER("Textures\\skyboxes\\skybox1.png"),
};

//
// Shaders.
//

struct DVertex {
      Vec3 pos;
      Vec4 color;
      Vec2 uv;
};

struct Vertex {
	Vec3 pos;
	Vec2 uv;
	Vec3 normal;
};

struct MeshMap {
	Vertex* vertexArray;
	int size;
};

enum MeshId {
	MESH_CUBE = 0,
	MESH_QUAD,
	MESH_SIZE,
};

const Vertex cubeArray[] = {
	{vec3(-0.5f,-0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f,-0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f,-0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f,-0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f, 0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f, 0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f, 0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f,-0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f, 0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f, 0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f,-0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f, 0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f,-0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f,-0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f, 0.5f), vec2(1,0), vec3(0,0,1)},
};

const Vertex quadArray[] = {
	{vec3(-0.5f,-0.5f, 0), vec2(0,1), vec3(1,1,1)},
	{vec3(-0.5f, 0.5f, 0), vec2(0,0), vec3(1,1,1)},
	{vec3( 0.5f, 0.5f, 0), vec2(1,0), vec3(1,1,1)},
	{vec3( 0.5f,-0.5f, 0), vec2(1,1), vec3(1,1,1)},
};

MeshMap meshArrays[] = {
	{(Vertex*)cubeArray, sizeof(cubeArray)},
	{(Vertex*)quadArray, sizeof(quadArray)},
};

//

enum SamplerType {
	SAMPLER_NORMAL = 0,
	SAMPLER_VOXEL_1,
	SAMPLER_VOXEL_2,
	SAMPLER_VOXEL_3,
	SAMPLER_SIZE,
};

//

enum FrameBufferType {
	FRAMEBUFFER_3dMsaa = 0,
	FRAMEBUFFER_3dNoMsaa,
	FRAMEBUFFER_Reflection,
	FRAMEBUFFER_2d,

	FRAMEBUFFER_DebugMsaa,
	FRAMEBUFFER_DebugNoMsaa,

	FRAMEBUFFER_SIZE,
};

//

#define GLSL(src) "#version 430\n \
	#extension GL_ARB_bindless_texture 			: enable\n \
	#extension GL_ARB_shading_language_include 	: enable\n \
	#extension GL_ARB_uniform_buffer_object 	: enable\n \
	#extension GL_ARB_gpu_shader5               : enable\n \
	#extension GL_ARB_gpu_shader_fp64           : enable\n \
	#extension GL_ARB_shader_precision          : enable\n \
	#extension GL_ARB_conservative_depth        : enable\n \
	#extension GL_ARB_texture_cube_map_array    : enable\n \
	#extension GL_ARB_separate_shader_objects   : enable\n \
	#extension GL_ARB_shading_language_420pack  : enable\n \
	#extension GL_ARB_shading_language_packing  : enable\n \
	#extension GL_ARB_explicit_uniform_location : enable\n" #src

//

const char* vertexShaderCube = GLSL (
	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;
	smooth out vec2 uv;

	layout(location = 0) in vec3 attr_verts;
	layout(location = 1) in vec2 attr_texUV;
	layout(location = 2) in vec3 attr_normal;

	uniform mat4x4 model;
	uniform mat4x4 view;
	uniform mat4x4 proj;

	uniform bool mode;

	uniform vec3 vertices[24];
	uniform vec2 setUV[24];
	uniform vec4 setColor;

	void main() {
		Color = setColor;

		vec4 pos;
		if(mode) {
			pos = vec4(vertices[gl_VertexID], 1);
			gl_Position = proj*view*pos;
			uv = setUV[gl_VertexID];
		} else {
			pos = vec4(attr_verts, 1);
			gl_Position = proj*view*model*pos;
			uv = attr_texUV;
		}
	}
);

const char* fragmentShaderCube = GLSL (
	layout(binding = 0) uniform sampler2D s;
	layout(binding = 1) uniform sampler2DArray sArray[2];

	smooth in vec2 uv;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	uniform float texZ;
	uniform bool alphaTest = false;
	uniform float alpha = 0.5f;

	void main() {
		vec4 texColor;
		if(texZ != -1) texColor = texture(sArray[0], vec3(uv, floor(texZ)));
		else texColor = texture(s, uv);

		color = texColor * Color;

		if(alphaTest) {
			if(color.a <= alpha) discard;
		}
	}
);

//

const char* vertexShaderQuad = GLSL (
	const vec2 quad[] = vec2[] (
		vec2( -0.5f, -0.5f ),
		vec2( -0.5f,  0.5f ),
		vec2(  0.5f, -0.5f ),
		vec2(  0.5f,  0.5f )
		);

	const ivec2 quad_uv[] = ivec2[] (
		ivec2(  0.0,  0.0 ),
		ivec2(  0.0,  1.0 ),
		ivec2(  1.0,  0.0 ),
		ivec2(  1.0,  1.0 )
		);

	uniform vec4 setUV;
	uniform float texZ;
	uniform vec4 mod;
	uniform vec4 setColor;
	uniform vec4 camera; // left bottom right top

	uniform bool primitiveMode = false;
	uniform vec2 verts[32];
	uniform vec2 uvs[32];

	out gl_PerVertex { vec4 gl_Position; };
	smooth out vec3 uv;
	out vec4 Color;

	void main() {

		if(primitiveMode) {
			uv = vec3(uvs[gl_VertexID],texZ);

			Color = setColor;

			vec2 model = verts[gl_VertexID];
			vec2 view = model/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
			gl_Position = vec4(view, 0, 1);

		} else {
			
			ivec2 pos = quad_uv[gl_VertexID];
			uv = vec3(setUV[pos.x], setUV[2 + pos.y], texZ);
			vec2 v = quad[gl_VertexID];

			Color = setColor;

			vec2 model = v*mod.zw + mod.xy;
			vec2 view = model/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
			gl_Position = vec4(view, 0, 1);
		}

	}
);

const char* fragmentShaderQuad = GLSL (
	layout(binding = 0) uniform sampler2D s;
	layout(binding = 1) uniform sampler2DArray sArray[2];

	smooth in vec3 uv;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	void main() {
		vec4 texColor;
		if(uv.z > -1) texColor = texture(sArray[0], vec3(uv.xy, floor(uv.z)));
		else texColor = texture(s, uv.xy);

		color = texColor * Color;
	}
);

//

const char* vertexShaderFont = GLSL (
	const vec2 quad[] = vec2[] (
		vec2( -0.5f, -0.5f ),
		vec2( -0.5f,  0.5f ),
		vec2(  0.5f, -0.5f ),
		vec2(  0.5f,  0.5f )
		);

	const ivec2 quad_uv[] = ivec2[] (
		ivec2(  0.0,  0.0 ),
		ivec2(  0.0,  1.0 ),
		ivec2(  1.0,  0.0 ),
		ivec2(  1.0,  1.0 )
		);

	uniform vec4 setUV;
	uniform vec4 mod;
	uniform vec4 camera; // left bottom right top

	out gl_PerVertex { vec4 gl_Position; };
	smooth out vec2 uv;

	void main() {

		ivec2 pos = quad_uv[gl_VertexID];
		uv = vec2(setUV[pos.x], setUV[2 + pos.y]);
		vec2 v = quad[gl_VertexID];

		vec2 model = v*mod.zw + mod.xy;
		vec2 view = model/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
		gl_Position = vec4(view, 0, 1);

	}
);

const char* fragmentShaderFont = GLSL (
	layout(binding = 0) uniform sampler2D s;

	uniform float uvstep;

	smooth in vec2 uv;

	out vec4 color;

	// const float pixelFilter[] = float[] (
	//     1,2,3,4,5
 //    );

    const float fi[] = float[] ( 
        0.03f, 0.30f, 0.34f, 0.30f, 0.03f
    );

	void main() {

		vec3 cl = texture(s, uv - vec2(uvstep,0)).rgb;
		vec3 cr = texture(s, uv + vec2(uvstep,0)).rgb;
		vec3 cm = texture(s, uv).rgb;

		// cm.r = pow(cm.r, 1/2.2f);
		// cm.g = pow(cm.g, 1/2.2f);
		// cm.b = pow(cm.b, 1/2.2f);

		// cm.r = pow(cm.r, 2.2f);
		// cm.g = pow(cm.g, 2.2f);
		// cm.b = pow(cm.b, 2.2f);

		vec4 finalColor;
		// finalColor.r = cl.g*fi[0] + cl.b*fi[1] + cm.r*fi[2] + cm.g*fi[3] + cm.b*fi[4];
		// finalColor.g = cl.b*fi[0] + cm.r*fi[1] + cm.g*fi[2] + cm.b*fi[3] + cr.r*fi[4];
		// finalColor.b = cm.r*fi[0] + cm.g*fi[1] + cm.b*fi[2] + cr.r*fi[3] + cr.g*fi[4];

		finalColor = vec4(cm,1);

		color = finalColor;
	}
);

//

struct ParticleVertex {
	Mat4 m;
	Vec4 c;
};

const char* vertexShaderParticle = GLSL (
	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;
	out vec2 uv;

	// layout(location = 0) in vec3 attr_verts;
	// layout(location = 1) in vec2 attr_texUV;
	// // layout(location = 2) in vec3 attr_normal;
	// layout(location = 3) in vec4 attr_color;

	layout(location = 0) uniform vec3 vertices[4];
	layout(location = 4) uniform vec2 attr_texUV[4];

	layout(location = 8) in vec4 attr_color;
	layout(location = 9) in mat4x4 model;

	uniform mat4x4 view;
	uniform mat4x4 proj;

	void main() {
		gl_Position = proj*view*model*vec4(vertices[gl_VertexID], 1);
		// gl_Position = proj*view*vec4(vertices[gl_VertexID], 1);

		// gl_Position = proj*view*model*vec4(attr_verts, 1);
		uv = attr_texUV[gl_VertexID];
		Color = attr_color;
	}
);

const char* fragmentShaderParticle = GLSL (
	layout(binding = 0) uniform sampler2D s;

	in vec2 uv;
	in vec4 Color;

	out float gl_FragDepth;
	out vec4 color;

	// uniform float alphaTest = 0.5f;

	void main() {
		color = texture(s, uv) * Color;
		// color = texture(s, uv.xy);
		// color = Color;
		// color = vec4(1,0,0,1);

		// if(color.a <= alphaTest) discard;
	}
);

//

const char* vertexShaderCubeMap = GLSL (
	const vec3 cube[] = vec3[] (
	  vec3( -1.0f,  1.0f, -1.0f ),
	  vec3( -1.0f, -1.0f, -1.0f ),
	  vec3(  1.0f, -1.0f, -1.0f ),
	  vec3(  1.0f, -1.0f, -1.0f ),
	  vec3(  1.0f,  1.0f, -1.0f ),
	  vec3( -1.0f,  1.0f, -1.0f ),

	  vec3( -1.0f, -1.0f,  1.0f ),
	  vec3( -1.0f, -1.0f, -1.0f ),
	  vec3( -1.0f,  1.0f, -1.0f ),
	  vec3( -1.0f,  1.0f, -1.0f ),
	  vec3( -1.0f,  1.0f,  1.0f ),
	  vec3( -1.0f, -1.0f,  1.0f ),

	  vec3(  1.0f, -1.0f, -1.0f ),
	  vec3(  1.0f, -1.0f,  1.0f ),
	  vec3(  1.0f,  1.0f,  1.0f ),
	  vec3(  1.0f,  1.0f,  1.0f ),
	  vec3(  1.0f,  1.0f, -1.0f ),
	  vec3(  1.0f, -1.0f, -1.0f ),

	  vec3( -1.0f, -1.0f,  1.0f ),
	  vec3( -1.0f,  1.0f,  1.0f ),
	  vec3(  1.0f,  1.0f,  1.0f ),
	  vec3(  1.0f,  1.0f,  1.0f ),
	  vec3(  1.0f, -1.0f,  1.0f ),
	  vec3( -1.0f, -1.0f,  1.0f ),

	  vec3( -1.0f,  1.0f, -1.0f ),
	  vec3(  1.0f,  1.0f, -1.0f ),
	  vec3(  1.0f,  1.0f,  1.0f ),
	  vec3(  1.0f,  1.0f,  1.0f ),
	  vec3( -1.0f,  1.0f,  1.0f ),
	  vec3( -1.0f,  1.0f, -1.0f ),

	  vec3( -1.0f, -1.0f, -1.0f ),
	  vec3( -1.0f, -1.0f,  1.0f ),
	  vec3(  1.0f, -1.0f, -1.0f ),
	  vec3(  1.0f, -1.0f, -1.0f ),
	  vec3( -1.0f, -1.0f,  1.0f ),
	  vec3(  1.0f, -1.0f,  1.0f )
	);

	out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[]; };

	uniform mat4x4 view;
	uniform mat4x4 proj;

	smooth out vec3 pos;

	uniform bool clipPlane = false;
	uniform vec4 cPlane;

	void main() {
		pos = cube[gl_VertexID];

		if(clipPlane) {
			gl_ClipDistance[0] = dot(cPlane, vec4(pos,1));
			// pos.z *= -1;
			// pos.y *= -1;
		}

		gl_Position = proj*view*vec4(pos,1);
	}
);

const char* fragmentShaderCubeMap = GLSL (
	layout(depth_less) out float gl_FragDepth;
	layout(binding = 0) uniform samplerCubeArray s;
	smooth in vec3 pos;

	out vec4 color;

	uniform bool clipPlane = false;

	uniform vec4 fogColor;

	float mapRange01(float value, float min, float max) {
		float off = min < 0 ? abs(min) : -min;
		return ((value+off)/((max+off)-(min+off)));
	};

	void main() {
		vec3 clipPos = pos;
		if(clipPlane) clipPos.y *= -1;
		color = texture(s, vec4(clipPos, 0));

		float d0 = -0.01f;
		if(clipPos.y <= 0) {
			vec4 c = texture(s, vec4(clipPos, 0));

			if(clipPos.y >= d0) {
				float f = mapRange01(clipPos.y, d0, 0);
				color = mix(fogColor, c, f);

			} else color = fogColor;

		} else color = texture(s, vec4(clipPos, 0));
	}
);

enum ShaderProgram {
	SHADER_CUBE = 0,
	SHADER_QUAD,
	SHADER_FONT,
	SHADER_PARTICLE,
	SHADER_CUBEMAP,
	SHADER_VOXEL,

	SHADER_SIZE,
};

MakeShaderInfo makeShaderInfo[SHADER_SIZE] = {
	{(char*)vertexShaderCube, (char*)fragmentShaderCube},
	{(char*)vertexShaderQuad, (char*)fragmentShaderQuad},
	{(char*)vertexShaderFont, (char*)fragmentShaderFont},
	{(char*)vertexShaderParticle, (char*)fragmentShaderParticle},
	{(char*)vertexShaderCubeMap, (char*)fragmentShaderCubeMap},
	{(char*)stbvox_get_vertex_shader(), (char*)stbvox_get_fragment_shader()},
};

//

#define HLSL(src) "" #src

const char* d3dShader = HLSL (

    struct PSInput {
          float4 pos : SV_POSITION;
          float4 color : COLOR;
          float2 uv : TEXCOORD;
    };

    PSInput vertexShader(float3 pos : POSITION, float4 color : COLOR, float2 uv : TEXCOORD ) {
        PSInput output;
        
    	output.pos = float4(pos,1);
    	output.color = color;
    	output.uv = uv;
        
        return output;
    }

    Texture2D simpleTexture : register(t0);
    SamplerState simpleSampler : register(s0);

    float4 pixelShader(PSInput input) : SV_Target {
        float4 output;

        output = float4(simpleTexture.Sample(simpleSampler, input.uv),1);

        return output;
    }
);

// &lt;pre&gt;matrix World;
// matrix View;
// matrix Projection;
 
// struct PS_INPUT
// {
//     float4 Pos : SV_POSITION;
//     float4 Color : COLOR0;
// };
 
// PS_INPUT VS( float4 Pos : POSITION, float4 Color : COLOR )
// {
//     PS_INPUT psInput;
 
//     Pos = mul( Pos, World );
//     Pos = mul( Pos, View );
 
//     psInput.Pos = mul( Pos, Projection );
//     psInput.Color = Color;
 
//     return psInput;
// }
 
// float4 PS( PS_INPUT psInput ) : SV_Target
// {
//     return psInput.Color;
// }
 
// technique10 Render
// {
//     pass P0
//     {
//         SetVertexShader( CompileShader( vs_4_0, VS() ) );
//         SetGeometryShader( NULL );
//         SetPixelShader( CompileShader( ps_4_0, PS() ) );
//     }
// }