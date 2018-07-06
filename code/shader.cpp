
#define GLSL(src) "#version 410\n \
	#extension GL_ARB_bindless_texture 			    : enable\n \
	#extension GL_ARB_shading_language_include 	: enable\n \
	#extension GL_ARB_uniform_buffer_object 	  : enable\n \
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

struct Vertex2d {
	Vec2 pos;
	Vec2 uv;
	Vec4 color;
};

Vertex2d vertex2d(Vec2 p) { return {p, {}, {1,1,1,1}}; };

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
	uniform vec4 colors[32];

	//

	out gl_PerVertex { vec4 gl_Position; };
	smooth out vec3 UV;
	out vec4 Color;

	void main() {

		if(primitiveMode) {
			vec2 pos = verts[gl_VertexID];
			UV = vec3(uvs[gl_VertexID],texZ);
			Color = colors[gl_VertexID] * setColor;

			vec2 view = pos/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
			gl_Position = vec4(view, 0, 1);

		} else {
			ivec2 pos = quad_uv[gl_VertexID];
			UV = vec3(setUV[pos.x], setUV[2 + pos.y], texZ);
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

	smooth in vec3 UV;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	void main() {
		vec4 texColor;
		if(UV.z > -1) texColor = texture(sArray[0], vec3(UV.xy, floor(UV.z)));
		else texColor = texture(s, UV.xy);

		color = texColor * Color;
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

//

#define SHADERLIST \
	SHADERFUNC(Cube) \
	SHADERFUNC(Quad) \
	SHADERFUNC(CubeMap)

//

#define SHADERFUNC(name) SHADER_##name,
enum ShaderProgram {
	SHADER_START = -1,
	SHADERLIST
	SHADER_Voxel,

	SHADER_SIZE,
};
#undef SHADERFUNC

#define SHADERFUNC(name) {(char*)vertexShader##name, (char*)fragmentShader##name},
MakeShaderInfo makeShaderInfo[SHADER_SIZE] = {
	SHADERLIST
	{(char*)stbvox_get_vertex_shader(), (char*)stbvox_get_fragment_shader()},
};
#undef SHADERFUNC