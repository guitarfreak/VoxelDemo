
#define Font_Error_Glyph (int)0x20-1

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

	bool isSubPixel;
};

enum TextStatus {
	TEXTSTATUS_END = 0, 
	TEXTSTATUS_NEWLINE, 
	TEXTSTATUS_WRAPPED, 
	TEXTSTATUS_DEFAULT, 
	TEXTSTATUS_SIZE, 
};

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

enum {
	TEXT_MARKER_BOLD = 0,
	TEXT_MARKER_ITALIC,
	TEXT_MARKER_COLOR,
};

#define Marker_Size 3
#define Bold_Marker "<b>"
#define Italic_Marker "<i>"
#define Color_Marker "<c>"

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

enum {
	TEXTSHADOW_MODE_NOSHADOW = 0,
	TEXTSHADOW_MODE_SHADOW,
	TEXTSHADOW_MODE_OUTLINE,
};