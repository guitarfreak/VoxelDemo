#pragma once
#include <math.h>

#define M_E         2.7182818284590452354               ///< e
#define M_LOG2E     1.4426950408889634074               ///< log_2 e
#define M_LOG10E    0.43429448190325182765              ///< log_10 e
#define M_LN2       0.69314718055994530942              ///< log_e 2
#define M_LN10      2.30258509299404568402              ///< log_e 10
#define M_2PI       6.2831853071795864769252867665590   ///< 2*pi
#define M_PI        3.1415926535897932384626433832795   ///< pi
#define M_3PI_2		4.7123889803846898576939650749193	///< 3/2*pi
#define M_PI_2      1.5707963267948966192313216916398   ///< pi/2
#define M_PI_4      0.78539816339744830962              ///< pi/4
#define M_1_PI      0.31830988618379067154              ///< 1/pi
#define M_2_PI      0.63661977236758134308              ///< 2/pi
#define M_2_SQRTPI  1.12837916709551257390              ///< 2/sqrt(pi)
#define M_SQRT2     1.41421356237309504880              ///< sqrt(2)
#define M_SQRT1_2   0.70710678118654752440              ///< 1/sqrt(2)
#define M_PI_180    0.0174532925199432957692369076848   ///< pi/180
#define M_180_PI    57.295779513082320876798154814105   ///< 180/pi
//#define M_BOLTZ		1.3806503e-23						///< boltzmann constant
#define M_BOLTZ		1.44269504							///< boltzmann constant

#define swap(type, a, b) 	\
		type swap##type = a;		\
		a = b;							\
		b = swap##type;

inline float mapRange(float value, float min, float max, float rangeMin, float rangeMax) {
	float off = min < 0 ? abs(min) : -abs(min);
	float result = ((value+off)/((max+off)-(min+off))) * 
				  (rangeMax-rangeMin) + rangeMin;

	return result;
};

int mod(int a, int b) {
	int result;
	result = a % b;
	if(result < 0) result += b;
	return result;
}

inline float min(float a, float b) {
	return a <= b ? a : b;
}

inline float min(float a, float b, float c) {
	return min(min(a, b), min(b, c));
}

inline float max(float a, float b) {
	return a >= b ? a : b;
}

inline float max(float a, float b, float c) {
	return max(max(a, b), max(b, c));
}

inline int maxReturnIndex(float a, float b, float c) {
	float result = max(a,b,c);
	int index;
	if(result == a) index = 0;
	else if(result == b) index = 1;
	else index = 2;
	return index;
}

inline float clampMin(float min, float a) {
	return a < min ? min : a;
}

inline float clampMax(float a, float max) {
	return a > max ? max : a;
}

inline float clamp(float n, float min, float max) {
	float result = clampMax(clampMin(min, n), max);
	return result;
};

inline void clamp(float* n, float min, float max) {
	*n = clampMax(clampMin(min, *n), max);
};

inline bool valueBetween(float v, float min, float max) {
	bool result = (v >= min && v <= max);
	return result;
}

inline bool valueBetween2(float v, float min, float max) {
	bool result = (v > min && v <= max);
	return result;
}

inline float roundFloat(float f, int x) {
	return floor(f*x + 0.5) / x;
	// 1.7*3 + 0.5 / 3
};

/** a roughly b */
inline bool roughlyEqual(float a, float b, float margin) {
	return (a > b - margin) && (a < b + margin);
};

inline float radianToDegree(float angle) {
	return angle*((float)180 / M_PI);
}

inline float degreeToRadian(float angle) {
	return angle*((float)M_PI / 180);
}

inline float sign(float p1[2], float p2[2], float p3[2]) {
	return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
}

inline int sign(float x) {
	int s;
	if(x < 0) s = -1;
	if(x > 0) s = 1;
	else s = 0;

	return s; 
}	

inline bool pointIsLeftOfLine(float a[2], float b[2], float c[2]) {
	return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0;
}

inline int round(int i) {
	return (int)floor(i + 0.5f);
}

inline int colorFloatToInt(float color) {
	return (int)round(color * 255);
};

inline float colorIntToFloat(int color) {
	return ((float)1 / 255) * color;
};

inline uint mapRGB(int r, int g, int b) {
	return (r << 16 | g << 8 | b);
};

uint mapRGBA(float r, float g, float b, float a) {
	int ri = colorFloatToInt(r);
	int gi = colorFloatToInt(g);
	int bi = colorFloatToInt(b);
	int ai = colorFloatToInt(a);
	return (ri << 24 | gi << 16 | bi << 8 | ai);
}

inline uint mapRGB(float r, float g, float b) {
	return mapRGB(colorFloatToInt(r), colorFloatToInt(g), colorFloatToInt(b));
}

inline uint mapRGB(float color[3]) {
	return mapRGB(color[0], color[1], color[2]);
}

inline uint mapRGBA(float color[4]) {
	return mapRGBA(color[0], color[1], color[2], color[3]);
}

inline uint mapRGBA(float c) {
	return mapRGBA(c,c,c,c);
}

inline void colorAddAlpha(uint &color, float alpha) {
	color = (color << 8) + colorFloatToInt(alpha);
}

void rgbToHsl(float color[3], double r, double g, double b) {
	//NOTE: color[3] = {h, s, l}

	double M = 0.0, m = 0.0, c = 0.0;
	M = max(r, g, b);
	m = min(r, g, b);
	c = M - m;
	color[2] = 0.5 * (M + m);
	if (c != 0.0)
	{
		if (M == r)
		{
			color[0] = fmod(((g - b) / c), 6.0);
		}
		else if (M == g)
		{
			color[0] = ((b - r) / c) + 2.0;
		}
		else/*if(M==b)*/
		{
			color[0] = ((r - g) / c) + 4.0;
		}
		color[0] *= 60.0;
		color[1] = c / (1.0 - fabs(2.0 * color[2] - 1.0));
	}
	else
	{
		color[0] = 0;
		color[1] = 0;
		color[2] = r;
	}
}

void vSet3(float res[3], float x, float y, float z);
void hslToRgb(float color[3], double h, double s, double l) {
	double c = 0.0, m = 0.0, x = 0.0;
	c = (1.0 - fabs(2 * l - 1.0)) * s;
	m = 1.0 * (l - 0.5 * c);
	x = c * (1.0 - fabs(fmod(h / 60.0, 2) - 1.0));
	if (h == 360) h = 0;
	if (h >= 0.0 && h < 60)  vSet3(color, c + m, x + m, m);
	else if (h >= 60 && h < 120) vSet3(color, x + m, c + m, m);
	else if (h >= 120 && h < 180) vSet3(color, m, c + m, x + m);
	else if (h >= 180 && h < 240) vSet3(color, m, x + m, c + m);
	else if (h >= 240 && h < 300) vSet3(color, x + m, m, c + m);
	else if (h >= 300 && h < 360) vSet3(color, c + m, m, x + m);
	else vSet3(color, m, m, m);
}

void hueToRgb(float color[3], double h) {
	if (h >= 0.0 && h < 60)  {
		float x = mapRange(h, 0, 60, 0, 1);
		vSet3(color, 1, x, 0);
	}
	else if (h >= 60 && h < 120) {
		float x = mapRange(h, 60, 120, 1, 0);
		vSet3(color, x, 1, 0);
	}
	else if (h >= 120 && h < 180) {
		float x = mapRange(h, 120, 180, 0, 1);
		vSet3(color, 0, 1, x);
	}
	else if (h >= 180 && h < 240) {
		float x = mapRange(h, 180, 240, 1, 0);
		vSet3(color, 0, x, 1);
	}
	else if (h >= 240 && h < 300) {
		float x = mapRange(h, 240, 300, 0, 1);
		vSet3(color, x, 0, 1);
	}
	else if (h >= 300 && h < 360) {
		float x = mapRange(h, 300, 360, 1, 0);
		vSet3(color, 1, 0, x);
	}
	else vSet3(color, 1, 0, 0);
}

float colorGet(uint color, int i) {
	if (i < 0 || i > 3) {
		printf("colorGet wants to access element: %f", i);
		return -1;
	}
	int colorChannel = color >> ((3 - i) * 8) & 255;
	float colorChannelFloat = colorIntToFloat(colorChannel);
	return colorChannelFloat;
}

void colorGetRGB(uint color, float v[3]) {
	v[0] = colorIntToFloat(color >> 24 & 255);
	v[1] = colorIntToFloat(color >> 16 & 255);
	v[2] = colorIntToFloat(color >> 8 & 255);
}

void colorGetRGBA(uint color, float v[4]) {
	v[0] = colorIntToFloat(color >> 24 & 255);
	v[1] = colorIntToFloat(color >> 16 & 255);
	v[2] = colorIntToFloat(color >> 8 & 255);
	v[3] = colorIntToFloat(color & 255);
}

void colorAdd(uint &c1, uint c2) {
	float v1[4], v2[4];
	colorGetRGBA(c1, v1);
	colorGetRGBA(c2, v2);
	for (int i = 0; i < 3; ++i) {
		v1[i] += v2[i];
		v1[i] = clamp(v1[i], 0, 1);
	}
	c1 = mapRGBA(v1);
}

int colorSet(uint &color, int i, float f) {
	if (i < 0 || i > 3) {
		printf("colorGet wants to access element: %f", i);
		return -1;
	}

	float vec[4];
	colorGetRGBA(color,vec);
	vec[i] = f;
	color = mapRGBA(vec);
	return 0;
}

void colorSetRGB(uint &color, float r, float g, float b) {
	color = mapRGBA(r, g, b, colorGet(color, 3));
}

void colorSetRGB(uint &color, float rgb[3]) {
	colorSetRGB(color, rgb[0], rgb[1], rgb[2]);
}

void colorSetAlpha(uint &color, float a) {
	color = (color ^ (color & 255)) | colorFloatToInt(a);
}

float linearTween(float t, float b, float c, float d) {
	return c*t / d + b;
};

float easeInQuad(float t, float b, float c, float d) {
	t /= d;
	return c*t*t + b;
};

float easeOutQuad(float t, float b, float c, float d) {
	t /= d;
	return -c * t*(t - 2) + b;
};

float easeInOutQuad(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t + b;
	t--;
	return -c / 2 * (t*(t - 2) - 1) + b;
};

float easeInCubic(float t, float b, float c, float d) {
	t /= d;
	return c*t*t*t + b;
};

float easeOutCubic(float t, float b, float c, float d) {
	t /= d;
	t--;
	return c*(t*t*t + 1) + b;
};

float easeInOutCubic(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t*t + b;
	t -= 2;
	return c / 2 * (t*t*t + 2) + b;
};

float easeInQuart(float t, float b, float c, float d) {
	t /= d;
	return c*t*t*t*t + b;
};

float easeOutQuart(float t, float b, float c, float d) {
	t /= d;
	t--;
	return -c * (t*t*t*t - 1) + b;
};

float easeInOutQuart(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t*t*t + b;
	t -= 2;
	return -c / 2 * (t*t*t*t - 2) + b;
};

float easeInQuint(float t, float b, float c, float d) {
	t /= d;
	return c*t*t*t*t*t + b;
};

float easeOutQuint(float t, float b, float c, float d) {
	t /= d;
	t--;
	return c*(t*t*t*t*t + 1) + b;
};

float easeInOutQuint(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * t*t*t*t*t + b;
	t -= 2;
	return c / 2 * (t*t*t*t*t + 2) + b;
};

float easeInSine(float t, float b, float c, float d) {
	return -c * cos(t / d * M_PI_2) + c + b;
};

float easeOutSine(float t, float b, float c, float d) {
	return c * sin(t / d * M_PI_2) + b;
};

float easeInOutSine(float t, float b, float c, float d) {
	return -c / 2 * (cos(M_PI*t / d) - 1) + b;
};

float easeInExpo(float t, float b, float c, float d) {
	return c * pow(2, 10 * (t / d - 1)) + b;
};

float easeOutExpo(float t, float b, float c, float d) {
	return c * (-pow(2, -10 * t / d) + 1) + b;
};

float easeInOutExpo(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return c / 2 * pow(2, 10 * (t - 1)) + b;
	t--;
	return c / 2 * (-pow(2, -10 * t) + 2) + b;
};

float easeInCirc(float t, float b, float c, float d) {
	t /= d;
	return -c * (sqrt(1 - t*t) - 1) + b;
};

float easeOutCirc(float t, float b, float c, float d) {
	t /= d;
	t--;
	return c * sqrt(1 - t*t) + b;
};

float easeInOutCirc(float t, float b, float c, float d) {
	t /= d / 2;
	if (t < 1) return -c / 2 * (sqrt(1 - t*t) - 1) + b;
	t -= 2;
	return c / 2 * (sqrt(1 - t*t) + 1) + b;
};

int randomInt(int from, int to) {
	return rand() % (to - from + 1) + from;
};

float randomFloat(float from, float to, float offset) {
	float oneOverOffset = 1/offset;
	from *= oneOverOffset;
	to *= oneOverOffset;
	int rInt = randomInt(from, to);
	// float result = (float)(rand() % (int)((to - from + 1) + from)) * offset;
	float result = (float)rInt * offset;
	return result;
};

inline double mapRangePrecise(double value, double min, double max, double rangeMin, double rangeMax) {
	double offset = 0;
	const double width = max - min;
	const double rangeWidth = rangeMax - rangeMin;
	if (min < 0) offset = abs(min);
	if (min > 0) offset = -abs(min);
	value += offset;
	min += offset;
	max += offset;
	return ((double)value / width)*rangeWidth + rangeMin;
};

// angle is radian
float vectorToAngle(float vector[2]) {
	float angle = atan2(vector[1], vector[0]);
	if (angle < 0) angle = M_2PI - abs(angle);
	return angle;
}

bool pointInTriangle(float pt[2], float v1[2], float v2[2], float v3[2]) {
	bool b1, b2, b3;

	b1 = sign(pt, v1, v2) < 0.0f;
	b2 = sign(pt, v2, v3) < 0.0f;
	b3 = sign(pt, v3, v1) < 0.0f;

	return ((b1 == b2) && (b2 == b3));
}

//
// Vectors
//

// inline float* vec4Init(float a, float b, float c, float d)
// {
// 	float v[4] = { a, b, c, d };
// 	return v;
// }

inline void vSet4(float res[4], float a, float b, float c, float d) {
	res[0] = a;
	res[1] = b;
	res[2] = c;
	res[3] = d;
}

inline void vSet3(float res[3], float x, float y, float z) {
	res[0] = x;
	res[1] = y;
	res[2] = z;
}

inline void vSet3(float res[3], const float vec[3]) {
	res[0] = vec[0];
	res[1] = vec[1];
	res[2] = vec[2];
}

inline void vSet2(float res[2], const float vec[2]) {
	res[0] = vec[0];
	res[1] = vec[1];
}

inline void vSet2(float res[2], float x, float y) {
	res[0] = x;
	res[1] = y;
}

inline void vSet4(float res[4], const float vec[4]) {
	res[0] = vec[0];
	res[1] = vec[1];
	res[2] = vec[2];
	res[3] = vec[3];
}

inline void vSub3(float a_less_b[3], const float a[3], const float b[3]) {
	a_less_b[0] = a[0]-b[0];
	a_less_b[1] = a[1]-b[1];
	a_less_b[2] = a[2]-b[2];
}

inline void vSub3(float a[3], const float b[3]) {
	a[0] -= b[0];
	a[1] -= b[1];
	a[2] -= b[2];
}

inline void vClamp3(float a[3], float min, float max) {
	for(int dim=0; dim<3; dim++)
		if(a[dim]<min) a[dim]=min; 
		else if(a[dim]>max) a[dim] = max;
}

inline void vAdd3(float a_plus_b[3], const float a[3], const float b[3]) {
	a_plus_b[0] = a[0]+b[0];
	a_plus_b[1] = a[1]+b[1];
	a_plus_b[2] = a[2]+b[2];
}

inline void vAdd3(float a[3], const float b[3]) {
	a[0] += b[0];
	a[1] += b[1];
	a[2] += b[2];
}

inline void vAdd3(float a[3], const float b) {
	a[0] += b;
	a[1] += b;
	a[2] += b;
}

// Multiply-then add
inline void vMAD3(float a_plus_b_times_s[3], const float a[3], const float b[3], float s) {
	a_plus_b_times_s[0] = a[0]+b[0]*s;
	a_plus_b_times_s[1] = a[1]+b[1]*s;
	a_plus_b_times_s[2] = a[2]+b[2]*s;
}

inline void vMAD3(float a[3], const float b[3], float s) {
	a[0] += b[0]*s;
	a[1] += b[1]*s;
	a[2] += b[2]*s;
}

inline void vInterpol3(float a_times_u_plus_b_times_v[3], 
					   const float a[3], float u, 
					   const float b[3], float v) {
	a_times_u_plus_b_times_v[0] = a[0]*u+b[0]*v;
	a_times_u_plus_b_times_v[1] = a[1]*u+b[1]*v;
	a_times_u_plus_b_times_v[2] = a[2]*u+b[2]*v;
}

inline void vInterpolTri3(float a_times_u_plus_b_times_v_plus_c_times_w[3], 
						  const float a[3], float u, 
						  const float b[3], float v, 
						  const float c[3], float w) {
	a_times_u_plus_b_times_v_plus_c_times_w[0] = a[0]*u+b[0]*v+c[0]*w;
	a_times_u_plus_b_times_v_plus_c_times_w[1] = a[1]*u+b[1]*v+c[1]*w;
	a_times_u_plus_b_times_v_plus_c_times_w[2] = a[2]*u+b[2]*v+c[2]*w;
}

inline void vScl3(float a_times_s[3], const float a[3], float s) {
	a_times_s[0] = a[0]*s;
	a_times_s[1] = a[1]*s;
	a_times_s[2] = a[2]*s;
}

inline void vScl3(float a[3], float s) {
	a[0]*=s;
	a[1]*=s;
	a[2]*=s;
}

inline void vScl3(float a_times_s[3], const float a[3], float s[3]) {
	a_times_s[0] = a[0]*s[0];
	a_times_s[1] = a[1]*s[1];
	a_times_s[2] = a[2]*s[2];
}

inline void vScl3(float a[3], const float s[3]) {
	a[0]*=s[0];
	a[1]*=s[1];
	a[2]*=s[2];
}

inline float vDot3(const float a[3], const float b[3]) {
	return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
inline float vDot3(float a[3], float b[3]) {
	return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

inline void vCross3(float cp[3], const float a[3], const float b[3]) {
	// assert(cp!=a && cp!=b);

	cp[0]=a[1]*b[2]-a[2]*b[1];
	cp[1]=a[2]*b[0]-a[0]*b[2];
	cp[2]=a[0]*b[1]-a[1]*b[0];
}

inline float vSqrLen3(const float a[3]) {
	return vDot3(a,a);
}

inline float vSqrLen3(const float a[3], const float b[3]) {
	return (a[0]-b[0])*(a[0]-b[0])
		+(a[1]-b[1])*(a[1]-b[1])
		+(a[2]-b[2])*(a[2]-b[2]);
}

inline float vLen3(const float a[3]) {
	return sqrt(vDot3(a, a));
}
inline float vLen3(float a[3]) {
	return sqrt(vDot3(a,a));
}

inline float vLen3(const float a[3], const float b[3]) {
	return sqrt(vSqrLen3(a, b));
}

inline float vLen2(const float a[2], const float b[2]) {
	float a3[] = { a[0], a[1], 0 };
	float b3[] = { b[0], b[1], 0 };
	return vLen3(a, b);
}

inline float vLen2(float a[2]) {
	return sqrt(a[0] * a[0] + a[1] * a[1]);
}

inline void vNorm3(float a_norm[3], const float a[3]) {
	float sqrlen = vDot3(a,a);
	if(sqrlen>0) sqrlen = 1.0f/sqrt(sqrlen);
	vScl3(a_norm,a,sqrlen);
}

inline void vNorm3(float a[3]) {
	float sqrlen = vDot3(a, a);
	if (sqrlen>0) sqrlen = 1.0f / sqrt(sqrlen);
	vScl3(a, sqrlen);
}

inline void vNorm2(float a[2]) {
	float length = vLen2(a);
	a[0] /= length;
	a[1] /= length;
}

inline void vFaceNormal(float n[3], const float a[3], const float b[3], const float c[3]) {
	float u[3], v[3];
	vSub3(u,b,a);
	vSub3(v,c,a);
	vCross3(n,u,v);
}

inline void vFaceCenter(float n[3], const float a[3], const float b[3], const float c[3]) {
	n[0] = (1/(float)3) * (a[0] + b[0] + c[0]);
	n[1] = (1/(float)3) * (a[1] + b[1] + c[1]);
	n[2] = (1/(float)3) * (a[2] + b[2] + c[2]);
}

inline bool vEqual(float a[3], float b[3]) {
	return ((a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]));  
}

inline bool vEqual(float a[3], float b[3], int roundingFactor) {
	auto roundfloat = [](float f, int x) { return floor(f*x+0.5)/x; };
	return (roundfloat(a[0], roundingFactor) == roundfloat(b[0], roundingFactor)) 
		&& (roundfloat(a[1], roundingFactor) == roundfloat(b[1], roundingFactor))
		&& (roundfloat(a[2], roundingFactor) == roundfloat(b[2], roundingFactor));
}

inline bool vEqual(float a[3], float x, float y, float z) {
	return ((a[0] == x) && (a[1] == y) && (a[2] == z));  
}

inline void vRotate2(float a[2], int degrees) {
	// TODO: cant use degreeToRadian function...
	float radians = degrees*((float)M_PI / 180);
	float length = vLen2(a);
	float cs = cos(radians);
	float sn = sin(radians);
	float px = a[0] * cs - a[1] * sn;
	float py = a[0] * sn + a[1] * cs;
	a[0] = px;
	a[1] = py;
}

inline float vAngleR(float a[3], float b[3]) {
	float x = vDot3(a,b);
	float y = (vLen3(a) * vLen3(b));
	float angle = acos(clamp(x/y, -1, 1));
	return angle;
}

inline float vAngleR2d(float a[3], float b[3]) {
	return atan2(b[1] - a[1], b[0] - a[0]);
}

inline float vAngleR(float v[3], float v1[3], float v2[3]) {
	float a[3], b[3];
	vSub3(a, v1, v);
	vSub3(b, v2, v);
	return vAngleR(a, b);
}

inline float vAngleD(float a[3], float b[3]) {
	float x = vDot3(a,b);
	float y = (vLen3(a) * vLen3(b));
	float angle = acos(clamp(x/y, -1, 1));
	angle = angle * ((float)180/M_PI);
	return angle;
}

inline float vAngleD(float v[3], float v1[3], float v2[3]) {
	float a[3], b[3];
	vSub3(a, v1, v);
	vSub3(b, v2, v);
	return vAngleD(a, b);
}

inline void vCircumcenter2d(float center[3], float a[3], float b[3], float c[3]) {
	if(!(a[0] == 0 && a[1] == 0)) return;

	float x = c[1]*(pow(b[0],2) + pow(b[1],2)) - b[1]*(pow(c[0],2) + pow(c[1],2));
	float y = b[0]*(pow(c[0],2) + pow(c[1],2)) - c[0]*(pow(b[0],2) + pow(b[1],2));
	float d = 2*(b[0]*c[1] - b[1]*c[0]);
	x = x / d;
	y = y / d;
	center[0] = x;
	center[1] = y;
	return;
}

inline float vCircumcenter3d(float center[3], float a[3], float b[3], float c[3]) {

	//		  |c-a|^2 [(b-a)x(c-a)]x(b-a) + |b-a|^2 (c-a)x[(b-a)x(c-a)]
	//m = a + ---------------------------------------------------------.
	//						    2 | (b-a)x(c-a) |^2

	//Vector3f a,b,c // are the 3 pts of the tri

	float ac[3], ab[3], abXac[3], toCircumsphereCenter[3];
	vSub3(ac, c, a);
	vSub3(ab, b, a);
	vCross3(abXac, ab, ac);

	// this is the vector from a TO the circumsphere center
	float t1[3], t2[3];
	vCross3(t1, abXac, ab);
	vCross3(t2, ac, abXac);
	vScl3(t1, pow(vLen3(ac),2));
	vScl3(t2, pow(vLen3(ab),2));
	vAdd3(t1, t2);
	vScl3(t1, (float)1/(pow(vLen3(abXac),2) * 2));
	vSet3(toCircumsphereCenter, t1);

	// The 3 space coords of the circumsphere center then:
	vAdd3(center, a, toCircumsphereCenter);

	float circumsphereRadius = vLen3(toCircumsphereCenter);
	return circumsphereRadius;
}

//
//
//

union Vec2 {
	struct {
		float x, y;
	};

	struct {
		float w, h;
	};

	float e[2];
};

union Vec2i {
	struct {
		int x, y;
	};

	struct {
		int w, h;
	};

	int e[2];
};

union Vec3 {
	struct {
		float x, y, z;
	};

	struct {
		Vec2 xy;
		float z;
	};

	struct {
		float x;
		Vec2 yz;
	};

	float e[3];
};

union Vec3i {
	struct {
		int x, y, z;
	};

	struct {
		Vec2i xy;
		int z;
	};

	struct {
		int x;
		Vec2i yz;
	};

	float e[3];
};

union Vec4 {
	struct {
		float x, y, z, w;
	};

	struct {
		Vec3 xyz;
		float w;
	};

	struct {
		float r, g, b, a;
	};

	float e[4];
};

union Mat4 {
	struct {
		float xa, xb, xc, xd;
		float ya, yb, yc, yd;
		float za, zb, zc, zd;
		float wa, wb, wc, wd;
	};

	struct {
		float x1, y1, z1, w1;
		float x2, y2, z2, w2;
		float x3, y3, z3, w3;
		float x4, y4, z4, w4;
	};

	float e[16];

	float e2[4][4];
};

union Rect {
	struct {
		Vec2 min;
		Vec2 max;
	};
	struct {
		Vec2 cen;
		Vec2 dim;
	};
	struct {
		float e[4];
	};
};

union Rect3 {
	struct {
		Vec3 min;
		Vec3 max;
	};
	struct {
		Vec3 cen;
		Vec3 dim;
	};
	struct {
		float e[6];
	};
};

union Rect3i {
	struct {
		Vec3i min;
		Vec3i max;
	};
	struct {
		Vec3i cen;
		Vec3i dim;
	};
	struct {
		float e[6];
	};
};

//
//
//

inline Vec2i vec2i(int a, int b) {
	Vec2i vec;
	vec.x = a;
	vec.y = b;
	return vec;
}

inline Vec2i vec2i(Vec2 a) {
	Vec2i vec;
	vec.x = a.x;
	vec.y = a.y;
	return vec;
}

inline bool operator==(Vec2i a, Vec2i b) {
	bool equal = (a.x == b.x) && (a.y == b.y);
	return equal;
}

inline bool operator!=(Vec2i a, Vec2i b) {
	return !(a==b);
}

inline Vec2i operator+(Vec2i a, int b) {
	a.x += b;
	a.y += b;
	return a;
}

inline Vec2i operator+(Vec2i a, Vec2i b) {
	a.x += b.x;
	a.y += b.y;
	return a;
}

inline Vec2i & operator+=(Vec2i & a, Vec2i b) {
	a = a + b;
	return a;
}

inline Vec2i & operator+=(Vec2i & a, int b) {
	a = a + b;
	return a;
}

inline Vec2i operator-(Vec2i a, int b) {
	a.x -= b;
	a.y -= b;
	return a;
}

inline Vec2i operator-(Vec2i a, Vec2i b) {
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

inline Vec2i & operator-=(Vec2i & a, Vec2i b) {
	a = a - b;
	return a;
}

inline Vec2i & operator-=(Vec2i & a, int b) {
	a = a - b;
	return a;
}

inline Vec2i operator*(Vec2i a, int b) {
	a.x *= b;
	a.y *= b;
	return a;
}

inline Vec2i operator/(Vec2i a, Vec2i b) {
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

// //
// //
// //

inline Vec2 vec2(float a, float b) {
	Vec2 vec;
	vec.x = a;
	vec.y = b;
	return vec;
}

inline Vec2 vec2(Vec2i a) {
	Vec2 vec;
	vec.x = a.x;
	vec.y = a.y;
	return vec;
}

// inline Vec2 vec2(float* a) /*float a[2]*/
// {
// 	Vec2 vec;
// 	vec.x = a[0];
// 	vec.y = a[1];
// 	return vec;
// }

inline Vec2 vec2(float a) {
	Vec2 vec;
	vec.x = a;
	vec.y = a;
	return vec;
}

inline Vec2 operator*(Vec2 a, float b) {
	a.x *= b;
	a.y *= b;
	return a;
}

inline Vec2 operator*(float b, Vec2 a) {
	a.x *= b;
	a.y *= b;
	return a;
}

inline float operator*(Vec2 a, Vec2 b) {
	float dot = a.x*b.x + a.y*b.y;
	return dot;
}

inline Vec2 & operator*=(Vec2 & a, float b) {
	a = a * b;
	return a;
}

inline Vec2 operator+(Vec2 a, float b) {
	a.x += b;
	a.y += b;
	return a;
}

inline Vec2 operator+(Vec2 a, Vec2 b) {
	a.x += b.x;
	a.y += b.y;
	return a;
}

inline Vec2 & operator+=(Vec2 & a, Vec2 b) {
	a = a + b;
	return a;
}

inline Vec2 & operator+=(Vec2 & a, float b) {
	a = a + b;
	return a;
}

inline Vec2 operator-(Vec2 a, float b) {
	a.x -= b;
	a.y -= b;
	return a;
}

inline Vec2 operator-(Vec2 a, Vec2 b) {
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

inline Vec2 operator-(Vec2 a) {
	a.x = -a.x;
	a.y = -a.y;
	return a;
}

inline Vec2 & operator-=(Vec2 & a, Vec2 b) {
	a = a - b;
	return a;
}

inline Vec2 & operator-=(Vec2 & a, float b) {
	a = a - b;
	return a;
}

inline Vec2 operator/(Vec2 a, float b) {
	a.x /= b;
	a.y /= b;
	return a;
}

inline bool operator==(Vec2 a, Vec2 b) {
	bool equal = (a.x == b.x) && (a.y == b.y);
	return equal;
}

inline bool operator!=(Vec2 a, Vec2 b) {
	return !(a==b);
}

inline Vec2 mulVec2(Vec2 a, Vec2 b) {
	Vec2 result;
	result.x = a.x * b.x;
	result.y = a.y * b.y;

	return result;
}

inline float lenVec2(Vec2 a) {
	float length = sqrt(a.x*a.x + a.y*a.y);
	return length;
}

inline Vec2 normVec2(Vec2 a) {
	Vec2 norm;
	float len = sqrt(a.x*a.x + a.y*a.y);
	if(len > 0) norm = a/len;
	else norm = {0,0};

	return norm;
}

inline Vec2 normVec2Unsafe(Vec2 a) {
	Vec2 result = a/sqrt(a.x*a.x + a.y*a.y);

	return result;
}

inline float angleVec2(Vec2 dir1, Vec2 dir2) {
	float dot = normVec2(dir1) * normVec2(dir2);
	dot = clamp(dot, -1, 1);
	float angle = acos(dot);
	return angle;
}

inline float angleDirVec2(Vec2 dir) {
	float angle = atan2(dir.y, dir.x);
	return angle;
}

inline float lenLine(Vec2 p0, Vec2 p1) {
	float result = lenVec2(p1 - p0);
	return result;
}

inline Vec2 lineMidPoint(Vec2 p1, Vec2 p2) {
	float x = (p1.x + p2.x)/2;
	float y = (p1.y + p2.y)/2;
	Vec2 midPoint = vec2(x,y);

	return midPoint;
}

inline Vec2 lineNormal(Vec2 p1, Vec2 p2) {
	float dx = p2.x - p1.x;
	float dy = p2.y - p1.y;
	Vec2 normal = vec2(-dy,dx); // or (dy,-dx)
	normal = normVec2(normal);

	return normal;
}

inline Vec2 lineNormal(Vec2 dir) {
	Vec2 normal = vec2(-dir.y,dir.x);
	normal = normVec2(normal);

	return normal;
}

inline int lineSide(Vec2 p1, Vec2 p2, Vec2 point) {
	int side = sign( (p2.x-p1.x)*(point.y-p1.y) - (p2.y-p1.y)*(point.x-p1.x) );
	return side;
}

inline float detVec2(Vec2 a, Vec2 b) {
	a = normVec2(a);
	b = normVec2(b);
	float det = a.x * b.y - b.x * a.y;
	return det;	
}

inline Vec2 midOfTwoVectors(Vec2 p0, Vec2 p1, Vec2 p2) {
	Vec2 dir1 = normVec2(p0 - p1);
	Vec2 dir2 = normVec2(p2 - p1);
	Vec2 mid = lineMidPoint(dir1, dir2);
	mid = normVec2(mid);
	return mid;
}

inline bool getLineIntersection(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Vec2 * i) {
    Vec2 s1 = p1 - p0;
    Vec2 s2 = p3 - p2;

    float s, t;

    s = (-s1.y * (p0.x - p2.x) + s1.x * (p0.y - p2.y)) / (-s2.x * s1.y + s1.x * s2.y);
    t = ( s2.x * (p0.y - p2.y) - s2.y * (p0.x - p2.x)) / (-s2.x * s1.y + s1.x * s2.y);

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
    {
        *i = p0 + t*s1;
        return true;
    }

    return false;
}

// clockwise
Vec2 rotateVec2(Vec2 a, float radian) {
	float cs = cos(-radian);
	float sn = sin(-radian);

	float nx = a.x * cs - a.y * sn;
	float ny = a.x * sn + a.y * cs;

	return vec2(nx, ny);
}

float distancePointLine(Vec2 v1, Vec2 v2, Vec2 point, bool infinite = false) {
	Vec2 diff = v2 - v1;
    if (diff == vec2(0,0))
    {
    	diff = point - v1;
        return lenVec2(diff);
    }

    float t = ((point.x - v1.x) * diff.x + (point.y - v1.y) * diff.y) / 
    		   (diff.x * diff.x + diff.y * diff.y);

   	if(infinite) {
		diff = point - (v1 + t*diff);
   	} else {
	    if 		(t < 0) diff = point - v1;
	    else if (t > 1) diff = point - v2;
	    else 			diff = point - (v1 + t*diff);
   	}

    return lenVec2(diff);
}

Vec2 lineClosestPoint(Vec2 v1, Vec2 v2, Vec2 point) {
	Vec2 result = {};
	Vec2 diff = v2 - v1;
    if (diff == vec2(0,0)) {
    	result = v1;
    } else {
	    float t = ((point.x - v1.x) * diff.x + (point.y - v1.y) * diff.y) / 
	    		   (diff.x * diff.x + diff.y * diff.y);

	    if 		(t < 0) result = v1;
	    else if (t > 1) result = v2;
	    else 			result = v1 + t*diff;
    }

	return result;
}

void closestPointToTriangle(float closest[2], float a[2], float b[2], float c[2], float p[2]) {
	// get 2 closest points
	float p1[2], p2[2];
	float lena = vLen2(p, a);
	float lenb = vLen2(p, b);
	float lenc = vLen2(p, c);
	float maxLen = max(lena, lenb, lenc);
	float nearest[2];
	if (maxLen == lena) {
		vSet2(p1, b);
		vSet2(p2, c);
		vSet2(nearest, a);
	}
	if (maxLen == lenb) {
		vSet2(p1, a);
		vSet2(p2, c);
		vSet2(nearest, b);
	}
	if (maxLen == lenc) {
		vSet2(p1, a);
		vSet2(p2, b);
		vSet2(nearest, c);
	}

	// check if point inside normal rectangle
	float p1LinePoint[2] = { p1[0] + 1, -p1[1] + 1 };
	float p2LinePoint[2] = { p2[0] + 1, -p2[1] + 1 };
	bool resultp1 = pointIsLeftOfLine(p1, p1LinePoint, p);
	bool resultp2 = pointIsLeftOfLine(p2, p2LinePoint, p);
	if (resultp1 != resultp2)
	{
		vSet2(closest, p);
	}
	else
	{
		// closest point is nearest triangle point
		vSet2(closest, nearest);
	}
}

Vec2 projectPointOnLine(Vec2 p0, Vec2 lp0, Vec2 lp1) {
	Vec2 a = lp1 - lp0;
	Vec2 b = p0 - lp0;
	Vec2 result = lp0 + (((a*b)*a) / pow(lenVec2(a),2));

 // Point e1 = new Point(v2.x - v1.x, v2.y - v1.y);
  // Point e2 = new Point(p.x - v1.x, p.y - v1.y);
  // double valDp = dotProduct(e1, e2);
  // get squared length of e1
  // double len2 = e1.x * e1.x + e1.y * e1.y;
  // Point p = new Point((int)(v1.x + (val * e1.x) / len2),
                      // (int)(v1.y + (val * e1.y) / len2));

	return result;
}

inline bool lineCircleIntersection(Vec2 lp0, Vec2 lp1, Vec2 cp, float r, Vec2 * i) {
	Vec2 d = lp1 - lp0;
	Vec2 f = lp0 - cp;

	float a = d*d;
	float b = 2*f*d;
	float c = f*f - r*r;

	float discriminant = b*b-4*a*c;
	if( discriminant < 0 )
	{
		return false;
	}
	else
	{
		discriminant = sqrt( discriminant );

		float t1 = (-b - discriminant)/(2*a);
		float t2 = (-b + discriminant)/(2*a);

		if( t1 >= 0 && t1 <= 1 )
		{
			*i = lp0 + d*t1;
			return true;
		}

		if( t2 >= 0 && t2 <= 1 )
		{
			*i = lp0 + d*t2;
			return true;
		}

		return false;
	}

	return false;
}

inline Vec2 calculatePosition(Vec2 oldPosition, Vec2 velocity, Vec2 acceleration, float time) {
	oldPosition += 0.5f*acceleration*time*time + velocity*time;
	return oldPosition;
}

inline Vec2 calculateVelocity(Vec2 oldVelocity, Vec2 acceleration, float time) {
	oldVelocity += acceleration*time;
	return oldVelocity;
}

inline Vec2 clampMin(Vec2 dim, Vec2 v) {
	Vec2 result;
	result.x = clampMin(dim.x, v.x);
	result.y = clampMin(dim.y, v.y);

	return result;
}	

inline Vec2 clampMax(Vec2 v, Vec2 dim) {
	Vec2 result = v;
	v.x = clampMax(v.x, dim.x);
	v.y = clampMax(v.x, dim.y);
	
	return result;
}	

inline Vec2 toVec2(Vec3 a) {
	Vec2 result;
	result.x = a.x;
	result.y = a.y;

	return result;
}

inline Vec3 toVec3(Vec2 a);
inline Vec3 cross(Vec3 a, Vec3 b);

inline Vec2 perpToPoint(Vec2 dir, Vec2 dirPoint) {
	Vec3 ab = {dir.x, dir.y, 0};
	Vec3 ap = {dirPoint.x, dirPoint.y, 0};

	Vec3 abxap = {	ab.y*ap.z - ab.z*ap.y,
					ab.z*ap.x - ab.x*ap.z,
					ab.x*ap.y - ab.y*ap.x };

	Vec3 xab = {	abxap.y*ab.z - abxap.z*ab.y,
					abxap.z*ab.x - abxap.x*ab.z,
					abxap.x*ab.y - abxap.y*ab.x };

	Vec2 result = xab.xy;

	return result;
}

inline Vec2 perpToPoint(Vec2 a, Vec2 b, Vec2 p) {
	Vec2 ab = {	b.x - a.x,
				b.y - a.y, };

	Vec2 ap = {	p.x - a.x,
				p.y - a.y, };

	Vec2 result = perpToPoint(ab, ap);

	return result;
}

inline float cross(Vec2 a, Vec2 b) {
	float result = a.x*b.y - a.y*b.x;
	return result;
}

inline Vec2 dirTurnRight(Vec2 dir) {
	Vec2 result = vec2(dir.y, -dir.x);
	return result;
}

inline Vec2 dirTurnLeft(Vec2 dir) {
	Vec2 result = vec2(-dir.y, dir.x);
	return result;
}

inline bool equalVec2(Vec2 a, Vec2 b, float margin) {
	bool equalX = roughlyEqual(a.x, b.x, margin);
	bool equalY = roughlyEqual(a.y, b.y, margin);
	bool equal = equalX && equalY;
	
	return equal;
}

inline Vec2 mapVec2(Vec2 a, Vec2 oldMin, Vec2 oldMax, Vec2 newMin, Vec2 newMax) {
	Vec2 result = a;
	result.x = mapRange(result.x, oldMin.x, oldMax.x, newMin.x, newMax.x);
	result.y = mapRange(result.y, oldMin.y, oldMax.y, newMin.y, newMax.y);

	return result;
}

inline Vec2 angleToDir(float angleInRadians) {
	Vec2 result = vec2(cos(angleInRadians), sin(angleInRadians));
	return result;
}

inline bool vecBetweenVecs(Vec2 v, Vec2 left, Vec2 right) {
	bool result;
	float ca = cross(left,v);
	float cb = cross(right,v);

	result = ca < 0 && cb > 0;
	return result;
}

//
//
//

inline Vec3 vec3(float a, float b, float c) {
	Vec3 vec;
	vec.x = a;
	vec.y = b;
	vec.z = c;
	return vec;
}

inline Vec3 vec3(Vec3i v) {
	Vec3 vec;
	vec.x = v.x;
	vec.y = v.y;
	vec.z = v.z;
	return vec;
}

inline Vec3 vec3(float a[3]) {
	Vec3 vec;
	vec.x = a[0];
	vec.y = a[1];
	vec.z = a[2];
	return vec;
}

inline Vec3 vec3(float a) {
	Vec3 vec;
	vec.x = a;
	vec.y = a;
	vec.z = a;
	return vec;
}

inline Vec3 vec3(Vec2 a, float b) {
	Vec3 vec;
	vec.xy = a;
	vec.z = b;
	return vec;
}

inline Vec3 vec3(float a, Vec2 b) {
	Vec3 vec;
	vec.x = a;
	vec.yz = b;
	return vec;
}

inline Vec3 operator*(Vec3 a, float b) {
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}

inline Vec3 operator*(float b, Vec3 a) {
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}

inline Vec3 operator*(Vec3 a, Vec3 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

inline Vec3 & operator*=(Vec3 & a, float b) {
	a = a * b;
	return a;
}

inline Vec3 operator+(Vec3 a, float b) {
	a.x += b;
	a.y += b;
	a.z += b;
	return a;
}

inline Vec3 operator+(Vec3 a, Vec3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

inline Vec3 & operator+=(Vec3 & a, Vec3 b) {
	a = a + b;
	return a;
}

inline Vec3 operator-(Vec3 a, float b) {
	a.x -= b;
	a.y -= b;
	a.z -= b;
	return a;
}

inline Vec3 operator-(Vec3 a, Vec3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

inline Vec3 & operator-=(Vec3 & a, Vec3 b) {
	a = a - b;
	return a;
}

inline Vec3 operator/(Vec3 a, float b) {
	a.x /= b;
	a.y /= b;
	a.z /= b;
	return a;
}

inline Vec3 operator-(Vec3 a) {
	a.x = -a.x;
	a.y = -a.y;
	a.z = -a.z;
	return a;
}

inline bool operator==(Vec3 a, Vec3 b) {
	bool equal = (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
	return equal;
}

inline bool operator!=(Vec3 a, Vec3 b) {
	return !(a==b);
}

inline float dot(Vec3 a, Vec3 b) {
	float result = a.x*b.x + a.y*b.y + a.z*b.z;
	return result;
}

inline Vec3 cross(Vec3 a, Vec3 b) {
	Vec3 result;
	result.x = a.y*b.z - a.z*b.y;
	result.y = a.z*b.x - a.x*b.z;
	result.z = a.x*b.y - a.y*b.x;

	return result;
}

inline Vec3 toVec3(Vec2 a) {
	Vec3 result;
	result.x = a.x;
	result.y = a.y;
	result.z = 0;

	return result;
}

inline float lenVec3(Vec3 a) {
	float sqrlen = dot(a,a);
	if(sqrlen > 0) sqrlen = 1.0f/sqrt(sqrlen);
	return sqrlen;
}

inline Vec3 normVec3(Vec3 a) {
	float sqrlen = lenVec3(a);
	return a*sqrlen;
}

Vec3 projectPointOnLine(Vec3 lPos, Vec3 lDir, Vec3 p) {
	Vec3 result;
	result = lPos + ((dot(p-lPos, lDir) / dot(lDir,lDir))) * lDir;
	return result;
}

bool boxRaycast(Vec3 lp, Vec3 ld, Rect3 box, float* distance = 0, int* face = 0) {
	// ld is unit
	Vec3 dirfrac;
	dirfrac.x = 1.0f / ld.x;
	dirfrac.y = 1.0f / ld.y;
	dirfrac.z = 1.0f / ld.z;
	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	// r.org is origin of ray
	float t1 = (box.min.x - lp.x)*dirfrac.x;
	float t2 = (box.max.x - lp.x)*dirfrac.x;
	float t3 = (box.min.y - lp.y)*dirfrac.y;
	float t4 = (box.max.y - lp.y)*dirfrac.y;
	float t5 = (box.min.z - lp.z)*dirfrac.z;
	float t6 = (box.max.z - lp.z)*dirfrac.z;

	float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

	float t;
	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
	if (tmax < 0) {
	    t = tmax;
	    return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tmin > tmax) {
	    t = tmax;
	    return false;
	}

	if(face) {
		     if(tmin == t1) *face = 0;
		else if(tmin == t2) *face = 1;
		else if(tmin == t3) *face = 2;
		else if(tmin == t4) *face = 3;
		else if(tmin == t5) *face = 4;
		else if(tmin == t6) *face = 5;
	}

	t = tmin;
	if(distance != 0) *distance = t;
	return true;
}

int getBiggestAxis(Vec3 v, int smallerAxis[2] = 0) {
	int biggestAxis = maxReturnIndex(abs(v.x), abs(v.y), abs(v.z));
	if(smallerAxis != 0) {
		smallerAxis[0] = mod(biggestAxis-1, 3);
		smallerAxis[1] = mod(biggestAxis+1, 3);
	}

	return biggestAxis;
}

//
//
//

inline Vec3i vec3i(int a, int b, int c) {
	Vec3i vec;
	vec.x = a;
	vec.y = b;
	vec.z = c;
	return vec;
}

inline Vec3i vec3i(Vec3 a) {
	return vec3i(a.x,a.y,a.z);
}

inline Vec3i operator+(Vec3i a, float b) {
	a.x += b;
	a.y += b;
	a.z += b;
	return a;
}

inline Vec3i operator+(Vec3i a, Vec3i b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

inline Vec3i & operator+=(Vec3i & a, Vec3i b) {
	a = a + b;
	return a;
}

inline Vec3i operator-(Vec3i a, float b) {
	a.x -= b;
	a.y -= b;
	a.z -= b;
	return a;
}

inline Vec3i operator-(Vec3i a, Vec3i b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

inline Vec3i operator/(Vec3i a, int b) {
	a.x /= b;
	a.y /= b;
	a.z /= b;
	return a;
}

inline Vec3i & operator-=(Vec3i & a, Vec3i b) {
	a = a - b;
	return a;
}

inline bool operator==(Vec3i a, Vec3i b) {
	bool equal = (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
	return equal;
}

//
//
//

inline Vec4 vec4(float a, float b, float c, float d) {
	Vec4 vec;
	vec.x = a;
	vec.y = b;
	vec.z = c;
	vec.w = d;
	return vec;
}

inline Vec4 vec4(float a[4]) {
	Vec4 vec;
	vec.x = a[0];
	vec.y = a[1];
	vec.z = a[2];
	vec.w = a[3];
	return vec;
}

inline Vec4 vec4(float a) {
	Vec4 vec;
	vec.x = a;
	vec.y = a;
	vec.z = a;
	vec.w = a;
	return vec;
}

inline Vec4 vec4(Vec3 a, float w) {
	Vec4 vec;
	vec.xyz = a;
	vec.w = w;
	return vec;
}

//
//
//

inline Mat4 operator*(Mat4 a, Mat4 b) {
	Mat4 r;
	int i = 0;
	for(int y = 0; y < 16; y += 4) {
		for(int x = 0; x < 4; x++) {
			r.e[i++] = a.e[y]*b.e[x] + a.e[y+1]*b.e[x+4] + a.e[y+2]*b.e[x+8] + a.e[y+3]*b.e[x+12];
		}
	}
	return r;
}

inline void rowToColumn(Mat4* m) {
	for(int y = 0; y < 4; y++) {
		for(int x = 0; x < 4; x++) {
			float temp = m->e2[y][x];
			m->e2[y][x] = m->e2[x][y];
			m->e2[x][y] = temp;
		}
	}
}

inline void scaleMatrix(Mat4* m, Vec3 a) {
	*m = {};
	m->x1 = a.x;
	m->y2 = a.y;
	m->z3 = a.z;
	m->w4 = 1;
}

inline void rotationMatrix(Mat4* m, Vec3 a) {
	*m = {	cos(a.y)*cos(a.z), cos(a.z)*sin(a.x)*sin(a.y)-cos(a.x)*sin(a.z), cos(a.x)*cos(a.z)*sin(a.x)+sin(a.x)*sin(a.z), 0,
			cos(a.y)*sin(a.z), cos(a.x)*cos(a.z)+sin(a.x)*sin(a.y)*sin(a.z), -cos(a.z)*sin(a.x)+cos(a.x)*sin(a.y)*sin(a.z), 0,
			-sin(a.y), 		   cos(a.y)*sin(a.x), 							 cos(a.x)*cos(a.y), 0,
			0, 0, 0, 1};
}

inline void rotationMatrixX(Mat4* m, float a) {
	float ca = cos(a);
	float sa = sin(a);
	*m = {	1, 0, 0, 0,
			0, ca, sa, 0,
			0, -sa, ca, 0,
			0, 0, 0, 1};
}

inline void rotationMatrixY(Mat4* m, float a) {
	float ca = cos(a);
	float sa = sin(a);
	*m = {	ca, 0, -sa, 0,
			0, 1, 0, 0,
			sa, 0, ca, 0,
			0, 0, 0, 1};
}

inline void rotationMatrixZ(Mat4* m, float a) {
	float ca = cos(a);
	float sa = sin(a);
	*m = {	ca, sa, 0, 0,
			-sa, ca, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1};
}

inline void translationMatrix(Mat4* m, Vec3 a) {
	*m = {};
	m->x1 = 1;
	m->y2 = 1;
	m->z3 = 1;
	m->w4 = 1;

	m->w1 = a.x;
	m->w2 = a.y;
	m->w3 = a.z;
}

inline void viewMatrix(Mat4* m, Vec3 cPos, Vec3 cLook, Vec3 cUp, Vec3 cRight) {
	*m = {	cRight.x, cRight.y, cRight.z, -(dot(cPos,cRight)), 
			cUp.x, 	  cUp.y, 	cUp.z,    -(dot(cPos,cUp)), 
			cLook.x,  cLook.y,  cLook.z,  -(dot(cPos,cLook)), 
			0, 		  0, 		0, 		  1 };
}

inline void projMatrix(Mat4* m, float fov, float ar, float n, float f) {
	*m = { 	1/(ar*tan(fov*0.5f)), 0, 				 0, 			 0,
			0, 					  1/(tan(fov*0.5f)), 0, 			 0,
			0, 					  0, 				 -((f+n)/(f-n)), -((2*f*n)/(f-n)),
			0, 					  0, 				 -1, 			 0 };
}

//
//
//

union Quat {
	struct {
		float w, x, y, z;
	};

	struct {
		float w;
		Vec3 xyz;
	};

	float e[4];
};

Quat quat() {
	Quat r = {1,0,0,0};
	return r;
}

Quat quat(float w, float x, float y, float z) {
	Quat r = {w,x,y,z};
	return r;
}

Quat quat(float a, Vec3 axis) {
	Quat r;
	r.w = cos(a*0.5f);
	r.x = axis.x * sin(a*0.5f);
	r.y = axis.y * sin(a*0.5f);
	r.z = axis.z * sin(a*0.5f);
	return r;
}

float quatMagnitude(Quat q) {
	float result = sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
	return result;
}

Quat normQuat(Quat q) {
	Quat result;
	float m = quatMagnitude(q);
	result = quat(q.w/m, q.x/m, q.y/m, q.z/m);
	return result;
}

Quat operator*(Quat a, Quat b) {
	Quat r;
	r.w = (a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
	r.x = (a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y);
	r.y = (a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x);
	r.z = (a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w);
	return r;
}

void quatRotationMatrix(Mat4* m, Quat q) {
	float w = q.w, x = q.x, y = q.y, z = q.z;
	float x2 = x*x, y2 = y*y, z2 = z*z;
	float w2 = w*w;
	*m = {	w2+x2-y2-z2, 2*x*y-2*w*z, 2*x*z+2*w*y, 0,
			2*x*y+2*w*z, w2-x2+y2-z2, 2*y*z-2*w*x, 0,
			2*x*z-2*w*y, 2*y*z+2*w*x, w2-x2-y2+z2, 0,
			0, 			 0, 		  0, 		   1};
}

Vec3 operator*(Quat q, Vec3 v) {
	Quat r = q*quat(0, v.x, v.y, v.z);
	return r.xyz;
}

Vec3 rotateVec3(Vec3 v, float a, Vec3 axis) {
	Vec3 r = quat(a, axis)*v;
	return normVec3(r);
}

void rotateVec3(Vec3* v, float a, Vec3 axis) {
	*v = quat(a, axis)*(*v);
	*v = normVec3(*v);
}

//
//
//

inline Rect rect(Vec2 min, Vec2 max) {
	Rect r;
	r.min = min;
	r.max = max;

	return r;
}

inline Rect rect(float left, float bottom, float right, float top) {
	Rect r;
	r.min = vec2(left, bottom);
	r.max = vec2(right, top);
	return r;
}

inline Rect rectSides(float left, float right, float bottom, float top) {
	Rect r = rect(left, bottom, right, top);
	return r;
}

inline Rect rectMinDim(Vec2 min, Vec2 dim) {
	Rect r;
	r.min = min;
	r.max = min + dim;

	return r;
}

inline Rect rectULDim(Vec2 ul, Vec2 dim) {
	Rect r;
	r.min = vec2(ul.x, ul.y - dim.h);
	r.max = r.min + dim;

	return r;
}

inline Rect rectCenDim(Vec2 cen, Vec2 dim) {
	float w2 = dim.w/2;
	float h2 = dim.h/2;
	Rect r;
	r.min = vec2(cen.x - w2, cen.y - h2);
	r.max = vec2(cen.x + w2, cen.y + h2);

	return r;
}

// NOTE: really?
inline Rect rectCenDim(float x, float y, float w, float h) {
	Rect r;
	r.min = vec2(x - w/2, y - h/2);
	r.max = vec2(x + w/2, y + h/2);

	return r;
}

inline Vec2 rectGetDim(Rect r) {
	Vec2 v;
	v = r.max - r.min;
	return v;
}

inline Vec2 rectGetCen(Rect r) {
	Vec2 dim;
	dim = rectGetDim(r);
	Vec2 v;
	v = r.min + dim/2;
	return v;
}

inline Rect rectGetCenDim(Rect r) {
	Rect newR;
	newR.dim = r.max - r.min;
	newR.cen = r.min + newR.dim/2;

	return newR;
}

inline Rect rectGetMinMax(Rect r) {
	Rect newR;
	Vec2 halfDim = r.dim/2;
	newR.min = r.cen - halfDim;
	newR.max = r.cen + halfDim;

	return newR;
}

inline Vec2 rectGetUL(Rect r) {
	Vec2 result = vec2(r.min.x, r.max.y);
	return result;
}

inline Vec2 rectGetDR(Rect r) {
	Vec2 result = vec2(r.max.x, r.min.y);
	return result;
}

inline bool operator==(Rect r1, Rect r2) {
	bool equal = (r1.min == r2.min) && (r1.max == r2.max);
	return equal;
}

inline bool rectIntersection(Rect r1, Rect r2) {
	bool hasIntersection = !(r2.min.x > r1.max.x ||
							 r2.max.x < r1.min.x ||
							 r2.max.y < r1.min.y ||
							 r2.min.y > r1.max.y);
	return hasIntersection;
}

bool rectGetIntersection(Rect * intersectionRect, Rect r1, Rect r2) {
	bool hasIntersection = rectIntersection(r1, r2);
	if (hasIntersection)
	{
		intersectionRect->min.x = max(r1.min.x, r2.min.x);
		intersectionRect->max.x = min(r1.max.x, r2.max.x);
		intersectionRect->max.y = min(r1.max.y, r2.max.y);
		intersectionRect->min.y = max(r1.min.y, r2.min.y);
	}
	else *intersectionRect = rect(0,0,0,0);
	
	return hasIntersection;
};

bool pointInRect(Vec2 p, Rect r) {
	bool inRect = ( p.x > r.min.x &&
					p.x < r.max.x &&
					p.y > r.min.y &&
					p.y < r.max.y   );

	return inRect;
}

bool rectEmpty(Rect r) {
	bool result = (r == rect(0,0,0,0));
	return result;
}

Rect rectAddOffset(Rect r, Vec2 offset) {
	Rect result = rect(r.min + offset, r.max + offset);

	return result;	
}

Rect rectExpand(Rect r, Rect rExpand) {
	Vec2 dim = rectGetDim(rExpand);
	r.min -= dim/2;
	r.max += dim/2;

	return r;
}

Rect rectExpand(Rect r, Vec2 dim) {
	r.min -= dim/2;
	r.max += dim/2;

	return r;
}

bool rectInsideRect(Rect r0, Rect r1) {
	bool result = (r0.min.x > r1.min.x &&
	               r0.min.y > r1.min.y &&
	               r0.max.x < r1.max.x &&
	               r0.max.y < r1.max.y);
	return result;
}

Vec2 rectInsideRectClamp(Rect r0, Rect r1) {
	Vec2 offset = vec2(0,0);
	if(r0.min.x < r1.min.x) offset += vec2(r1.min.x-r0.min.x, 0);
	if(r0.min.y < r1.min.y) offset += vec2(0, r1.min.y-r0.min.y);
	if(r0.max.x > r1.max.x) offset += vec2(r1.max.x-r0.max.x, 0);
	if(r0.max.y > r1.max.y) offset += vec2(0, r1.max.y-r0.max.y);

	return offset;
}

Rect mapRect(Rect r, Rect oldInterp, Rect newInterp) {
	Rect result = r;
	result.min = mapVec2(result.min, oldInterp.min, oldInterp.max, newInterp.min, newInterp.max);
	result.max = mapVec2(result.max, oldInterp.min, oldInterp.max, newInterp.min, newInterp.max);

	return result;
}

//
//
//

inline Rect3 rect3(Vec3 min, Vec3 max) {
	Rect3 r;
	r.min = min;
	r.max = max;

	return r;
}

inline Rect3 rect3CenDim(Vec3 cen, Vec3 dim) {
	Rect3 r;
	r.min = cen-dim*0.5f;
	r.max = cen+dim*0.5f;

	return r;
}

// Rect rectExpand(Rect r, Rect rExpand) {
// 	Vec2 dim = rectGetDim(rExpand);
// 	r.min -= dim/2;
// 	r.max += dim/2;

// 	return r;
// }

Rect3 rect3Expand(Rect3 r, Vec3 dim) {
	r.min -= (dim/2);
	r.max += (dim/2);

	return r;
}

//
//
//

inline Rect3i rect3i(Vec3i min, Vec3i max) {
	Rect3i r;
	r.min = min;
	r.max = max;

	return r;
}

inline Rect3i rect3iCenRad(Vec3i cen, Vec3i radius) {
	Rect3i r;
	r.min = cen-radius;
	r.max = cen+radius;

	return r;
}

Rect3i rect3iExpand(Rect3i r, Vec3i dim) {
	r.min -= (dim/2);
	r.max += (dim/2);

	return r;
}

//
//
//

float ellipseDistanceCenterEdge(float width, float height, Vec2 dir) {
	// dir has to be normalized
	float dirOverWidth = dir.x/width;
	float dirOverHeight = dir.y/height;
	float length = sqrt(1 / (dirOverWidth*dirOverWidth + dirOverHeight*dirOverHeight));
	return length;
}

bool ellipseGetLineIntersection(float a, float b, float h, float k, Vec2 p0, Vec2 p1, Vec2 &i1, Vec2 &i2) {
	// y = mx + c
	float m;
	if(p1.x-p0.x == 0) m = 1000000;
	// if(p1.x-p0.x == 0) m = 0.00001f;
	else m = (p1.y-p0.y)/(p1.x-p0.x);
	float c = p0.y - m*p0.x;

	float aa = a*a;
	float bb = b*b;
	float mm = m*m;
	float temp1 = c + m*h;
	float d = aa*mm + bb - pow(temp1,2) - k*k + 2*temp1*k;

	if (d < 0)
	{
		return false;
	}
	else
	{
		Vec2 inter1;
		Vec2 inter2;
		float q = aa*mm + bb;
		float r = h*bb - m*aa*(c-k);
		float s = bb*temp1 + k*aa*mm;

		float u = a*b*sqrt(d);
		inter1.x = (r - u) / q;
		inter2.x = (r + u) / q;

		float v = a*b*m*sqrt(d);
		inter1.y = (s - v) / q;
		inter2.y = (s + v) / q;

		float lLine = lenVec2(p1 - p0);
		float lp0 = lenVec2(inter1 - p0);
		float lp1 = lenVec2(inter2 - p0);
		if(lp0 <= lp1) {
			i1 = inter1;
			i2 = inter2;
		} else {
			i1 = inter2;
			i2 = inter1;			
		}
		if(lenVec2(i1 - p0) > lLine) return false;
		// if(lp0 > lLine) return false;

		return true;
	}

	return false;
}

Vec2 ellipseNormal(Vec2 pos, float width, float height, Vec2 point) {
	Vec2 dir = vec2((point.x-pos.x)/pow(width,2), (point.y-pos.y)/pow(height,2));
	dir = normVec2(dir);
	dir *= -1;
	return dir;
}

Mat4 orthographicProjection(float left, float right, float bottom, float top, float nearr, float farr) {
	Mat4 mat;
	float v[16] = { 2/(right-left), 0, 0, 0, 
					0, 2/(top-bottom), 0, 0, 
					0, 0, (-2)/(farr-nearr), 0, 
					(-(right+left)/(right-left)), (-(top+bottom)/(top-bottom)), (farr+nearr)/(farr-nearr), 1};

	for(int i = 0; i < 16; ++i) mat.e[i] = v[i];

	return mat;
}

float polygonArea(Vec2* polygon, int count) {
	float signedArea = 0;
	for(int i = 0; i < count; i++) {
		int secondIndex = i+1;
		if(i == count-1) secondIndex = 0;
		Vec2 p1 = polygon[i];
		Vec2 p2 = polygon[secondIndex];

		signedArea += (p1.x * p2.y - p2.x * p1.y);
	}

	return signedArea / 2;
}




void whiteNoise(Rect region, int sampleCount, Vec2* samples) {
	for(int i = 0; i < sampleCount; ++i) {
		Vec2 randomPos = vec2(randomInt(region.min.x, region.max.x), 
		                      randomInt(region.min.y, region.max.y));
		samples[i] = randomPos;
	}
}

int blueNoise(Rect region, float radius, Vec2** noiseSamples, int numOfSamples = 0) {
	Vec2 regionDim = rectGetDim(region);
	if(numOfSamples > 0) {
		radius = (regionDim.w*regionDim.h*(float)M_SQRT2) / (2*numOfSamples);
	}
	float cs = (radius/(float)M_SQRT2)*2; // Square diagonal
	int sampleMax = ((regionDim.w+1)*(regionDim.h+1)) / cs;

	Vec2* samples = *noiseSamples;
	samples = getTArray(Vec2, sampleMax);
	*noiseSamples = samples;
	int testCount = 64;
	int gridW = regionDim.w/cs + 1;
	int gridH = regionDim.h/cs + 1;
	int gridSize = (gridH+1)*(gridW+1);
	int* grid = getTArray(int, gridSize);
	memset(grid, -1, gridSize*sizeof(int));

	int* activeList = getTArray(int, max(gridH, gridW));
	int sampleCount = 1;
	samples[0] = vec2(randomInt(0, regionDim.w), randomInt(0, regionDim.h));
	activeList[0] = 0;
	int activeListSize = 1;
	Vec2 pos = samples[0];
	grid[(int)(pos.y/cs)*gridW+(int)(pos.x/cs)] = 0;

	Rect regionOrigin = rectAddOffset(region, region.min*-1);
	while(activeListSize > 0) {

		int activeIndex = randomInt(0,activeListSize-1);
		int sampleIndex = activeList[activeIndex];
		Vec2 sample = samples[sampleIndex];
		for(int i = 0; i < testCount; ++i) {
			float angle = randomFloat(0, M_2PI, 0.01f);
			float distance = randomFloat(radius*2, radius*3, 0.01f);
			Vec2 newSample = sample+angleToDir(angle)*distance;

			if(!pointInRect(newSample, regionOrigin)) continue;

			// get samples around newSample with 6*r, 
			// making room for 3 circles which should be enough?
			Rect sampleRegion = rectCenDim(newSample,vec2(radius*6));
			sampleRegion.min = sampleRegion.min/cs;
			sampleRegion.max = sampleRegion.max/cs;
			bool validPosition = true;
			for(int y = sampleRegion.min.y; y < (int)sampleRegion.max.y+1; ++y) {
				for(int x = sampleRegion.min.x; x < (int)sampleRegion.max.x+1; ++x) {
					int index = grid[y*gridW+x];
					if(index > -1) {
						Vec2 s = samples[index];
						float distance = lenVec2(s - newSample);
						if(distance < radius*2) {
							validPosition = false;
							break;
						}
					}
				}
				if(!validPosition) break;
			}

			if(validPosition) {
				samples[sampleCount] = newSample;
				activeList[activeListSize] = sampleCount;
				grid[(int)(newSample.y/cs)*gridW+(int)(newSample.x/cs)] = sampleCount;
				sampleCount++;
				activeListSize++;
			}
		}

		// delete active sample after testCoutn times
		activeList[activeIndex] = activeList[activeListSize-1];
		activeListSize--;
	}

	for(int i = 0; i < sampleCount; ++i) samples[i] += region.min;

	return sampleCount;
}
