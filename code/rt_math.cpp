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



int mod(int a, int b) {
	int result;
	result = a % b;
	if(result < 0) result += b;

	return result;
}

float modf(float val, float d) {
	float result = fmod(val, d);
	if(result < 0 && abs(result) < d) {
		result = d + result;
	}

	return result;
}

template <class T> inline void swap(T* a, T* b) {
	T temp = *a;
	*a = *b;
	*b = temp;
}

inline float diff(float a, float b) { return abs(a - b); }
inline float sameSign(float a, float b) { return (a < 0 && b < 0) || (a > 0 && b > 0); }

template <class T> inline T min(T a, T b) { return a <= b ? a : b; }
template <class T> inline T min(T a, T b, T c) { return min(min(a, b), min(b, c)); }
template <class T> inline T max(T a, T b) { return a >= b ? a : b; }
template <class T> inline T max(T a, T b, T c) { return max(max(a, b), max(b, c)); }

template <class T> inline T clampMin(T a, T mi) { return max(a, mi); };
template <class T> inline T clampMax(T a, T ma) { return min(a, ma); };
template <class T> inline T clamp(T a, T mi, T ma) { return min(max(a, mi), ma); };
template <class T> inline T clamp01(T a) { return clamp(a, (T)0, (T)1); };

template <class T> inline void clampMin(T* a, T mi) { *a = clampMin(*a, mi); };
template <class T> inline void clampMax(T* a, T ma) { *a = clampMax(*a, ma); };
template <class T> inline void clamp(T* a, T mi, T ma) { *a = clamp(*a, mi, ma); };
template <class T> inline void clamp01(T* a) { *a = clamp01(*a); };

float lerp(float percent, float min, float max) { return min + percent * (max-min); }

template <class T> inline T mapRange(T value, T min, T max, T rangeMin, T rangeMax) {
	T result = ((value-min)/((max-min)-(min-min))) * (rangeMax-rangeMin) + rangeMin;
	return result;
};

template <class T> inline T mapRange01(T value, T min, T max) {
	T result = ((value-min)/((max-min)-(min-min)));
	return result;
};

inline float mapRangeClamp(float value, float min, float max, float rangeMin, float rangeMax) {
	float result = mapRange(value, min, max, rangeMin, rangeMax);
	result = clamp(result, rangeMin, rangeMax);

	return result;
};

template <class T> inline bool between(T v, T min, T max) { return v >= min && v <= max; }

inline float radianToDegree(float angle) { return angle*(180.0f / M_PI); }
inline float degreeToRadian(float angle) { return angle*(M_PI / 180.0f); }

inline int sign(float x) { return x < 0 ? -1 : (x > 0 ? 1 : 0); }	

inline int roundInt(double i) { return floor(i + 0.5); }
inline int roundIntf(float i) { return floor(i + 0.5f); }

inline double roundUp(double i) { return ceil(i); }
inline float roundUpf(float i) { return ceil(i); }
inline double round(double i) { return floor(i + 0.5); }
inline float roundf(float i) { return floor(i + 0.5f); }
inline double roundDown(double i) { return floor(i); }
inline float roundDownf(float i) { return floor(i); }

inline double roundDigits(double f, int d) { return floor(f*(10*d) + 0.5) / (10*d); };
inline double roundMod(double i, double s) { return (round(i/s))*s; }

inline int roundMod(int i, int s) { return (i/s)*s; }
inline int roundModUp(int i, int s) { return ceil(i/(double)s)*s; }

inline double divSave(double a, double b) { return b==0 ? 0 : a/b; }
inline int triangularNumber(int n) { return n*(n+1) / 2; }
inline float root(float a, float root) { return powf(a, 1/root); }
inline float logBase(float v, float base) { return log(v) / log(base); }

float camDistanceFromFOVandWidth(float fovInDegrees, float w) {
	float angle = degreeToRadian(fovInDegrees);
	float sideAngle = ((M_PI-angle)/2.0f);
	float side = w/sin(angle) * sin(sideAngle);
	float h = side*sin(sideAngle);
	
	return h;
}

float cubicBezierInterpolationSeemless(float A, float B, float CC, float D, float t) {
	float a = -A/2.0f + (3.0f*B)/2.0f - (3.0f*CC)/2.0f + D/2.0f;
	float b = A - (5.0f*B)/2.0f + 2.0f*CC - D / 2.0f;
	float c = -A/2.0f + CC/2.0f;
	float d = B;
	
	return a*t*t*t + b*t*t + c*t + d;
}

//

enum {
	EASE_Linear,
	EASE_InQuad,
	EASE_OutQuad,
	EASE_InOutQuad,
	EASE_InCubic,
	EASE_OutCubic,
	EASE_InOutCubic,
	EASE_InQuart,
	EASE_OutQuart,
	EASE_InOutQuart,
	EASE_InQuint,
	EASE_OutQuint,
	EASE_InOutQuint,
	EASE_InSine,
	EASE_OutSine,
	EASE_InOutSine,
	EASE_InExpo,
	EASE_OutExpo,
	EASE_InOutExpo,
	EASE_InCirc,
	EASE_OutCirc,
	EASE_InOutCirc,
};

float easeFunction(int type, float t, float b, float c, float d) {
	switch(type) {
		case EASE_Linear: 
			return c*t / d + b;

		case EASE_InQuad: 
			t /= d;
			return c*t*t + b;

		case EASE_OutQuad: 
			t /= d;
			return -c * t*(t - 2) + b;

		case EASE_InOutQuad: 
			t /= d / 2;
			if (t < 1) return c / 2 * t*t + b;
			t--;
			return -c / 2 * (t*(t - 2) - 1) + b;

		case EASE_InCubic: 
			t /= d;
			return c*t*t*t + b;

		case EASE_OutCubic: 
			t /= d;
			t--;
			return c*(t*t*t + 1) + b;

		case EASE_InOutCubic: 
			t /= d / 2;
			if (t < 1) return c / 2 * t*t*t + b;
			t -= 2;
			return c / 2 * (t*t*t + 2) + b;

		case EASE_InQuart: 
			t /= d;
			return c*t*t*t*t + b;

		case EASE_OutQuart: 
			t /= d;
			t--;
			return -c * (t*t*t*t - 1) + b;

		case EASE_InOutQuart: 
			t /= d / 2;
			if (t < 1) return c / 2 * t*t*t*t + b;
			t -= 2;
			return -c / 2 * (t*t*t*t - 2) + b;

		case EASE_InQuint: 
			t /= d;
			return c*t*t*t*t*t + b;

		case EASE_OutQuint: 
			t /= d;
			t--;
			return c*(t*t*t*t*t + 1) + b;

		case EASE_InOutQuint: 
			t /= d / 2;
			if (t < 1) return c / 2 * t*t*t*t*t + b;
			t -= 2;
			return c / 2 * (t*t*t*t*t + 2) + b;

		case EASE_InSine: 
			return -c * cos(t / d * M_PI_2) + c + b;

		case EASE_OutSine: 
			return c * sin(t / d * M_PI_2) + b;

		case EASE_InOutSine: 
			return -c / 2 * (cos(M_PI*t / d) - 1) + b;

		case EASE_InExpo: 
			return c * pow(2, 10 * (t / d - 1)) + b;

		case EASE_OutExpo: 
			return c * (-pow(2, -10 * t / d) + 1) + b;

		case EASE_InOutExpo: 
			t /= d / 2;
			if (t < 1) return c / 2 * pow(2, 10 * (t - 1)) + b;
			t--;
			return c / 2 * (-pow(2, -10 * t) + 2) + b;

		case EASE_InCirc: 
			t /= d;
			return -c * (sqrt(1 - t*t) - 1) + b;

		case EASE_OutCirc: 
			t /= d;
			t--;
			return c * sqrt(1 - t*t) + b;

		case EASE_InOutCirc: 
			t /= d / 2;
			if (t < 1) return -c / 2 * (sqrt(1 - t*t) - 1) + b;
			t -= 2;
			return c / 2 * (sqrt(1 - t*t) + 1) + b;
	}

	return 0;
}

//
// Vectors
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
		float r, g, b;
	};

	struct {
		Vec2 xy;
		// float z;
	};

	struct {
		float nothing;
		// float x;
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
		// int z;
	};

	struct {
		int nothing;
		// int x;
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

	struct {
		Vec3 rgb;
		float a;
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
		float left;
		float bottom;
		float right;
		float top;
	};

	struct {
		float l;
		float b;
		float r;
		float t;
	};

	float e[4];
};

union Recti {
	struct {
		Vec2i min;
		Vec2i max;
	};

	struct {
		Vec2i cen;
		Vec2i dim;
	};

	struct {
		int left;
		int bottom;
		int right;
		int top;
	};

	struct {
		int l;
		int b;
		int r;
		int t;
	};

	int e[4];
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

	float e[6];
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

	float e[6];
};

union Quat {
	struct {
		float w, x, y, z;
	};

	struct {
		float nothing;
		Vec3 xyz;
	};

	float e[4];
};

//
// @Vec2
//

inline Vec2 vec2(float a, float b) { return {a, b}; };
inline Vec2 vec2(float a) { return {a, a}; };

inline Vec2 vec2(Vec2i a) { return {a.x, a.y}; }
inline Vec2 vec2(Vec3 a) { return {a.x, a.y}; }

inline Vec2 operator*(Vec2 a, float b) { return {a.x*b, a.y*b}; };
inline Vec2 operator*(float b, Vec2 a) { return {a.x*b, a.y*b}; };
inline Vec2 operator*(Vec2 a, Vec2 b) { return {a.x*b.x, a.y*b.y}; };
inline Vec2 & operator*=(Vec2& a, Vec2 b) { return a = a * b; };
inline Vec2 & operator*=(Vec2& a, float b) { return a = a * b; };

inline Vec2 operator+(Vec2 a, float b) { return {a.x+b, a.y+b}; };
inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x+b.x, a.y+b.y}; };
inline Vec2 & operator+=(Vec2& a, Vec2 b) { return a = a + b; };
inline Vec2 & operator+=(Vec2& a, float b) { return a = a + b; };

inline Vec2 operator-(Vec2 a) { return {-a.x, -a.y}; };
inline Vec2 operator-(Vec2 a, float b) { return {a.x-b, a.y-b}; };
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x-b.x, a.y-b.y}; };
inline Vec2 & operator-=(Vec2& a, Vec2 b) { return a = a - b; };
inline Vec2 & operator-=(Vec2& a, float b) { return a = a - b; };

inline Vec2 operator/(Vec2 a, float b) { return {a.x/b, a.y/b}; };
inline Vec2 operator/(Vec2 a, Vec2 b) { return {a.x/b.x, a.y/b.y}; };
inline Vec2 & operator/=(Vec2& a, Vec2 b) { return a = a / b; };
inline Vec2 & operator/=(Vec2& a, float b) { return a = a / b; };

inline bool operator==(Vec2 a, Vec2 b) { return (a.x == b.x) && (a.y == b.y); };
inline bool operator!=(Vec2 a, Vec2 b) { return !(a==b); };

inline Vec2 operator+(Vec2 a, Vec2i b) { return {a.x+b.x, a.y+b.y}; }

//

inline float dot(Vec2 a, Vec2 b) { return a.x*b.x + a.y*b.y; }
inline float dot(Vec2 a) { return dot(a,a); }
inline float len(Vec2 a) { return sqrt(dot(a)); };
inline float cross(Vec2 a, Vec2 b) { return a.x*b.y - a.y*b.x; };
inline Vec2 norm(Vec2 a) { return a/len(a); };
inline float determinant(Vec2 a, Vec2 b) { return cross(norm(a), norm(b)); }

inline float dotUnitToPercent(float dotProduct) { return 1 - acos(dotProduct)/M_PI_2; }

//

inline Vec2 clampMin(Vec2 v, Vec2 d) { return {max(v.x, d.x), max(v.y, d.y)}; }
inline Vec2 clampMax(Vec2 v, Vec2 d) { return {min(v.x, d.x), min(v.y, d.y)}; }
inline Vec2 clamp(Vec2 v, Rect r) { return {clamp(v.x, r.left, r.right), 
	                                        clamp(v.y, r.bottom, r.top)}; }
inline void clamp(Vec2* v, Rect r) { *v = clamp(*v, r); };

float mapRange(float v, Vec2 d, Vec2 r) { return mapRange(v, d.x, d.y, r.x, r.y); };
float mapRangeClamp(float v, Vec2 d, Vec2 r) { return mapRangeClamp(v, d.x, d.y, r.x, r.y); };
inline Vec2 mapRange(Vec2 a, Vec2 oldMin, Vec2 oldMax, Vec2 newMin, Vec2 newMax) {
	return {mapRange(a.x, oldMin.x, oldMax.x, newMin.x, newMax.x),
			mapRange(a.y, oldMin.y, oldMax.y, newMin.y, newMax.y)}; }

inline Vec2 lerp(float percent, Vec2 min, Vec2 max) {
	Vec2 result;
	result.x = lerp(percent, min.x, max.x);
	result.y = lerp(percent, min.y, max.y);
	return result;
}

Vec2 round(Vec2 a) { return {roundf(a.x), roundf(a.y)}; }

//

inline Vec2 rotateRight(Vec2 dir) { return vec2(dir.y, -dir.x); }
inline Vec2 rotateLeft(Vec2 dir) { return vec2(-dir.y, dir.x); }
inline float angleDirVec2(Vec2 dir) { return atan2(dir.y, dir.x); }
inline Vec2 angleToDir(float radians) { return vec2(cos(radians), sin(radians)); }

Vec2 rotateVec2(Vec2 a, float radian) {
	// clockwise

	float cs = cos(-radian);
	float sn = sin(-radian);

	float nx = a.x * cs - a.y * sn;
	float ny = a.x * sn + a.y * cs;

	return vec2(nx, ny);
}

inline float angleVec2(Vec2 dir1, Vec2 dir2) {
	float dt = dot(norm(dir1), norm(dir2));
	dt = clamp(dt, -1.0f, 1.0f);
	float angle = acos(dt);
	return angle;
}

//

inline float lenLine(Vec2 p0, Vec2 p1) { return len(p1 - p0); }

inline Vec2 lineMidPoint(Vec2 p1, Vec2 p2) {
	float x = (p1.x + p2.x)/2;
	float y = (p1.y + p2.y)/2;
	Vec2 midPoint = vec2(x,y);

	return midPoint;
}

inline int lineSide(Vec2 p1, Vec2 p2, Vec2 point) {
	int side = sign( (p2.x-p1.x)*(point.y-p1.y) - (p2.y-p1.y)*(point.x-p1.x) );
	return side;
}

inline Vec2 midOfTwoVectors(Vec2 p0, Vec2 p1, Vec2 p2) {
	Vec2 dir1 = norm(p0 - p1);
	Vec2 dir2 = norm(p2 - p1);
	Vec2 mid = lineMidPoint(dir1, dir2);
	mid = norm(mid);
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

float distancePointLine(Vec2 v1, Vec2 v2, Vec2 point, bool infinite = false) {
	Vec2 diff = v2 - v1;
    if (diff == vec2(0,0))
    {
    	diff = point - v1;
        return len(diff);
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

    return len(diff);
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

Vec2 projectPointOnLine(Vec2 p, Vec2 lp0, Vec2 lp1, bool clampDist = false) {

	Vec2 ld = norm(lp1 - lp0);
	float dist = (dot(p-lp0, ld) / dot(ld, ld));

	if(clampDist) dist = clamp(dist, 0.0f, len(lp1 - lp0));

	Vec2 result = lp0 + ld * dist;

	return result;
}

float sign(Vec2 p1, Vec2 p2, Vec2 p3) {
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool pointInTriangle(Vec2 pt, Vec2 v1, Vec2 v2, Vec2 v3) {
	bool b1, b2, b3;

	b1 = sign(pt, v1, v2) < 0.0f;
	b2 = sign(pt, v2, v3) < 0.0f;
	b3 = sign(pt, v3, v1) < 0.0f;

	return ((b1 == b2) && (b2 == b3));
}

Vec2 closestPointToTriangle(Vec2 p, Vec2 a, Vec2 b, Vec2 c) {
	bool insideTriangle = pointInTriangle(p, a, b, c);
	if(insideTriangle) return p;

	Vec2 p0 = projectPointOnLine(p, a, b, true);
	Vec2 p1 = projectPointOnLine(p, b, c, true);
	Vec2 p2 = projectPointOnLine(p, c, a, true);

	float d0 = len(p0 - p);
	float d1 = len(p1 - p);
	float d2 = len(p2 - p);

	float shortestDist = min(d0, d1, d2);

	Vec2 closestPoint = vec2(0,0);
	     if(shortestDist == d0) closestPoint = p0;
	else if(shortestDist == d1) closestPoint = p1;
	else if(shortestDist == d2) closestPoint = p2;

	return closestPoint;
}

inline bool lineCircleIntersection(Vec2 lp0, Vec2 lp1, Vec2 cp, float r, Vec2 * i) {
	Vec2 d = lp1 - lp0;
	Vec2 f = lp0 - cp;

	float a = dot(d,d);
	float b = 2*dot(f,d);
	float c = dot(f,f) - r*r;

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

		float lLine = len(p1 - p0);
		float lp0 = len(inter1 - p0);
		float lp1 = len(inter2 - p0);
		if(lp0 <= lp1) {
			i1 = inter1;
			i2 = inter2;
		} else {
			i1 = inter2;
			i2 = inter1;			
		}
		if(len(i1 - p0) > lLine) return false;
		// if(lp0 > lLine) return false;

		return true;
	}

	return false;
}

Vec2 ellipseNormal(Vec2 pos, float width, float height, Vec2 point) {
	Vec2 dir = vec2((point.x-pos.x)/pow(width,2), (point.y-pos.y)/pow(height,2));
	dir = -norm(dir);
	return dir;
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

//

inline Vec2 calculatePosition(Vec2 oldPosition, Vec2 velocity, Vec2 acceleration, float time) {
	oldPosition += 0.5f*acceleration*time*time + velocity*time;
	return oldPosition;
}

inline Vec2 calculateVelocity(Vec2 oldVelocity, Vec2 acceleration, float time) {
	oldVelocity += acceleration*time;
	return oldVelocity;
}

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

inline bool vecBetweenVecs(Vec2 v, Vec2 left, Vec2 right) {
	bool result;
	float ca = cross(left,v);
	float cb = cross(right,v);

	result = ca < 0 && cb > 0;
	return result;
}

Vec2 circumcenter(Vec2 a, Vec2 b, Vec2 c) {
	float x = c.y*(pow(b.x,2) + pow(b.y,2)) - b.y*(pow(c.x,2) + pow(c.y,2));
	float y = b.x*(pow(c.x,2) + pow(c.y,2)) - c.x*(pow(b.x,2) + pow(b.y,2));
	float d = 2*(b.x*c.y - b.y*c.x);
	x = x / d;
	y = y / d;

	Vec2 center = vec2(x,y);
	return center;
}

//

inline Vec2 quadraticBezierInterpolation(Vec2 p0, Vec2 p1, Vec2 p2, float t) {
	Vec2 pa = lerp(t, p0, p1);
	Vec2 pb = lerp(t, p1, p2);

	Vec2 v = lerp(t, pa, pb);
	return v;
}

inline Vec2 cubicBezierInterpolation(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, float t) {
	Vec2 pa = quadraticBezierInterpolation(p0, p1, p2, t);
	Vec2 pb = quadraticBezierInterpolation(p1, p2, p3, t);

	Vec2 v = lerp(t, pa, pb);
	return v;
}

inline Vec2 cubicBezierInterpolationSeemless(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, float t) {
	Vec2 v;
	v.x = cubicBezierInterpolationSeemless(p0.x, p1.x, p2.x, p3.x, t);
	v.y = cubicBezierInterpolationSeemless(p0.y, p1.y, p2.y, p3.y, t);
	return v;
}

inline float cubicBezierGuessLength(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3) {
	float length = lenLine(p0,p1) + lenLine(p1,p2) + lenLine(p2,p3) + lenLine(p3,p0);
	length = length / 2;
	return length;
}

void cubicBezierTesselate(Vec2* points, int* pointsCount, Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, float tolerance, int step = 0) {

	if(step == 0) *pointsCount = 0;

	if(step > 10) return;

	float d = distancePointLine(p0, p3, p1) + distancePointLine(p0, p3, p2);
	bool lineFlat = d < tolerance * lenLine(p0, p3);
	if(!lineFlat) {
		Vec2 p01 = lerp(0.5f, p0, p1);
		Vec2 p12 = lerp(0.5f, p1, p2);
		Vec2 p23 = lerp(0.5f, p2, p3);
		Vec2 p012 = lerp(0.5f, p01, p12);
		Vec2 p123 = lerp(0.5f, p12, p23);
		Vec2 p0123 = lerp(0.5f, p012, p123);

		if(step == 0) {
			if(points) points[(*pointsCount)++] = p0;
			else (*pointsCount)++;
		}

		cubicBezierTesselate(points, pointsCount, p0, p01, p012, p0123, tolerance, step+1);

		if(points) points[(*pointsCount)++] = p0123;
		else (*pointsCount)++;

		cubicBezierTesselate(points, pointsCount, p0123, p123, p23, p3, tolerance, step+1);

		if(step == 0) {
			if(points) points[(*pointsCount)++] = p3;
			else (*pointsCount)++;
		}

	} else {
		if(step == 0) {
			if(points) points[(*pointsCount)++] = p0;
			else (*pointsCount)++;
		}

		if(step == 0) {
			if(points) points[(*pointsCount)++] = p3;
			else (*pointsCount)++;
		}
	}
}


//
// @Vec2i
//

inline Vec2i vec2i(int a, int b) { return {a, b}; };
inline Vec2i vec2i(int a) { return {a, a}; };

inline Vec2i vec2i(Vec2 a) { return {a.x, a.y}; }

inline Vec2i operator*(Vec2i a, int b) { return {a.x*b, a.y*b}; };
inline Vec2i operator*(int b, Vec2i a) { return {a.x*b, a.y*b}; };
inline Vec2i operator*(Vec2i a, Vec2i b) { return {a.x*b.x, a.y*b.y}; };
inline Vec2i & operator*=(Vec2i& a, Vec2i b) { return a = a * b; };
inline Vec2i & operator*=(Vec2i& a, int b) { return a = a * b; };

inline Vec2i operator+(Vec2i a, int b) { return {a.x+b, a.y+b}; };
inline Vec2i operator+(Vec2i a, Vec2i b) { return {a.x+b.x, a.y+b.y}; };
inline Vec2i & operator+=(Vec2i& a, Vec2i b) { return a = a + b; };
inline Vec2i & operator+=(Vec2i& a, int b) { return a = a + b; };

inline Vec2i operator-(Vec2i a) { return {-a.x, -a.y}; };
inline Vec2i operator-(Vec2i a, int b) { return {a.x-b, a.y-b}; };
inline Vec2i operator-(Vec2i a, Vec2i b) { return {a.x-b.x, a.y-b.y}; };
inline Vec2i & operator-=(Vec2i& a, Vec2i b) { return a = a - b; };
inline Vec2i & operator-=(Vec2i& a, int b) { return a = a - b; };

inline Vec2i operator/(Vec2i a, int b) { return {a.x/b, a.y/b}; };
inline Vec2i operator/(Vec2i a, Vec2i b) { return {a.x/b.x, a.y/b.y}; };
inline Vec2i & operator/=(Vec2i& a, Vec2i b) { return a = a / b; };
inline Vec2i & operator/=(Vec2i& a, int b) { return a = a / b; };

inline bool operator==(Vec2i a, Vec2i b) { return (a.x == b.x) && (a.y == b.y); };
inline bool operator!=(Vec2i a, Vec2i b) { return !(a==b); };

//

inline Vec2i clampMin(Vec2i v, Vec2i d) { return {max(v.x, d.x), max(v.y, d.y)}; }
inline Vec2i clampMax(Vec2i v, Vec2i d) { return {min(v.x, d.x), min(v.y, d.y)}; }
inline Vec2i clamp(Vec2i v, Recti r) { return {clamp(v.x, r.left, r.right), 
	                                           clamp(v.y, r.bottom, r.top)}; }
inline void clamp(Vec2i* v, Recti r) { *v = clamp(*v, r); };

//
// @Vec3
//

inline Vec3 vec3(float a, float b, float c) { return {a, b, c}; };
inline Vec3 vec3(float a) { return {a, a, a}; };

inline Vec3 vec3(float a[3]) { return {a[0], a[1], a[2]}; }
inline Vec3 vec3(Vec3i a) { return {a.x, a.y, a.z}; }
inline Vec3 vec3(Vec2 a) { return {a.x, a.y, 0}; }
inline Vec3 vec3(Vec2 a, float b) { return {a.x, a.y, b}; }
inline Vec3 vec3(float b, Vec2 a) { return {b, a.x, a.y}; }

inline Vec3 operator*(Vec3 a, float b) { return {a.x*b, a.y*b, a.z*b}; };
inline Vec3 operator*(float b, Vec3 a) { return {a.x*b, a.y*b, a.z*b}; };
inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; };
inline Vec3 & operator*=(Vec3& a, Vec3 b) { return a = a * b; };
inline Vec3 & operator*=(Vec3& a, float b) { return a = a * b; };

inline Vec3 operator+(Vec3 a, float b) { return {a.x+b, a.y+b, a.z+b}; };
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; };
inline Vec3 & operator+=(Vec3& a, Vec3 b) { return a = a + b; };
inline Vec3 & operator+=(Vec3& a, float b) { return a = a + b; };

inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; };
inline Vec3 operator-(Vec3 a, float b) { return {a.x-b, a.y-b, a.z-b}; };
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; };
inline Vec3 & operator-=(Vec3& a, Vec3 b) { return a = a - b; };
inline Vec3 & operator-=(Vec3& a, float b) { return a = a - b; };

inline Vec3 operator/(Vec3 a, float b) { return {a.x/b, a.y/b, a.z/b}; };
inline Vec3 operator/(float b, Vec3 a) { return {b/a.x, b/a.y, b/a.z}; };
inline Vec3 operator/(Vec3 a, Vec3 b) { return {a.x/b.x, a.y/b.y, a.z/b.z}; };
inline Vec3 & operator/=(Vec3& a, Vec3 b) { return a = a / b; };
inline Vec3 & operator/=(Vec3& a, float b) { return a = a / b; };

inline bool operator==(Vec3 a, Vec3 b) { return (a.x==b.x) && (a.y==b.y) && (a.z==b.z); };
inline bool operator!=(Vec3 a, Vec3 b) { return !(a==b); };

//

inline float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float dot(Vec3 a) { return dot(a,a); }
inline float len(Vec3 a) { return sqrt(dot(a)); };
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, 
	                                        a.x*b.y-a.y*b.x}; }
inline Vec3 norm(Vec3 a) { return a/len(a); };

Vec3 lerp(float percent, Vec3 a, Vec3 b) {
	a.x = lerp(percent, a.x, b.x);
	a.y = lerp(percent, a.y, b.y);
	a.z = lerp(percent, a.z, b.z);

	return a;
}

//

Vec3 projectPointOnLine(Vec3 lPos, Vec3 lDir, Vec3 p) {
	Vec3 result;
	result = lPos + ((dot(p-lPos, lDir) / dot(lDir,lDir))) * lDir;
	return result;
}

int getBiggestAxis(Vec3 v, int smallerAxis[2] = 0) {
	float values[3] = {abs(v.x), abs(v.y), abs(v.z)};
	float maximum = max(values[0], values[1], values[2]);
	int biggestAxis;
	if(values[0] == maximum) biggestAxis = 0;
	else if(values[1] == maximum) biggestAxis = 1;
	else biggestAxis = 2;

	if(smallerAxis != 0) {
		int axis1 = mod(biggestAxis-1, 3);
		int axis2 = mod(biggestAxis+1, 3);
		smallerAxis[0] = v.e[biggestAxis] < 0 ? axis1 : axis2;
		smallerAxis[1] = v.e[biggestAxis] < 0 ? axis2 : axis1;
	}

	return biggestAxis;
}

inline Vec3 reflectVector(Vec3 dir, Vec3 normal) {
	Vec3 result = dir - 2*(dot(dir, normal))*normal;
	return result;
}

Vec3 randomUnitSphereDirection() {
	// Distribute in box and discard if not in sphere.

	Vec3 p;
	do {
		p = vec3(randomOffset(1), randomOffset(1), randomOffset(1));
	} while(len(p) > 1);

	p = norm(p);

	return p;
}

Vec3 randomUnitHalfSphereDirection(Vec3 dir) {
	Vec3 p = randomUnitSphereDirection();
	if(dot(dir, p) < 0) p = reflectVector(p, dir);

	return p;
}

Vec3 circumcenter(Vec3 a, Vec3 b, Vec3 c) {

	// This has been changed and is not tested.

	//		  |c-a|^2 [(b-a)x(c-a)]x(b-a) + |b-a|^2 (c-a)x[(b-a)x(c-a)]
	//m = a + ---------------------------------------------------------.
	//						    2 | (b-a)x(c-a) |^2

	//Vector3f a,b,c // are the 3 pts of the tri

	Vec3 ac = c - a;
	Vec3 ab = b - a;
	Vec3 abXac = cross(ab, ac);

	// this is the vector from a TO the circumsphere center

	Vec3 t1, t2;

	t1 = cross(abXac, ab);
	t2 = cross(ac, abXac);
	t1 *= pow(len(ac),2);
	t1 *= pow(len(ab),2);

	t1 += t2;
	t1 *= 1.0f/(pow(len(abXac),2) * 2);

	// The 3 space coords of the circumsphere center then:

	Vec3 toCircumsphereCenter = t1;
	Vec3 center = a + toCircumsphereCenter;
	return center;
}

void getPointsFromQuadAndNormal(Vec3 p, Vec3 normal, float size, Vec3 verts[4]) {
	int sAxis[2];
	int biggestAxis = getBiggestAxis(normal, sAxis);

	float s2 = size*0.5f;

	for(int i = 0; i < 4; i++) {
		Vec3 d = p;
		if(i == 0) { d.e[sAxis[0]] += -s2; d.e[sAxis[1]] += -s2; }
		else if(i == 1) { d.e[sAxis[0]] += -s2; d.e[sAxis[1]] +=  s2; }
		else if(i == 2) { d.e[sAxis[0]] +=  s2; d.e[sAxis[1]] +=  s2; }
		else if(i == 3) { d.e[sAxis[0]] +=  s2; d.e[sAxis[1]] += -s2; }
		verts[i] = d;
	}
}

//
// @Vec3i
//

inline Vec3i vec3i(int a, int b, int c) { return {a, b, c}; };
inline Vec3i vec3i(int a) { return {a, a, a}; };

inline Vec3i vec3i(Vec3 a) { return {a.x, a.y, a.z}; }
inline Vec3i vec3i(Vec2i a) { return {a.x, a.y, 0}; }
inline Vec3i vec3i(Vec2i a, int b) { return {a.x, a.y, b}; }

inline Vec3i operator*(Vec3i a, int b) { return {a.x*b, a.y*b, a.z*b}; };
inline Vec3i operator*(int b, Vec3i a) { return {a.x*b, a.y*b, a.z*b}; };
inline Vec3i operator*(Vec3i a, Vec3i b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; };
inline Vec3i & operator*=(Vec3i& a, Vec3i b) { return a = a * b; };
inline Vec3i & operator*=(Vec3i& a, int b) { return a = a * b; };

inline Vec3i operator+(Vec3i a, int b) { return {a.x+b, a.y+b, a.z+b}; };
inline Vec3i operator+(Vec3i a, Vec3i b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; };
inline Vec3i & operator+=(Vec3i& a, Vec3i b) { return a = a + b; };
inline Vec3i & operator+=(Vec3i& a, int b) { return a = a + b; };

inline Vec3i operator-(Vec3i a) { return {-a.x, -a.y, -a.z}; };
inline Vec3i operator-(Vec3i a, int b) { return {a.x-b, a.y-b, a.z-b}; };
inline Vec3i operator-(Vec3i a, Vec3i b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; };
inline Vec3i & operator-=(Vec3i& a, Vec3i b) { return a = a - b; };
inline Vec3i & operator-=(Vec3i& a, int b) { return a = a - b; };

inline Vec3i operator/(Vec3i a, int b) { return {a.x/b, a.y/b, a.z/b}; };
inline Vec3i operator/(Vec3i a, Vec3i b) { return {a.x/b.x, a.y/b.y, a.z/b.z}; };
inline Vec3i & operator/=(Vec3i& a, Vec3i b) { return a = a / b; };
inline Vec3i & operator/=(Vec3i& a, int b) { return a = a / b; };

inline bool operator==(Vec3i a, Vec3i b) { return (a.x==b.x) && (a.y==b.y) && (a.z==b.z); };
inline bool operator!=(Vec3i a, Vec3i b) { return !(a==b); };

//
// @Vec4
//

inline Vec4 vec4(float a, float b, float c, float d) { return {a, b, c, d}; }
inline Vec4 vec4(float a) { return {a, a, a, a}; };
inline Vec4 vec4(float a, float b) { return {a, a, a, b}; };

inline Vec4 vec4(Vec3 a, float b) { return {a.x, a.y, a.z, b}; }

inline Vec4 operator*(Vec4 a, float b) { return {a.x*b, a.y*b, a.z*b, a.w*b}; };
inline Vec4 operator*(float b, Vec4 a) { return {a.x*b, a.y*b, a.z*b, a.w*b}; };
inline Vec4 operator*(Vec4 a, Vec4 b) { return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w}; };
inline Vec4 & operator*=(Vec4& a, Vec4 b) { return a = a * b; };
inline Vec4 & operator*=(Vec4& a, float b) { return a = a * b; };

inline Vec4 operator+(Vec4 a, float b) { return {a.x+b, a.y+b, a.z+b, a.w+b}; };
inline Vec4 operator+(Vec4 a, Vec4 b) { return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; };
inline Vec4 & operator+=(Vec4& a, Vec4 b) { return a = a + b; };
inline Vec4 & operator+=(Vec4& a, float b) { return a = a + b; };

inline Vec4 operator-(Vec4 a) { return {-a.x, -a.y, -a.z, -a.w}; };
inline Vec4 operator-(Vec4 a, float b) { return {a.x-b, a.y-b, a.z-b, a.w-b}; };
inline Vec4 operator-(Vec4 a, Vec4 b) { return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; };
inline Vec4 & operator-=(Vec4& a, Vec4 b) { return a = a - b; };
inline Vec4 & operator-=(Vec4& a, float b) { return a = a - b; };

inline Vec4 operator/(Vec4 a, float b) { return {a.x/b, a.y/b, a.z/b, a.w/b}; };
inline Vec4 operator/(Vec4 a, Vec4 b) { return {a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w}; };
inline Vec4 & operator/=(Vec4& a, Vec4 b) { return a = a / b; };
inline Vec4 & operator/=(Vec4& a, float b) { return a = a / b; };

inline bool operator==(Vec4 a, Vec4 b) { return (a.x==b.x) && (a.y==b.y) && 
	                                            (a.z==b.z) && (a.w==b.w); };
inline bool operator!=(Vec4 a, Vec4 b) { return !(a==b); };

//
//
//

inline Mat4 operator*(Mat4 a, Mat4 b) {
	Mat4 r;
	int i = 0;
	for(int y = 0; y < 16; y += 4) {
		for(int x = 0; x < 4; x++) {
			r.e[i++] = a.e[y]*b.e[x]     + a.e[y+1]*b.e[x+4] + 
			           a.e[y+2]*b.e[x+8] + a.e[y+3]*b.e[x+12];
		}
	}
	return r;
}

inline Vec4 operator*(Mat4 m, Vec4 v) {
	Vec4 result;
	result.x = m.xa*v.x + m.xb*v.y + m.xc+v.z + m.xd+v.w;
	result.y = m.ya*v.x + m.yb*v.y + m.yc+v.z + m.yd+v.w;
	result.z = m.za*v.x + m.zb*v.y + m.zc+v.z + m.zd+v.w;
	result.w = m.wa*v.x + m.wb*v.y + m.wc+v.z + m.wd+v.w;

	return result;
}

inline Vec3 operator*(Mat4 m, Vec3 v) {
	Vec4 result = m*vec4(v,0);
	return result.xyz;
}

inline void rowToColumn(Mat4* m) {
	for(int x = 1; x < 4; x++) {
		for(int y = 0; y < x; y++) {
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

inline Mat4 scaleMatrix(Vec3 a) {
	Mat4 m = {};
	m.x1 = a.x;
	m.y2 = a.y;
	m.z3 = a.z;
	m.w4 = 1;

	return m;
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

inline Mat4 translationMatrix(Vec3 a) {
	Mat4 m = {};
	m.x1 = 1;
	m.y2 = 1;
	m.z3 = 1;
	m.w4 = 1;

	m.w1 = a.x;
	m.w2 = a.y;
	m.w3 = a.z;

	return m;
}

inline void viewMatrix(Mat4* m, Vec3 cPos, Vec3 cLook, Vec3 cUp, Vec3 cRight) {
	*m = {	cRight.x, cRight.y, cRight.z, -(dot(cPos,cRight)), 
			cUp.x, 	  cUp.y, 	cUp.z,    -(dot(cPos,cUp)), 
			cLook.x,  cLook.y,  cLook.z,  -(dot(cPos,cLook)), 
			0, 		  0, 		0, 		  1 };
}

inline Mat4 viewMatrix(Vec3 cPos, Vec3 cLook, Vec3 cUp, Vec3 cRight) {
	Mat4 m;
	viewMatrix(&m, cPos, cLook, cUp, cRight);

	return m;
}

inline void projMatrix(Mat4* m, float fov, float ar, float n, float f) {
	// ar -> w / h
	// fov -> vertical

	*m = { 	1/(ar*tan(fov*0.5f)), 0, 				 0, 			 0,
			0, 					  1/(tan(fov*0.5f)), 0, 			 0,
			0, 					  0, 				 -((f+n)/(f-n)), -((2*f*n)/(f-n)),
			0, 					  0, 				 -1, 			 0 };
}

inline Mat4 projMatrix(float fov, float ar, float n, float f) {
	Mat4 m;
	projMatrix(&m, fov, ar, n, f);
	
	return m;
}

Mat4 orthoMatrix(float l, float r, float b, float t, float n, float f) {
	Mat4 mat;
	float v[16] = { 2/(r-l), 0, 0, 0, 
					0, 2/(t-b), 0, 0, 
					0, 0, (-2)/(f-n), 0, 
					(-(r+l)/(r-l)), (-(t+b)/(t-b)), (f+n)/(f-n), 1};

	for(int i = 0; i < 16; ++i) mat.e[i] = v[i];

	return mat;
}

//
// @Quat
//

Quat quat() { return {1,0,0,0}; }
Quat quat(float w, float x, float y, float z) { return {w,x,y,z}; }
Quat quat(float a, Vec3 axis) {
	Quat r;
	r.w = cos(a*0.5f);
	r.x = axis.x * sin(a*0.5f);
	r.y = axis.y * sin(a*0.5f);
	r.z = axis.z * sin(a*0.5f);
	return r;
}

// Not the right name for this.
float quatDot(Quat q) {
	return (q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
}

float quatMagnitude(Quat q) {
	return sqrt(quatDot(q));
}

Quat quatNorm(Quat q) {
	Quat result;
	float m = quatMagnitude(q);
	return quat(q.w/m, q.x/m, q.y/m, q.z/m);
}

Quat quatConjugate(Quat q) {
	return quat(q.w, -q.x, -q.y, -q.z);
}

Quat quatScale(Quat q, float s) {
	return quat(q.w*s, q.x*s, q.y*s, q.z*s);
}

Quat quatInverse(Quat q) {
	return quatScale(quatConjugate(q), (1/quatDot(q)));
}

// quat*axis -> local rotation.
// axis*quat -> world rotation.
Quat operator*(Quat a, Quat b) {
	Quat r;
	r.w = (a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
	r.x = (a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y);
	r.y = (a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x);
	r.z = (a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w);
	return r;
}

// Does not make sense. Only used internally once.
Quat operator-(Quat a, Quat b) {
	Quat r;
	r.w = a.w - b.w;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	r.z = a.z - b.z;

	return r;
}
Quat operator+(Quat a, Quat b) {
	Quat r;
	r.w = a.w + b.w;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;

	return r;
}

Vec3 operator*(Quat q, Vec3 v) {
	Vec3 t = 2 * cross(q.xyz, v);
	Vec3 result = v + q.w * t + cross(q.xyz, t);

	return result;
}

bool operator==(Quat q0, Quat q1) {
	return q0.w==q1.w && q0.x==q1.x && q0.y==q1.y && q0.z==q1.z;
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

Mat4 quatRotationMatrix(Quat q) {
	float w = q.w, x = q.x, y = q.y, z = q.z;
	float x2 = x*x, y2 = y*y, z2 = z*z;
	float w2 = w*w;
	Mat4 m = {	w2+x2-y2-z2, 2*x*y-2*w*z, 2*x*z+2*w*y, 0,
				2*x*y+2*w*z, w2-x2+y2-z2, 2*y*z-2*w*x, 0,
				2*x*z-2*w*y, 2*y*z+2*w*x, w2-x2-y2+z2, 0,
				0, 			 0, 		  0, 		   1};

	return m;
}

Mat4 modelMatrix(Vec3 trans, Vec3 scale, float degrees = 0, Vec3 rot = vec3(0,0,0)) {
	Mat4 sm; scaleMatrix(&sm, scale);
	Mat4 rm; quatRotationMatrix(&rm, quat(degrees, rot));
	Mat4 tm; translationMatrix(&tm, trans);
	Mat4 model = tm*rm*sm;

	return model;
}

Vec3 rotateVec3(Vec3 v, float a, Vec3 axis) {
	Vec3 r = quat(a, axis)*v;
	return r;
}

void rotateVec3(Vec3* v, float a, Vec3 axis) {
	*v = rotateVec3(*v, a, axis);
}

Vec3 rotateVec3Around(Vec3 v, float a, Vec3 axis, Vec3 point) {
	Vec3 aroundOrigin = rotateVec3(v - point, a, axis);
	aroundOrigin += point;

	return aroundOrigin;
}

void rotateVec3Around(Vec3* v, float a, Vec3 axis, Vec3 point) {
	*v = rotateVec3Around(*v, a, axis, point);
}

Vec3 rotateVec3Around(Vec3 v, Quat q, Vec3 point) {
	Vec3 aroundOrigin = q * (v - point);
	aroundOrigin += point;

	return aroundOrigin;
}

void rotateVec3Around(Vec3* v, Quat q, Vec3 point) {
	*v = rotateVec3Around(*v, q, point);
}

// From Wikipedia.
Quat eulerAnglesToQuat(float pitch, float roll, float yaw) {
	Quat q;

    // Abbreviations for the various angular functions
	float cy = cos(yaw * 0.5);
	float sy = sin(yaw * 0.5);
	float cr = cos(roll * 0.5);
	float sr = sin(roll * 0.5);
	float cp = cos(pitch * 0.5);
	float sp = sin(pitch * 0.5);

	q.w = cy * cr * cp + sy * sr * sp;
	q.x = cy * sr * cp - sy * cr * sp;
	q.y = cy * cr * sp + sy * sr * cp;
	q.z = sy * cr * cp - cy * sr * sp;
	return q;
}
void quatToEulerAngles(Quat q, float* pitch, float* roll, float* yaw) {
	// roll (x-axis rotation)
	float sinr = +2.0 * (q.w * q.x + q.y * q.z);
	float cosr = +1.0 - 2.0 * (q.x * q.x + q.y * q.y);
	*roll = atan2(sinr, cosr);

	// pitch (y-axis rotation)
	float sinp = +2.0 * (q.w * q.y - q.z * q.x);
	if (fabs(sinp) >= 1)
		*pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
	else
		*pitch = asin(sinp);

	// yaw (z-axis rotation)
	float siny = +2.0 * (q.w * q.z + q.x * q.y);
	float cosy = +1.0 - 2.0 * (q.y * q.y + q.z * q.z);  
	*yaw = atan2(siny, cosy);
}

//
// @Rect
//

inline Rect  rect       () { return {0,0,0,0}; }
inline Rect  rect       (Vec2 min, Vec2 max) { return {min, max}; }
inline Rect  rect       (float left, float bottom, float right, float top) { return {left, bottom, right, top}; }
inline Rect  rectSides  (float left, float right, float bottom, float top) { return {left, bottom, right, top}; }

inline Rect  rectCenDim (Vec2 a, Vec2 d)    { return rect(a.x-d.w/2, a.y - d.h/2, a.x + d.w/2, a.y + d.h/2); }
inline Rect  rectBLDim  (Vec2 a, Vec2 d)    { return rect(a, a+d); };
inline Rect  rectTLDim  (Vec2 a, Vec2 d)    { return rect(a.x, a.y-d.h, a.x+d.w, a.y); };
inline Rect  rectTRDim  (Vec2 a, Vec2 d)    { return rect(a-d, a); };
inline Rect  rectBRDim  (Vec2 a, Vec2 d)    { return rect(a.x-d.w, a.y, a.x, a.y+d.h); };
inline Rect  rectLDim   (Vec2 a, Vec2 d)    { return rect(a.x, a.y-d.h/2, a.x+d.w, a.y+d.h/2); };
inline Rect  rectTDim   (Vec2 a, Vec2 d)    { return rect(a.x-d.w/2, a.y-d.h, a.x+d.w/2, a.y); };
inline Rect  rectRDim   (Vec2 a, Vec2 d)    { return rect(a.x-d.w, a.y-d.h/2, a.x, a.y+d.h/2); };
inline Rect  rectBDim   (Vec2 a, Vec2 d)    { return rect(a.x-d.w/2, a.y, a.x+d.w/2, a.y+d.h); };

inline Rect  rectCenDim (float x, float y, float w, float h) { return rectCenDim(vec2(x,y), vec2(w,h));};
inline Rect  rectBLDim  (float x, float y, float w, float h) { return rectBLDim(vec2(x,y), vec2(w,h)); };
inline Rect  rectTLDim  (float x, float y, float w, float h) { return rectTLDim(vec2(x,y), vec2(w,h)); };
inline Rect  rectTRDim  (float x, float y, float w, float h) { return rectTRDim(vec2(x,y), vec2(w,h)); };
inline Rect  rectBRDim  (float x, float y, float w, float h) { return rectBRDim(vec2(x,y), vec2(w,h)); };
inline Rect  rectLDim   (float x, float y, float w, float h) { return rectLDim(vec2(x,y), vec2(w,h)); };
inline Rect  rectTDim   (float x, float y, float w, float h) { return rectTDim(vec2(x,y), vec2(w,h)); };
inline Rect  rectRDim   (float x, float y, float w, float h) { return rectRDim(vec2(x,y), vec2(w,h)); };
inline Rect  rectBDim   (float x, float y, float w, float h) { return rectBDim(vec2(x,y), vec2(w,h)); };

inline float rectW      (Rect r)            { return r.right - r.left; };
inline float rectH      (Rect r)            { return r.top - r.bottom; };
inline float rectCenX   (Rect r)            { return r.left + rectW(r)/2; };
inline float rectCenY   (Rect r)            { return r.bottom + rectH(r)/2; };
inline Vec2  rectDim    (Rect r)            { return r.max - r.min; };
inline Vec2  rectCen    (Rect r)            { return r.min + rectDim(r)/2; };
inline Vec2  rectBL     (Rect r)            { return r.min; }
inline Vec2  rectL      (Rect r)            { return vec2(r.left, rectCen(r).y); }
inline Vec2  rectTL     (Rect r)            { return vec2(r.left, r.top); }
inline Vec2  rectT      (Rect r)            { return vec2(rectCen(r).x, r.top); }
inline Vec2  rectTR     (Rect r)            { return r.max; }
inline Vec2  rectR      (Rect r)            { return vec2(r.right, rectCen(r).y); }
inline Vec2  rectBR     (Rect r)            { return vec2(r.right, r.bottom); }
inline Vec2  rectB      (Rect r)            { return vec2(rectCen(r).x, r.bottom); }

inline Rect  rectSetCen (Rect r, Vec2 p)    { return rectCenDim(p, rectDim(r)); }
inline Rect  rectSetDim (Rect r, Vec2 d)    { return rectCenDim(rectCen(r), d); }
inline Rect  rectSetW   (Rect r, float p)   { return rectCenDim(rectCen(r), vec2(p, rectH(r))); }
inline Rect  rectSetH   (Rect r, float p)   { return rectCenDim(rectCen(r), vec2(rectW(r), p)); }
inline Rect  rectSetBL  (Rect r, Vec2 p)    { r.min = p; return r; }
inline Rect  rectSetTL  (Rect r, Vec2 p)    { r.left = p.x; r.top = p.y; return r; }
inline Rect  rectSetTR  (Rect r, Vec2 p)    { r.max = p; return r; }
inline Rect  rectSetBR  (Rect r, Vec2 p)    { r.right = p.x; r.bottom = p.y; return r; }
inline Rect  rectSetL   (Rect r, float p)   { r.left = p; return r; }
inline Rect  rectSetT   (Rect r, float p)   { r.top = p; return r; }
inline Rect  rectSetR   (Rect r, float p)   { r.right = p; return r; }
inline Rect  rectSetB   (Rect r, float p)   { r.bottom = p; return r; }

inline Rect  rectRSetL   (Rect r, float p)   { r.left = r.right - p; return r; }
inline Rect  rectRSetT   (Rect r, float p)   { r.top = r.bottom + p; return r; }
inline Rect  rectRSetR   (Rect r, float p)   { r.right = r.left + p; return r; }
inline Rect  rectRSetB   (Rect r, float p)   { r.bottom = r.top - p; return r; }

inline Rect  rectExpand (Rect r, Vec2 dim)  { return rect(r.min-dim/2, r.max+dim/2); }
inline Rect  rectExpand (Rect r, float s)   { return rect(r.min-vec2(s,s)/2, r.max+vec2(s,s)/2); }
inline Rect  rectTrans  (Rect r, Vec2 off)  { return rect(r.min+off, r.max+off); }
inline Rect  rectAddBL  (Rect r, Vec2 p)    { r.min += p; return r; }
inline Rect  rectAddTL  (Rect r, Vec2 p)    { r.left += p.x; r.top += p.y; return r; }
inline Rect  rectAddTR  (Rect r, Vec2 p)    { r.max += p; return r; }
inline Rect  rectAddBR  (Rect r, Vec2 p)    { r.right += p.x; r.bottom += p.y; return r; }
inline Rect  rectAddL   (Rect r, float p)   { r.left += p; return r; }
inline Rect  rectAddT   (Rect r, float p)   { r.top += p; return r; }
inline Rect  rectAddR   (Rect r, float p)   { r.right += p; return r; }
inline Rect  rectAddB   (Rect r, float p)   { r.bottom += p; return r; }

inline void  rectSetCen (Rect* r, Vec2 p)   { *r = rectCenDim(p, rectDim(*r)); }
inline void  rectSetDim (Rect* r, Vec2 d)   { *r = rectCenDim(rectCen(*r), d); }
inline void  rectSetW   (Rect* r, float p)  { *r = rectCenDim(rectCen(*r), vec2(p, rectH(*r))); }
inline void  rectSetH   (Rect* r, float p)  { *r = rectCenDim(rectCen(*r), vec2(rectW(*r), p)); }
inline void  rectSetBL  (Rect* r, Vec2 p)   { r->min = p; }
inline void  rectSetTL  (Rect* r, Vec2 p)   { r->left = p.x; r->top = p.y; }
inline void  rectSetTR  (Rect* r, Vec2 p)   { r->max = p; }
inline void  rectSetBR  (Rect* r, Vec2 p)   { r->right = p.x; r->bottom = p.y; }
inline void  rectSetL   (Rect* r, float p)  { r->left = p; }
inline void  rectSetT   (Rect* r, float p)  { r->top = p; }
inline void  rectSetR   (Rect* r, float p)  { r->right = p; }
inline void  rectSetB   (Rect* r, float p)  { r->bottom = p; }

inline void  rectRSetL  (Rect* r, float p)   { r->left = r->right - p; }
inline void  rectRSetT  (Rect* r, float p)   { r->top = r->bottom + p; }
inline void  rectRSetR  (Rect* r, float p)   { r->right = r->left + p; }
inline void  rectRSetB  (Rect* r, float p)   { r->bottom = r->top - p; }

inline void  rectExpand (Rect* r, Vec2 dim) { r->min -= dim/2; r->max += dim/2; }
inline void  rectExpand (Rect* r, float s)  { r->min -= vec2(s,s)/2; r->max += vec2(s,s)/2; }
inline void  rectTrans  (Rect* r, Vec2 off) { r->min += off; r->max += off; }
inline void  rectAddBL  (Rect* r, Vec2 p)   { r->min += p; }
inline void  rectAddTL  (Rect* r, Vec2 p)   { r->left += p.x; r->top += p.y; }
inline void  rectAddTR  (Rect* r, Vec2 p)   { r->max += p; }
inline void  rectAddBR  (Rect* r, Vec2 p)   { r->right += p.x; r->bottom += p.y; }
inline void  rectAddL   (Rect* r, float p)  { r->left += p; }
inline void  rectAddT   (Rect* r, float p)  { r->top += p; }
inline void  rectAddR   (Rect* r, float p)  { r->right += p; }
inline void  rectAddB   (Rect* r, float p)  { r->bottom += p; }

inline Rect rectCenDim(Rect r) {
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

Vec2 rectAlign(Rect r, Vec2i align) {
	     if(align == vec2i( 0, 0)) return rectCen(r);
	else if(align == vec2i(-1,-1)) return rectBL (r);
	else if(align == vec2i(-1, 0)) return rectL  (r);
	else if(align == vec2i(-1, 1)) return rectTL (r);
	else if(align == vec2i( 0, 1)) return rectT  (r);
	else if(align == vec2i( 1, 1)) return rectTR (r);
	else if(align == vec2i( 1, 0)) return rectR  (r);
	else if(align == vec2i( 1,-1)) return rectBR (r);
	else if(align == vec2i( 0,-1)) return rectB  (r);

	return vec2(0,0);
}

Rect rectAlignDim(Vec2 v, Vec2i align, Vec2 dim) {
	     if(align == vec2i(0,  0)) return rectCenDim( v, dim );
	else if(align == vec2i(-1,-1)) return rectBLDim ( v, dim );
	else if(align == vec2i(-1, 0)) return rectLDim  ( v, dim );
	else if(align == vec2i(-1, 1)) return rectTLDim ( v, dim );
	else if(align == vec2i( 0, 1)) return rectTDim  ( v, dim );
	else if(align == vec2i( 1, 1)) return rectTRDim ( v, dim );
	else if(align == vec2i( 1, 0)) return rectRDim  ( v, dim );
	else if(align == vec2i( 1,-1)) return rectBRDim ( v, dim );
	else if(align == vec2i( 0,-1)) return rectBDim  ( v, dim );

	return rect(0,0,0,0);
}

Rect rectAlignDim(Rect r, Vec2i align, Vec2 dim) {
	     if(align == vec2i(0,  0)) return rectCenDim( rectCen(r), dim );
	else if(align == vec2i(-1,-1)) return rectBLDim ( rectBL (r), dim );
	else if(align == vec2i(-1, 0)) return rectLDim  ( rectL  (r), dim );
	else if(align == vec2i(-1, 1)) return rectTLDim ( rectTL (r), dim );
	else if(align == vec2i( 0, 1)) return rectTDim  ( rectT  (r), dim );
	else if(align == vec2i( 1, 1)) return rectTRDim ( rectTR (r), dim );
	else if(align == vec2i( 1, 0)) return rectRDim  ( rectR  (r), dim );
	else if(align == vec2i( 1,-1)) return rectBRDim ( rectBR (r), dim );
	else if(align == vec2i( 0,-1)) return rectBDim  ( rectB  (r), dim );

	return r;
}

inline bool operator==(Rect r1, Rect r2) { return (r1.min == r2.min) && (r1.max == r2.max); }
inline bool operator!=(Rect r1, Rect r2) { return (r1.min != r2.min) && (r1.max != r2.max); }

inline bool rectIntersection(Rect r1, Rect r2) {
	bool hasIntersection = !(r2.min.x > r1.max.x ||
							 r2.max.x < r1.min.x ||
							 r2.max.y < r1.min.y ||
							 r2.min.y > r1.max.y);
	return hasIntersection;
}

Rect rectExpand(Rect r, float left, float bottom, float right, float top) {
	r.min.x += left;
	r.min.y += bottom;
	r.max.x += right;
	r.max.y += top;

	return r;
}

Rect rectIntersect(Rect r1, Rect r2) {
	bool hasIntersection = rectIntersection(r1, r2);
	Rect intersectionRect;
	if (hasIntersection) {
		intersectionRect.min.x = max(r1.min.x, r2.min.x);
		intersectionRect.max.x = min(r1.max.x, r2.max.x);
		intersectionRect.max.y = min(r1.max.y, r2.max.y);
		intersectionRect.min.y = max(r1.min.y, r2.min.y);
	} else intersectionRect = rect(0,0,0,0);
	
	return intersectionRect;
};

bool rectGetIntersection(Rect * intersectionRect, Rect r1, Rect r2) {
	bool hasIntersection = rectIntersection(r1, r2);
	if (hasIntersection) {
		intersectionRect->min.x = max(r1.min.x, r2.min.x);
		intersectionRect->max.x = min(r1.max.x, r2.max.x);
		intersectionRect->max.y = min(r1.max.y, r2.max.y);
		intersectionRect->min.y = max(r1.min.y, r2.min.y);
	}
	else *intersectionRect = rect(0,0,0,0);
	
	return hasIntersection;
};

bool pointInRect(Vec2 p, Rect r) {
	bool inRect = ( p.x >= r.min.x &&
					p.x <= r.max.x &&
					p.y >= r.min.y &&
					p.y <= r.max.y   );

	return inRect;
}

bool pointInRectEx(Vec2 p, Rect r) {
	bool inRect = ( p.x >= r.min.x &&
					p.x <  r.max.x &&
					p.y >= r.min.y &&
					p.y <  r.max.y   );

	return inRect;
}

bool rectEmpty(Rect r) {
	bool result = (r == rect(0,0,0,0));
	return result;
}

bool rectZero(Rect r) {
	return (rectW(r) <= 0.0f || rectH(r) <= 0.0f);
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

Vec2 rectDistancePos(Rect r, Vec2 p) {
	Vec2 result;
		 if(p.x >= r.max.x) result.x = p.x - r.max.x;
	else if(p.x <= r.min.x) result.x = p.x - r.min.x;
		 if(p.y >= r.max.y) result.y = p.y - r.max.y;
	else if(p.y <= r.min.y) result.y = p.y - r.min.y;
	return result;
}

Rect mapRange(Rect r, Rect oldInterp, Rect newInterp) {
	Rect result = r;
	result.min = mapRange(result.min, oldInterp.min, oldInterp.max, newInterp.min, newInterp.max);
	result.max = mapRange(result.max, oldInterp.min, oldInterp.max, newInterp.min, newInterp.max);

	return result;
}

Rect round(Rect r) {
	for(int i = 0; i < 4; i++) r.e[i] = roundf(r.e[i]);
	return r;
}

//
// @Recti
//

inline Recti recti() { return {0,0,0,0}; }
inline Recti recti(int left, int bottom, int right, int top) { 
	return {left,bottom,right,top}; 
}
inline Recti rectiRound(Rect r) {
	return recti(roundIntf(r.left), roundIntf(r.bottom), 
	             roundIntf(r.right), roundIntf(r.top));
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

Rect3 rect3Expand(Rect3 r, Vec3 dim) {
	r.min -= (dim/2);
	r.max += (dim/2);

	return r;
}

inline bool boxIntersection(Vec3 b1, Vec3 d1, Vec3 b2, Vec3 d2) {
	Vec3 min1 = b1 - d1/2.0f;
	Vec3 max1 = b1 + d1/2.0f;
	Vec3 min2 = b2 - d2/2.0f;
	Vec3 max2 = b2 + d2/2.0f;

	bool result = !( min2.x > max1.x ||
					 max2.x < min1.x ||
					 max2.y < min1.y ||
					 min2.y > max1.y ||
					 max2.z < min1.z ||
					 min2.z > max1.z);
	return result;
}

Vec3 boxNormals[6] = {vec3(-1,0,0), vec3(1,0,0), vec3(0,-1,0), vec3(0,1,0), vec3(0,0,-1), vec3(0,0,1)};
float boxRaycast(Vec3 lp, Vec3 ld, Vec3 boxPos, Vec3 boxDim, Vec3* intersection = 0, Vec3* intersectionNormal = 0, bool secondIntersection = false) {

	Vec3 boxHalfDim = boxDim/2;
	Vec3 boxMin = boxPos - boxHalfDim;
	Vec3 boxMax = boxPos + boxHalfDim;

	// ld is unit
	Vec3 dirfrac = 1.0f / ld;
	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	// r.org is origin of ray
	float t1 = (boxMin.x - lp.x)*dirfrac.x;
	float t2 = (boxMax.x - lp.x)*dirfrac.x;
	float t3 = (boxMin.y - lp.y)*dirfrac.y;
	float t4 = (boxMax.y - lp.y)*dirfrac.y;
	float t5 = (boxMin.z - lp.z)*dirfrac.z;
	float t6 = (boxMax.z - lp.z)*dirfrac.z;

	float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

	float distance;
	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behind us
	if (tmax < 0) return -1;

	// if tmin > tmax, ray doesn't intersect AABB
	if (tmin > tmax) return -1;

	distance = secondIntersection ? tmax : tmin;

	if(distance < 0) return -1;

	if(intersection) *intersection = lp + ld*distance;

	if(intersectionNormal) {
		     if(distance == t1) *intersectionNormal = boxNormals[0];
		else if(distance == t2) *intersectionNormal = boxNormals[1];
		else if(distance == t3) *intersectionNormal = boxNormals[2];
		else if(distance == t4) *intersectionNormal = boxNormals[3];
		else if(distance == t5) *intersectionNormal = boxNormals[4];
		else if(distance == t6) *intersectionNormal = boxNormals[5];
	}

	return distance;
}

//
// @Rect3i
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
// @Color.
//

inline int colorFloatToInt(float color) { return round(color * 255); };
inline float colorIntToFloat(int color) { return (1.0f / 255) * color; };

Vec3 rgbToHsl(double r, double g, double b) {
	Vec3 color;

	double M = 0.0, m = 0.0, c = 0.0;
	M = max(r, g, b);
	m = min(r, g, b);
	c = M - m;
	color.z = 0.5 * (M + m);
	if (c != 0.0) {
	 	      if (M == r) color.x = modf(((g - b) / c), 6.0);
		else  if (M == g) color.x = ((b - r) / c) + 2.0;
		else /*if(M==b)*/ color.x = ((r - g) / c) + 4.0;

		color.x *= 60.0;
		color.y = c / (1.0 - fabs(2.0 * color.z - 1.0));

	} else {
		color.x = 0;
		color.y = 0;
		color.z = r;
	}

	return color;
}

Vec3 hslToRgb(double h, double s, double l) {
	Vec3 color;
	double c = 0.0, m = 0.0, x = 0.0;
	c = (1.0 - fabs(2 * l - 1.0)) * s;
	m = 1.0 * (l - 0.5 * c);
	x = c * (1.0 - fabs(modf(h / 60.0, 2) - 1.0));
	if (h == 360) h = 0;
	     if (h >= 0.0 && h < 60)  color = vec3(c + m, x + m, m);
	else if (h >=  60 && h < 120) color = vec3(x + m, c + m, m);
	else if (h >= 120 && h < 180) color = vec3(m, c + m, x + m);
	else if (h >= 180 && h < 240) color = vec3(m, x + m, c + m);
	else if (h >= 240 && h < 300) color = vec3(x + m, m, c + m);
	else if (h >= 300 && h < 360) color = vec3(c + m, m, x + m);
	else color = vec3(m, m, m);

	return color;
}

inline Vec3 rgbToHsl(Vec3 c) { return rgbToHsl(c.r, c.g, c.b); }
inline Vec3 hslToRgb(Vec3 c) { return hslToRgb(c.x, c.y, c.z); }

inline Vec3 hslToRgbf(Vec3 hsl) {
	hsl.x = modf(hsl.x, 1.0f);
	hsl.y = clamp01(hsl.y);
	hsl.z = clamp01(hsl.z);

	Vec3 c = hslToRgb(360 * hsl.x, hsl.y, hsl.z);
	return c;
}
inline Vec3 hslToRgbf(float h, float s, float l) { return hslToRgbf(vec3(h,s,l)); }

inline Vec3 rgbToHslf(Vec3 rgb) {
	Vec3 hsl = rgbToHsl(rgb);
	Vec3 hslFloat = vec3(hsl.x / 360.0f, hsl.y, hsl.z);
	return hslFloat;
}

Vec3 linearToGamma(Vec3 color) {
	color.x = powf(color.x, 2.2f);
	color.y = powf(color.y, 2.2f);
	color.z = powf(color.z, 2.2f);
	return color;
}

Vec3 gammaToLinear(Vec3 color) {
	color.x = powf(color.x, 1/2.2f);
	color.y = powf(color.y, 1/2.2f);
	color.z = powf(color.z, 1/2.2f);
	return color;
}

Vec4 linearToGamma(Vec4 color) { return vec4(linearToGamma(color.rgb), color.a); }
Vec4 gammaToLinear(Vec4 color) { return vec4(gammaToLinear(color.rgb), color.a); }

// Taken from d3dx_dxgiformatconvert.inl.
inline float srgbToLinear(float val) {
    if( val < 0.04045f )
        val /= 12.92f;
    else
        val = pow((val + 0.055f)/1.055f,2.4f);
    return val;
}
inline float linearToSrgb(float val) { 
    if( val < 0.0031308f )
        val *= 12.92f;
    else
        val = 1.055f * pow(val,1.0f/2.4f) - 0.055f;
    return val;
}

//
//
//
