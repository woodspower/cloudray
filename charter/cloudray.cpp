//[header]
// A simple program to demonstrate how to implement Whitted-style ray-tracing
//[/header]
//[compile]
// Download the cloudray.cpp file to a folder.
// Open a shell/terminal, and run the following command where the files is saved:
//
// c++ -o cloudray cloudray.cpp -std=c++11 -O3
//
// Run with: ./cloudray. Open the file ./out.png in Photoshop or any program
// reading PPM files.
//[/compile]
//[ignore]
// Copyright (C) 2012  www.scratchapixel.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//[/ignore]


#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <utility>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <cstring>
#include <string>
#include <time.h>

#define VIEW_WIDTH  640
#define VIEW_HEIGHT 480
const float kInfinity = std::numeric_limits<float>::max();

class Vec3f {
public:
    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float xx) : x(xx), y(xx), z(xx) {}
    Vec3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    Vec3f operator * (const float &r) const { return Vec3f(x * r, y * r, z * r); }
    Vec3f operator * (const Vec3f &v) const { return Vec3f(x * v.x, y * v.y, z * v.z); }
    Vec3f operator - (const Vec3f &v) const { return Vec3f(x - v.x, y - v.y, z - v.z); }
    Vec3f operator + (const Vec3f &v) const { return Vec3f(x + v.x, y + v.y, z + v.z); }
    Vec3f operator - () const { return Vec3f(-x, -y, -z); }
    bool operator == (const float xx) const { return (x==xx && y==xx && z==xx); }
    Vec3f& operator += (const Vec3f &v) { x += v.x, y += v.y, z += v.z; return *this; }
    friend Vec3f operator * (const float &r, const Vec3f &v)
    { return Vec3f(v.x * r, v.y * r, v.z * r); }
    friend std::ostream & operator << (std::ostream &os, const Vec3f &v)
    { return os << v.x << ", " << v.y << ", " << v.z; }
    float x, y, z;
};

class Vec2f
{
public:
    Vec2f() : x(0), y(0) {}
    Vec2f(float xx) : x(xx), y(xx) {}
    Vec2f(float xx, float yy) : x(xx), y(yy) {}
    Vec2f operator * (const float &r) const { return Vec2f(x * r, y * r); }
    Vec2f operator + (const Vec2f &v) const { return Vec2f(x + v.x, y + v.y); }
    float x, y;
};

Vec3f normalize(const Vec3f &v)
{
    float mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (mag2 > 0) {
        float invMag = 1 / sqrtf(mag2);
        return Vec3f(v.x * invMag, v.y * invMag, v.z * invMag);
    }

    return v;
}

inline
float dotProduct(const Vec3f &a, const Vec3f &b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3f crossProduct(const Vec3f &a, const Vec3f &b)
{
    return Vec3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline
float clamp(const float &lo, const float &hi, const float &v)
{ return std::max(lo, std::min(hi, v)); }

inline
float deg2rad(const float &deg)
{ return deg * M_PI / 180; }

inline
float rad2deg(const float &rad)
{ return rad * 180 / M_PI; }

inline
Vec3f mix(const Vec3f &a, const Vec3f& b, const float &mixValue)
{ return a * (1 - mixValue) + b * mixValue; }

struct Options
{
    // samples per pixel
    uint32_t spp;
    uint32_t diffuseSpliter;
    uint32_t width;
    uint32_t height;
    float fov;
    float imageAspectRatio;
    uint8_t maxDepth;
    Vec3f backgroundColor;
    float bias;
    // viewpoint to cast the original rays
    Vec3f viewpoint;
};


class Light
{
public:
    Light(const Vec3f &p, const Vec3f &i) : position(p), intensity(i) {}
    Vec3f position;
    Vec3f intensity;
};

enum MaterialType { DIFFUSE_AND_GLOSSY, REFLECTION_AND_REFRACTION, REFLECTION };
enum RayStatus { VALID_RAY, NOHIT_RAY, INVISIBLE_RAY, OVERFLOW_RAY };
enum RayType { ORIG_RAYTYPE, RELECTION_RAYTYPE, REFRACTION_RAYTYPE, DIFFUSE_RAYTYPE };

// shade point on each object
struct Surface {
    Vec3f globalAmt;
    Vec3f N; // normal
    /* DEBUG info : vertion index and vertiacl index of this point */
    uint32_t   v;
    uint32_t   h;
};


class Object
{
 public:
    Object() :
        name("NO NAME"),
        materialType(DIFFUSE_AND_GLOSSY),
        ior(1.3), Kd(0.8), Ks(0.2), diffuseColor(0.2), specularExponent(25),
        verticalRes(0), horizonRes(0), verticalRange(0), horizonRange(0), pSurfaces(nullptr) {}
    virtual ~Object() {}
    virtual bool intersect(const Vec3f &, const Vec3f &, float &, uint32_t &, Vec2f &) const = 0;
    virtual Surface* getSurfaceProperties(const Vec3f &, const Vec3f &, const uint32_t &, const Vec2f &, Vec3f &, Vec2f &) const = 0;
    virtual Vec3f evalDiffuseColor(const Vec2f &) const { return diffuseColor; }
    virtual Vec3f pointRel2Abs(const Vec3f &) const =0;
    virtual Vec3f pointAbs2Rel(const Vec3f &) const =0;
    void setName(std::string objName)
    {
        name = objName;
    }
    std::string getName(void) {return name;}
    // store the pre-caculated shade value of each point
    void reset(void)
    {
        uint32_t size = sizeof(Surface) * verticalRange * horizonRange;
        float theta, phi;
        Surface *curr;
        std::memset(pSurfaces, 0, size);
        // DEBUG
        for (uint32_t v = 0; v < verticalRange; ++v) {
            for (uint32_t h = 0; h < horizonRange; ++h) {
                curr = (pSurfaces + v*horizonRange + h);
                curr->v = v;
                curr->h = h;
                // v(0, 1, 2, ... , 17) ==> theta(0, 10, 20, ..., 180)
                theta = deg2rad(std::min(180.f, (v+1)/verticalRes));
                phi = deg2rad(std::min(360.f, (h+1)/horizonRes));
                curr->N = Vec3f(sin(phi)*sin(theta), cos(theta), cos(phi)*sin(theta));
            }
        }
    }
    void initSurfaces(float vRes, float hRes, uint32_t vRange, uint32_t hRange)
    {
        verticalRes  = vRes;
        horizonRes   = hRes;
        verticalRange = vRange;
        horizonRange  = hRange;
        uint32_t size = sizeof(Surface) * verticalRange * horizonRange;
        pSurfaces = (Surface *)malloc(size);
        reset();
    }
    Surface *getSurfaceStaticProperties(float theta, float phi) const
    {
        uint32_t v,h;
        v = floor(theta*verticalRes);
        h = floor(phi*horizonRes);
//        std::printf("getSurfaceStaticProperties v(%d) h(%d)\n", v, h);
        if (v>=verticalRange) {
            std::printf("ERROR: obj(%s) getSurfaceStaticProperties v(%d) h(%d)\n", name.c_str(), v, h);
            v = verticalRange - 1 ;
            //return nullptr;
        }
        if (h>=horizonRange) {
            std::printf("ERROR: obj(%s) getSurfaceStaticProperties v(%d) h(%d)\n", name.c_str(), v, h);
            h = horizonRange - 1 ;
            //return nullptr;
        }
        return pSurfaces + v*horizonRange + h;

    }
    void dumpSurface(const Options &option) const
    {
        char outfile[256];
        std::sprintf(outfile,
            "obj[%s]_x.%d_y.%d_z.%d_fov.%d_dep.%d_spp.%d_split.%d.ppm", name.c_str(), (int)option.viewpoint.x,
            (int)option.viewpoint.y, (int)option.viewpoint.z, (int)option.fov, option.maxDepth, option.spp,
            option.diffuseSpliter);
        // save framebuffer to file
        std::ofstream ofs;
        /* text file for compare */
        ofs.open(outfile);
        ofs << "P3\n" << horizonRange << " " << verticalRange << "\n255\n";
        for (uint32_t j = 0; j < verticalRange; ++j) {
            for (uint32_t i = 0; i < horizonRange; ++i) {
                int r = (int)(255 * clamp(0, 1, (pSurfaces + j*horizonRange + i)->globalAmt.x));
                int g = (int)(255 * clamp(0, 1, (pSurfaces + j*horizonRange + i)->globalAmt.y));
                int b = (int)(255 * clamp(0, 1, (pSurfaces + j*horizonRange + i)->globalAmt.z));
                ofs << r << " " << g << " " << b << "\n ";
            }
        }
        ofs.close();

    }
    // material properties
    MaterialType materialType;
    float ior;
    float Kd, Ks;
    Vec3f diffuseColor;
    float specularExponent;
    std::string  name;
    // vertical resolution is the factor to split x from [min, max] or THETA from [0, 180]
    // we set the vertical resolution as r/10 by now.
    float verticalRes;
    // vertical point range is ceil(180*verticalRes)
    uint32_t verticalRange;
    // horizontal resolution is the factor to split y from [min, max] or PHI from [0, 360]
    // we set the horizontal resolution as r/10 by now.
    float horizonRes;
    // horizontal point range is ceil(360*horizonRange)
    uint32_t horizonRange;
    // the number point is verticalRange * horizonRange
    struct Surface * pSurfaces;
};

class Ray
{
public:
    Ray(const Vec3f &orig, const Vec3f &dir) : orig(orig), dir(dir){
        hitObject = nullptr;
        status = NOHIT_RAY;
        validCount = nohitCount = invisibleCount = overflowCount = 0;
    }
    Vec3f orig;
    Vec3f dir;
    // Hitted object of current ray
    Object *hitObject;
    Vec3f  hitPoint;
    // child reflection rays
    std::vector<std::unique_ptr<Ray>> reflectionLink;
    // child refraction rays
    std::vector<std::unique_ptr<Ray>> refractoinLink;
    // child diffuse rays
    std::vector<std::unique_ptr<Ray>> diffuseLink;

    // status of current ray
    enum RayStatus status;

    // Counter of valid child rays
    uint32_t validCount;
    // Counter of invalid rays as per nohit
    uint32_t nohitCount;
    // Counter of invisible diffuse reflection rays
    uint32_t invisibleCount;
    // Counter of invalid rays as per overflow
    uint32_t overflowCount;
};

struct RayStore
{
    // Pixel point of current processing original ray
    Vec3f currPixel;
    enum RayType currType;
    Ray *currRay;
//    struct Ray rays[VIEW_WIDTH][VIEW_HEIGHT];
    // creating the rays tracker
    std::vector<std::unique_ptr<Ray>> origViewLink[VIEW_WIDTH][VIEW_HEIGHT];

    // Counter of total memory to record rays
    uint64_t totalMem;
    // Counter of total rays, DO NOT include overflow rays
    uint32_t totalRays;
    // Counter of origin rays from eyes
    uint32_t originRays;
    // Counter of reflectoin rays
    uint32_t reflectionRays;
    // Counter of reflectoin rays
    uint32_t refractionRays;
    // Counter of diffuse reflectoin rays
    uint32_t diffuseRays;
    // Counter of invisible diffuse reflectoin rays
    uint32_t invisibleRays;
    // Counter of valid rays as per hitted
    //uint32_t hittedRays;
    // Counter of invalid rays as per overflow
    uint32_t overflowRays;
    // Counter of valid rays
    uint32_t validRays;
    // Counter of invalid rays
    uint32_t invalidRays;
    // Counter of invalid rays as per nohit
    uint32_t nohitRays;
};

bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) x0 = x1 = - 0.5 * b / a;
    else {
        float q = (b > 0) ?
            -0.5 * (b + sqrt(discr)) :
            -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) std::swap(x0, x1);
    return true;
}

class Sphere : public Object{
public:
    Sphere(const std::string name, const Vec3f &c, const float &r) : center(c), radius(r), radius2(r * r) 
    {
        float res = 0.025;
        setName(name);
        initSurfaces(r*res, r*res, ceil(180*r*res), ceil(360*r*res));
    }
    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear, uint32_t &index, Vec2f &uv) const
    {
        // analytic solution
        Vec3f L = orig - center;
        float a = dotProduct(dir, dir);
        float b = 2 * dotProduct(dir, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        tnear = t0;

        return true;
    }

    Vec3f pointRel2Abs(const Vec3f &rel) const
    {
        return center + rel*radius;
    }

    Vec3f pointAbs2Rel(const Vec3f &abs) const
    {
        return (abs - center) * (1/radius);
    }

    Surface* getSurfaceProperties(const Vec3f &P, const Vec3f &I, const uint32_t &index, const Vec2f &uv, Vec3f &N, Vec2f &st) const
    { 
        Surface *pStatic; // LEO: currently only static
        N = normalize(P - center);
        /* caculate the shadepoint on the surface */
/*
        uint32_t theta = floor(deg2rad(acos(N.z)));
        uint32_t phi = floor((atan2(N.y, N.x)));
*/
        float theta = rad2deg(acos(N.y));
        float phi = rad2deg(atan2(N.x, N.z));
        if (phi < 0) phi += 360;
        pStatic = getSurfaceStaticProperties(theta, phi);
        return pStatic;
    }

    Vec3f center;
    float radius, radius2;
};

bool rayTriangleIntersect(
    const Vec3f &v0, const Vec3f &v1, const Vec3f &v2,
    const Vec3f &orig, const Vec3f &dir,
    float &tnear, float &u, float &v)
{
    Vec3f edge1 = v1 - v0;
    Vec3f edge2 = v2 - v0;
    Vec3f pvec = crossProduct(dir, edge2);
    float det = dotProduct(edge1, pvec);
    if (det == 0 || det < 0) return false;

    Vec3f tvec = orig - v0;
    u = dotProduct(tvec, pvec);
    if (u < 0 || u > det) return false;

    Vec3f qvec = crossProduct(tvec, edge1);
    v = dotProduct(dir, qvec);
    if (v < 0 || u + v > det) return false;

    float invDet = 1 / det;
    
    tnear = dotProduct(edge2, qvec) * invDet;
    u *= invDet;
    v *= invDet;

    return true;
}

class MeshTriangle : public Object
{
public:
    MeshTriangle(
        const std::string name, 
        const Vec3f *verts,
        const uint32_t *vertsIndex,
        const uint32_t &numTris,
        const Vec2f *st)
    {
        setName(name);
        uint32_t maxIndex = 0;
        for (uint32_t i = 0; i < numTris * 3; ++i)
            if (vertsIndex[i] > maxIndex) maxIndex = vertsIndex[i];
        maxIndex += 1;
        vertices = std::unique_ptr<Vec3f[]>(new Vec3f[maxIndex]);
        memcpy(vertices.get(), verts, sizeof(Vec3f) * maxIndex);
        vertexIndex = std::unique_ptr<uint32_t[]>(new uint32_t[numTris * 3]);
        memcpy(vertexIndex.get(), vertsIndex, sizeof(uint32_t) * numTris * 3);
        numTriangles = numTris;
        stCoordinates = std::unique_ptr<Vec2f[]>(new Vec2f[maxIndex]);
        memcpy(stCoordinates.get(), st, sizeof(Vec2f) * maxIndex);
    }

    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear, uint32_t &index, Vec2f &uv) const
    {
        bool intersect = false;
        for (uint32_t k = 0; k < numTriangles; ++k) {
            const Vec3f & v0 = vertices[vertexIndex[k * 3]];
            const Vec3f & v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vec3f & v2 = vertices[vertexIndex[k * 3 + 2]];
            float t, u, v;
            if (rayTriangleIntersect(v0, v1, v2, orig, dir, t, u, v) && t < tnear) {
                tnear = t;
                uv.x = u;
                uv.y = v;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

    Vec3f pointRel2Abs(const Vec3f &rel) const
    {
        return rel;
    }

    Vec3f pointAbs2Rel(const Vec3f &abs) const
    {
        return abs;
    }

    Surface * getSurfaceProperties(const Vec3f &P, const Vec3f &I, const uint32_t &index, const Vec2f &uv, Vec3f &N, Vec2f &st) const
    {
        const Vec3f &v0 = vertices[vertexIndex[index * 3]];
        const Vec3f &v1 = vertices[vertexIndex[index * 3 + 1]];
        const Vec3f &v2 = vertices[vertexIndex[index * 3 + 2]];
        Vec3f e0 = normalize(v1 - v0);
        Vec3f e1 = normalize(v2 - v1);
        N = normalize(crossProduct(e0, e1));
        const Vec2f &st0 = stCoordinates[vertexIndex[index * 3]];
        const Vec2f &st1 = stCoordinates[vertexIndex[index * 3 + 1]];
        const Vec2f &st2 = stCoordinates[vertexIndex[index * 3 + 2]];
        st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
        return nullptr;
    }

    Vec3f evalDiffuseColor(const Vec2f &st) const
    {
        float scale = 5;
        float pattern = (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
        return mix(Vec3f(0.815, 0.235, 0.031), Vec3f(0.937, 0.937, 0.231), pattern);
    }

    std::unique_ptr<Vec3f[]> vertices;
    uint32_t numTriangles;
    std::unique_ptr<uint32_t[]> vertexIndex;
    std::unique_ptr<Vec2f[]> stCoordinates;
};

// [comment]
// Compute reflection direction
// [/comment]
Vec3f reflect(const Vec3f &I, const Vec3f &N)
{
    return I - 2 * dotProduct(I, N) * N;
}

// [comment]
// Compute diffuse direction
// tanScale is tan(output)/tan(input)
// we choose [-2,2] as typical tanscale
// [/comment]
Vec3f diffuse(const Vec3f &I, const Vec3f &N, const float tanScale)
{
    return tanScale * I - (tanScale + 1) * dotProduct(I, N) * N;
}

// [comment]
// Compute refraction direction using Snell's law
//
// We need to handle with care the two possible situations:
//
//    - When the ray is inside the object
//
//    - When the ray is outside.
//
// If the ray is outside, you need to make cosi positive cosi = -N.I
//
// If the ray is inside, you need to invert the refractive indices and negate the normal N
// [/comment]
Vec3f refract(const Vec3f &I, const Vec3f &N, const float &ior)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    Vec3f n = N;
    if (cosi < 0) { cosi = -cosi; } else { std::swap(etai, etat); n= -N; }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

// [comment]
// Compute Fresnel equation
//
// \param I is the incident view direction
//
// \param N is the normal at the intersection point
//
// \param ior is the mateural refractive index
//
// \param[out] kr is the amount of light reflected
// [/comment]
void fresnel(const Vec3f &I, const Vec3f &N, const float &ior, float &kr)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    if (cosi > 0) {  std::swap(etai, etat); }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        kr = 1;
    }
    else {
        float cost = sqrtf(std::max(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        kr = (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
}

// [comment]
// Returns true if the ray intersects an object, false otherwise.
//
// \param orig is the ray origin
//
// \param dir is the ray direction
//
// \param objects is the list of objects the scene contains
//
// \param[out] tNear contains the distance to the cloesest intersected object.
//
// \param[out] index stores the index of the intersect triangle if the interesected object is a mesh.
//
// \param[out] uv stores the u and v barycentric coordinates of the intersected point
//
// \param[out] *hitObject stores the pointer to the intersected object (used to retrieve material information, etc.)
//
// \param isShadowRay is it a shadow ray. We can return from the function sooner as soon as we have found a hit.
// [/comment]
bool trace(
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    float &tNear, uint32_t &index, Vec2f &uv, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vec2f uvK;
        if (objects[k]->intersect(orig, dir, tNearK, indexK, uvK) && tNearK < tNear) {
            *hitObject = objects[k].get();
            tNear = tNearK;
            //LEO: here is a bug in case of object Sphere, indexK will be a random value.
            index = indexK;
            uv = uvK;
        }
    }

    return (*hitObject != nullptr);
}

void dumpRay(
    const std::vector<std::unique_ptr<Ray>> &curr, uint32_t idx, uint32_t depth)
{
    #define MAX_DUMP_DEPTH 256 
    char prefix[MAX_DUMP_DEPTH*2] = "";
    if(depth >= MAX_DUMP_DEPTH) {
        std::printf("dumpRay out of stack\n");
        return;
    }

    std::printf("%*s%d orig(%f,%f,%f)------>direction(%f,%f,%f)*\n", depth, "#", depth,
                curr[idx]->orig.x, curr[idx]->orig.y, curr[idx]->orig.z, 
                curr[idx]->dir.x, curr[idx]->dir.y, curr[idx]->dir.z);
    if(curr[idx]->hitObject != nullptr) {
        std::printf("%*s%d hit object: %s, point(%f, %f, %f)\n", depth, "#", depth, 
                curr[idx]->hitObject->name.c_str(), 
                curr[idx]->hitPoint.x, curr[idx]->hitPoint.y, curr[idx]->hitPoint.y);

        for(uint32_t i=0; i<curr[idx]->diffuseLink.size(); i++) {
            std::printf("%*s%d diffuse%d:\n", depth+1, "#", depth+1, i);
            dumpRay(curr[idx]->diffuseLink, i, depth+1);
        }
    }
    else
        std::printf("%*s%d nohit\n", depth, "#", depth);
}

void dumpPix(
    RayStore &rayStore,
    uint32_t i,
    uint32_t j) 
{
    std::printf("***********dump rays of pixel(%d,%d)**************\n", i, j);
    dumpRay(rayStore.origViewLink[i][j], 0, 1);
/*
    auto it= rayStore.origViewLink[i][j].begin();
    std::printf("***********tracing ray(%d,%d): orig(%f,%f,%f)------>direction(%f,%f,%f)*************\n",
                i, j, curr[0]->orig.x, curr[0]->orig.y, curr[0]->orig.z, curr[0]->dir.x, curr[0]->dir.y, curr[0]->dir.z);
    if(curr[0]->hitObject != nullptr) {
        std::printf("hit object: %s\n", curr[0]->hitObject->name.c_str());
    }
*/
}

/* precast ray from light to object*/
Vec3f precastRay(
    RayStore &rayStore,
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    const Vec3f intensity,
    const Options &options,
    uint32_t depth,
    // index of object which is direct casted
    Object *targetObject=nullptr,
    Surface *targetSurface=nullptr,
    Vec3f targetPoint = 0)
{
    Ray * newRay = nullptr;
    Ray * currRay = nullptr;

    if (depth > options.maxDepth) {
        rayStore.overflowRays++;
        if(rayStore.currRay != nullptr) {
            rayStore.currRay->status = OVERFLOW_RAY;
            rayStore.currRay->overflowCount++;
        }
        return options.backgroundColor;
    }

    rayStore.totalRays++;
    
    Vec3f hitColor = options.backgroundColor;
    float tnear = kInfinity;
    Vec2f uv;
    Vec3f N; // normal
    Vec2f st; // st coordinates
    uint32_t index = 0;
    Object *hitObject = nullptr;
    Surface *hitSurface = nullptr;
    Vec3f hitPoint = 0;
    bool hitted = trace(orig, dir, objects, tnear, index, uv, &hitObject);
    if(hitted && depth >=1)
        std::printf("this should not happen.depth=%d, orig(%f,%f,%f)-->dir(%f,%f,%f), hitObject(%s)\n", 
                    depth, orig.x, orig.y, orig.z, dir.x, dir.y, dir.z, hitObject->name.c_str());
    if (targetObject != nullptr) {
        if (hitted && tnear <=1) {
      //      std::printf("targetPoint(%f,%f,%f) is in shadow of tnear(%f)\n", dir.x, dir.y, dir.z, tnear);
            rayStore.nohitRays++;
            if(rayStore.currRay != nullptr) {
                rayStore.currRay->status = NOHIT_RAY;
                rayStore.currRay->nohitCount++;
            }
            return hitColor;
        }
        hitObject = targetObject;
        hitSurface = targetSurface;
        hitPoint = targetPoint;
        tnear = 1.0;
    }
    else {
        if (!hitted) {
            rayStore.nohitRays++;
            if(rayStore.currRay != nullptr) {
                rayStore.currRay->status = NOHIT_RAY;
                rayStore.currRay->nohitCount++;
            }
            return hitColor;
        }
        Vec3f hitPoint = orig + dir * tnear;
        hitSurface = hitObject->getSurfaceProperties(hitPoint, dir, index, uv, N, st);
    }
    if (hitSurface == nullptr) {
        std::printf("BUG: ####targetSurface should not be NULL here\n");
        return hitColor;
    }

/*
        Vec3f relTgt = hitObject->pointAbs2Rel(orig + dir);
        Vec3f relHit = hitObject->pointAbs2Rel(hitPoint);
        std::printf("Hitpoint not match. tgtRel(%f,%f,%f), hitpointRel(%f,%f,%f), v(%d, %d), h(%d, %d), tnear(%f)\n", 
                    relTgt.x, relTgt.y, relTgt.z, relHit.x, relHit.y, relHit.z,
                    v, hitSurface->v, h, hitSurface->h, tnear);
        if(hitSurface->v != v || hitSurface->h != h) {
            std::printf("##########################################\n");
        }
*/

/*
    float kr;
    fresnel(dir, hitSurface->N, hitObject->ior, kr);
    if (intensity.x < 0 || intensity.y <0 || intensity.z < 0 || kr <0)
        std::printf("ERROR: intensity=(%f,%f,%f), kr=%f\n", intensity.x, intensity.y, intensity.z, kr);
    hitSurface->globalAmt += intensity * kr;
*/
    float kr = targetObject->Kd;
    Vec3f lightDir = hitPoint - orig;
    lightDir = normalize(lightDir);
    float LdotN = std::max(0.f, dotProduct(-lightDir, hitSurface->N));
    hitSurface->globalAmt += intensity * LdotN * kr;
    if(rayStore.currRay != nullptr) {
        rayStore.currRay->status = VALID_RAY;
        rayStore.currRay->validCount++;
        rayStore.currRay->hitObject = hitObject;
        rayStore.currRay->hitPoint = hitPoint;
    }

    switch (hitObject->materialType) {
        case REFLECTION_AND_REFRACTION:
        {
            Vec3f reflectionDirection = normalize(reflect(dir, hitSurface->N));
            Vec3f refractionDirection = normalize(refract(dir, hitSurface->N, hitObject->ior));
            Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, hitSurface->N) < 0) ?
                hitPoint - hitSurface->N * options.bias :
                hitPoint + hitSurface->N * options.bias;
            Vec3f refractionRayOrig = (dotProduct(refractionDirection, hitSurface->N) < 0) ?
                hitPoint - hitSurface->N * options.bias :
                hitPoint + hitSurface->N * options.bias;
            rayStore.reflectionRays++;
            // tracker the ray
            if(rayStore.currRay != nullptr) {
                newRay = new Ray(reflectionRayOrig, reflectionDirection);
                rayStore.totalMem += sizeof(Ray);
                rayStore.currRay->reflectionLink.push_back(std::unique_ptr<Ray>(newRay));
                currRay = rayStore.currRay;
                rayStore.currRay = newRay;
            }
            Vec3f reflectionColor = precastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, intensity*(1-kr), options, depth + 1);
            rayStore.currRay = currRay;
            rayStore.refractionRays++;
            // tracker the ray
            if(rayStore.currRay != nullptr) {
                newRay = new Ray(refractionRayOrig, refractionDirection);
                rayStore.totalMem += sizeof(Ray);
                rayStore.currRay->refractoinLink.push_back(std::unique_ptr<Ray>(newRay));
                currRay = rayStore.currRay;
                rayStore.currRay = newRay;
            }
            Vec3f refractionColor = precastRay(rayStore, refractionRayOrig, refractionDirection, objects, intensity*(1-kr), options, depth + 1);
            rayStore.currRay = currRay;
            hitColor = reflectionColor * kr + refractionColor * (1 - kr);
            break;
        }
        case REFLECTION:
        {
            Vec3f reflectionDirection = reflect(dir, hitSurface->N);
            Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, hitSurface->N) < 0) ?
                hitPoint - hitSurface->N * options.bias :
                hitPoint + hitSurface->N * options.bias;
            rayStore.reflectionRays++;
            // tracker the ray
            if(rayStore.currRay != nullptr) {
                newRay = new Ray(reflectionRayOrig, reflectionDirection);
                rayStore.totalMem += sizeof(Ray);
                rayStore.currRay->reflectionLink.push_back(std::unique_ptr<Ray>(newRay));
                currRay = rayStore.currRay;
                rayStore.currRay = newRay;
            }
            hitColor = precastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, intensity*(1-kr), options, depth + 1) * kr;
            rayStore.currRay = currRay;
            break;
        }
        default:
        {
            // Diffuse relfect
            // each relfect light will share part of the light
            // how many diffuse relfect light will be traced
            uint32_t count = options.diffuseSpliter;
            // Prevent memory waste before overflow
            if (depth+1 > options.maxDepth) {
                rayStore.overflowRays += count;
                if(rayStore.currRay != nullptr)
                    rayStore.currRay->overflowCount += count;
            }
            else {
                for (uint32_t i=1; i<=count; i++) {
                    Vec3f reflectionDirection = diffuse(dir, hitSurface->N, i*3.0/count);
                    //Vec3f reflectionDirection = normalize(reflect(dir, hitSurface->N));
                    Vec3f reflectionRayOrig = hitPoint + hitSurface->N * options.bias;
/*
                    Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, hitSurface->N) > 0) ?
                        hitPoint + hitSurface->N * options.bias :
                        hitPoint - hitSurface->N * options.bias;
*/
                    rayStore.diffuseRays++;
                    //std::printf("DEBUG--(%d, %d)-->Total rays(%d) = Origin rays(%d) + Reflection rays(%d) + Refraction rays(%d) + Diffuse rays(%d)\n", i, depth, rays.totalRays, rays.originRays, rays.reflectionRays, rays.refractionRays, rays.diffuseRays);
                    // tracker the ray
                    if(rayStore.currRay != nullptr) {
                        newRay = new Ray(reflectionRayOrig, reflectionDirection);
                        rayStore.totalMem += sizeof(Ray);
                        rayStore.currRay->diffuseLink.push_back(std::unique_ptr<Ray>(newRay));
                        currRay = rayStore.currRay;
                        rayStore.currRay = newRay;
                    }
                    precastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, intensity*(1-kr), options, depth + 1);
                    rayStore.currRay = currRay;
                }
            }
            
            break;
        }
    }
    return hitColor;
}
// [comment]
// Implementation of the Whitted-syle light transport algorithm (E [S*] (D|G) L)
//
// This function is the function that compute the color at the intersection point
// of a ray defined by a position and a direction. Note that thus function is recursive (it calls itself).
//
// If the material of the intersected object is either reflective or reflective and refractive,
// then we compute the reflection/refracton direction and cast two new rays into the scene
// by calling the castRay() function recursively. When the surface is transparent, we mix
// the reflection and refraction color using the result of the fresnel equations (it computes
// the amount of reflection and refractin depending on the surface normal, incident view direction
// and surface refractive index).
//
// If the surface is duffuse/glossy we use the Phong illumation model to compute the color
// at the intersection point.
// [/comment]
Vec3f castRay(
    RayStore &rayStore,
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    const std::vector<std::unique_ptr<Light>> &lights,
    const Options &options,
    uint32_t depth,
    bool test = false)
{
/*
    uint32_t  xPos = (uint32_t)rayStore.currPixel.x;
    uint32_t  yPos = (uint32_t)rayStore.currPixel.y;
*/
    Ray * newRay = nullptr;
    Ray * currRay = nullptr;

    if (depth > options.maxDepth) {
        rayStore.overflowRays++;
        if(rayStore.currRay != nullptr) {
            rayStore.currRay->status = OVERFLOW_RAY;
            rayStore.currRay->overflowCount++;
        }
        return options.backgroundColor;
    }

    rayStore.totalRays++;
    
    Vec3f hitColor = options.backgroundColor;
    Vec3f globalAmt = 0;
    float tnear = kInfinity;
    Vec2f uv;
    uint32_t index = 0;
    Object *hitObject = nullptr;
    Surface * hitSurface = nullptr;
    if (trace(orig, dir, objects, tnear, index, uv, &hitObject)) {
        Vec3f hitPoint = orig + dir * tnear;
        Vec3f N; // normal
        Vec2f st; // st coordinates
        hitSurface = hitObject->getSurfaceProperties(hitPoint, dir, index, uv, N, st);
        if (hitSurface != nullptr)
            globalAmt = hitSurface->globalAmt;
        if(rayStore.currRay != nullptr) {
            rayStore.currRay->status = VALID_RAY;
            rayStore.currRay->validCount++;
            rayStore.currRay->hitObject = hitObject;
            rayStore.currRay->hitPoint = hitPoint;
        }
        switch (hitObject->materialType) {
            case REFLECTION_AND_REFRACTION:
            {
                Vec3f reflectionDirection = normalize(reflect(dir, N));
                Vec3f refractionDirection = normalize(refract(dir, N, hitObject->ior));
                Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                    hitPoint - N * options.bias :
                    hitPoint + N * options.bias;
                Vec3f refractionRayOrig = (dotProduct(refractionDirection, N) < 0) ?
                    hitPoint - N * options.bias :
                    hitPoint + N * options.bias;
                rayStore.reflectionRays++;
                // tracker the ray
                if(rayStore.currRay != nullptr) {
                    newRay = new Ray(reflectionRayOrig, reflectionDirection);
                    rayStore.totalMem += sizeof(Ray);
                    rayStore.currRay->reflectionLink.push_back(std::unique_ptr<Ray>(newRay));
                    currRay = rayStore.currRay;
                    rayStore.currRay = newRay;
                }
                Vec3f reflectionColor = castRay(rayStore, reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1);
                rayStore.currRay = currRay;
                rayStore.refractionRays++;
                // tracker the ray
                if(rayStore.currRay != nullptr) {
                    newRay = new Ray(refractionRayOrig, refractionDirection);
                    rayStore.totalMem += sizeof(Ray);
                    rayStore.currRay->refractoinLink.push_back(std::unique_ptr<Ray>(newRay));
                    currRay = rayStore.currRay;
                    rayStore.currRay = newRay;
                }
                Vec3f refractionColor = castRay(rayStore, refractionRayOrig, refractionDirection, objects, lights, options, depth + 1);
                rayStore.currRay = currRay;
                float kr;
                fresnel(dir, N, hitObject->ior, kr);
                hitColor = reflectionColor * kr + refractionColor * (1 - kr);
                break;
            }
            case REFLECTION:
            {
                float kr;
                fresnel(dir, N, hitObject->ior, kr);
                Vec3f reflectionDirection = reflect(dir, N);
                Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                    hitPoint - N * options.bias :
                    hitPoint + N * options.bias;
                rayStore.reflectionRays++;
                // tracker the ray
                if(rayStore.currRay != nullptr) {
                    newRay = new Ray(reflectionRayOrig, reflectionDirection);
                    rayStore.totalMem += sizeof(Ray);
                    rayStore.currRay->reflectionLink.push_back(std::unique_ptr<Ray>(newRay));
                    currRay = rayStore.currRay;
                    rayStore.currRay = newRay;
                }
                hitColor = castRay(rayStore, reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1) * kr;
                rayStore.currRay = currRay;
                break;
            }
            default:
            {
#if 0
                // Diffuse relfect
                // each relfect light will share part of the light
                // how many diffuse relfect light will be traced
                uint32_t count = options.diffuseSpliter;
                float weight = 0.1/(count+1);
                //float weight = 0.001;
                hitColor = 0;
                // Prevent memory waste before overflow
                if (depth+1 > options.maxDepth) {
                    rayStore.overflowRays += count;
                    if(rayStore.currRay != nullptr)
                        rayStore.currRay->overflowCount += count;
                }
                else {
                    for (uint32_t i=1; i<=count; i++) {
                        Vec3f reflectionDirection = diffuse(dir, N, i*3.0/count);
                        //Vec3f reflectionDirection = normalize(reflect(dir, N));
                        Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, N) > 0) ?
                            hitPoint + N * options.bias :
                            hitPoint - N * options.bias;
                        rayStore.diffuseRays++;
                        //std::printf("DEBUG--(%d, %d)-->Total rays(%d) = Origin rays(%d) + Reflection rays(%d) + Refraction rays(%d) + Diffuse rays(%d)\n", i, depth, rays.totalRays, rays.originRays, rays.reflectionRays, rays.refractionRays, rays.diffuseRays);
                        // tracker the ray
                        if(rayStore.currRay != nullptr) {
                            newRay = new Ray(reflectionRayOrig, reflectionDirection);
                            rayStore.totalMem += sizeof(Ray);
                            rayStore.currRay->diffuseLink.push_back(std::unique_ptr<Ray>(newRay));
                            currRay = rayStore.currRay;
                            rayStore.currRay = newRay;
                        }
                        castRay(rayStore, reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1);
                        rayStore.currRay = currRay;
                        globalAmt += deltaAmt * powf(weight, depth+1);
                    }
                }
#endif

                Vec3f lightAmt = 0, specularColor = 0;
                // [comment]
                // We use the Phong illumation model int the default case. The phong model
                // is composed of a diffuse and a specular reflection component.
                // [/comment]
                Vec3f shadowPointOrig = (dotProduct(dir, N) < 0) ?
                    hitPoint + N * options.bias :
                    hitPoint - N * options.bias;
                // [comment]
                // Loop over all lights in the scene and sum their contribution up
                // We also apply the lambert cosine law here though we haven't explained yet what this means.
                // [/comment]
                for (uint32_t i = 0; i < lights.size(); ++i) {
                    Vec3f lightDir = lights[i]->position - hitPoint;
                    lightDir = normalize(lightDir);
#if 0
                    // square of the distance between hitPoint and the light
                    float lightDistance2 = dotProduct(lightDir, lightDir);
                    float LdotN = std::max(0.f, dotProduct(lightDir, N));
                    Object *shadowHitObject = nullptr;
                    float tNearShadow = kInfinity;
                    // is the point in shadow, and is the nearest occluding object closer to the object than the light itself?
                    bool inShadow = trace(shadowPointOrig, lightDir, objects, tNearShadow, index, uv, &shadowHitObject) &&
                        tNearShadow * tNearShadow < lightDistance2;
/*
                    rayStore.diffuseRays++;
                    rayStore.totalRays++;
*/
                    lightAmt += (1 - inShadow) * lights[i]->intensity * LdotN;
#endif
                    Vec3f reflectionDirection = reflect(-lightDir, N);
                    specularColor += powf(std::max(0.f, -dotProduct(reflectionDirection, dir)), hitObject->specularExponent) * lights[i]->intensity;
                }
/*
                if (lightAmt == 0) rayStore.invisibleRays++;
                else rayStore.validRays++;
*/
                //LEO: debug, donot caculate specular now.
                //hitColor = (globalAmt + lightAmt) * hitObject->evalDiffuseColor(st) * hitObject->Kd;
                hitColor = (globalAmt) * hitObject->evalDiffuseColor(st) + specularColor * hitObject->Ks;
                //hitColor = (lightAmt) * hitObject->evalDiffuseColor(st) * hitObject->Kd + specularColor * hitObject->Ks;
                
                break;
            }
        }
    }
    else {
        rayStore.nohitRays++;
        if(rayStore.currRay != nullptr) {
            rayStore.currRay->status = NOHIT_RAY;
            rayStore.currRay->nohitCount++;
        }
    }

    return hitColor;
}


/* prerender the object from light */
void prerender(
    RayStore &rayStore,
    const Options &options,
    const std::vector<std::unique_ptr<Object>> &objects,
    const std::vector<std::unique_ptr<Light>> &lights)
{
    Object *targetObject;
    Surface *targetSurface;
    Vec3f   targetPoint;
    Vec3f   testPoint;
    Vec3f orig = 0;
    uint32_t v=0, h=0, vMax=0, hMax=0;
    float vRes=0.f, hRes=0.f;
    for (uint32_t i=0; i<objects.size(); i++) {
        objects[i]->reset();
    }
    for (uint32_t l=0; l<lights.size(); l++) {
        orig = lights[l]->position;
        for (uint32_t i=0; i<objects.size(); i++) {
            if (objects[i]->name != "sph1")
                continue;
            if (objects[i]->pSurfaces == nullptr)
                continue;
            vMax = objects[i]->verticalRange;
            hMax = objects[i]->horizonRange;
            vRes = objects[i]->verticalRes;
            hRes = objects[i]->horizonRes;
            targetObject = objects[i].get();
            for (v=0; v<vMax; v++) {
                for (h=0; h<hMax; h++) {
                targetSurface = targetObject->pSurfaces + v*hMax + h;
                // dir should be relative to orig
                // dir = center + P.rel(theta, phi)*radius - orig
                Vec3f relPoint = targetSurface->N;
    //            Vec3f dir = objects[i]->pointRel2Abs(Vec3f(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta))) - orig;
                // target point is abs address from world zero.    
                targetPoint = targetObject->pointRel2Abs(relPoint);
                // set the test point a little bit far away the center of sphere. test point is rel address from orig.
                testPoint = targetObject->pointRel2Abs(relPoint * (1.0 + options.bias)) - orig;
                rayStore.originRays++;
                // tracker the ray
    /*
                Ray * currRay = new Ray(orig, dir);
                rayStore.totalMem += sizeof(Ray);
                rayStore.currRay = currRay;
                rayStore.origViewLink[i][j].push_back(std::unique_ptr<Ray>(currRay));
                std::printf("v/h(%d,%d): targetPoint[%f,%f,%f], testPoint[%f,%f,%f]\n",
                            v, h, targetPoint.x, targetPoint.y, targetPoint.z,
                            testPoint.x,testPoint.y, testPoint.z);
    */
                precastRay(rayStore, orig, testPoint, objects, lights[l]->intensity, options, 0, targetObject, targetSurface, targetPoint);
                //std::printf("light[%d]:%.0f%%\r",l, (h*vMax+v)*100.0/(vMax*hMax));
                }
            }
            // dump object shadepoint as ppm file
            objects[i]->dumpSurface(options);
        }
    }
}



// [comment]
// The main render function. This where we iterate over all pixels in the image, generate
// primary rays and cast these rays into the scene. The content of the framebuffer is
// saved to a file.
// [/comment]
void render(
    RayStore &rayStore,
    char * outfile,
    const Options &options,
    const std::vector<std::unique_ptr<Object>> &objects,
    const std::vector<std::unique_ptr<Light>> &lights)
{
    Vec3f orig = options.viewpoint;
    Vec3f *framebuffer = new Vec3f[options.width * options.height];
    Vec3f *pix = framebuffer;
    float scale = tan(deg2rad(options.fov * 0.5));
    float imageAspectRatio = options.width / (float)options.height;
    //Vec3f orig(0);
    for (uint32_t j = 0; j < options.height; ++j) {
        for (uint32_t i = 0; i < options.width; ++i) {
            // generate primary ray direction
            float x = (2 * (i + 0.5) / (float)options.width - 1) * imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / (float)options.height) * scale;
            Vec3f dir = normalize(Vec3f(x, y, -1));
            rayStore.originRays++;
            rayStore.currPixel = {(float)i, (float)j, -1.0};
            // tracker the ray
            Ray * currRay = new Ray(orig, dir);
            rayStore.totalMem += sizeof(Ray);
            rayStore.currRay = currRay;
            rayStore.origViewLink[i][j].push_back(std::unique_ptr<Ray>(currRay));
            *(pix++) = castRay(rayStore, orig, dir, objects, lights, options, 0);
            std::printf("%.0f%%\r",(j*options.width+i)*100.0/(options.width * options.height));
        }
        //std::printf("%f\r",(j*1.0/options.height));
    }

    // save framebuffer to file
    std::ofstream ofs;
    /* text file for compare */
    ofs.open(outfile);
    ofs << "P3\n" << options.width << " " << options.height << "\n255\n";
    for (uint32_t j = 0; j < options.height; ++j) {
        for (uint32_t i = 0; i < options.width; ++i) {
            int r = (int)(255 * clamp(0, 1, framebuffer[j*options.width + i].x));
            int g = (int)(255 * clamp(0, 1, framebuffer[j*options.width + i].y));
            int b = (int)(255 * clamp(0, 1, framebuffer[j*options.width + i].z));
            ofs << r << " " << g << " " << b << "\n ";
        }
    }
    ofs.close();

/*
    ofs.open(outfile);

    ofs << "P6\n" << options.width << " " << options.height << "\n255\n";
    for (uint32_t i = 0; i < options.height * options.width; ++i) {
        char r = (char)(255 * clamp(0, 1, framebuffer[i].x));
        char g = (char)(255 * clamp(0, 1, framebuffer[i].y));
        char b = (char)(255 * clamp(0, 1, framebuffer[i].z));
        ofs << r << g << b;
    }
    ofs.close();
*/


    delete [] framebuffer;
}

// [comment]
// In the main function of the program, we create the scene (create objects and lights)
// as well as set the options for the render (image widht and height, maximum recursion
// depth, field-of-view, etc.). We then call the render function().
// [/comment]
int main(int argc, char **argv)
{
    // creating the scene (adding objects and lights)
    std::vector<std::unique_ptr<Object>> objects;
    std::vector<std::unique_ptr<Light>> lights;
    
    
/*
    //Sphere *sph1 = new Sphere(Vec3f(-1, 0, -12), 2);
    Sphere *sph1 = new Sphere(Vec3f(0, 0, -12), 3);
    sph1->ior = 1.5;
    sph1->materialType = REFLECTION_AND_REFRACTION;

*/
    Sphere *sph1 = new Sphere("sph1", Vec3f(0, 0, -6), 2);
/*
    sph1->ior = 1.5;
    sph1->materialType = REFLECTION_AND_REFRACTION;
*/
    sph1->materialType = DIFFUSE_AND_GLOSSY;
    sph1->diffuseColor = Vec3f(0.6, 0.7, 0.8);

/*
    Sphere *sph2 = new Sphere("sph2", Vec3f(9, -2, -15), 2);
    sph2->materialType = DIFFUSE_AND_GLOSSY;
    sph2->diffuseColor = Vec3f(0.6, 0.7, 0.8);

    Sphere *sph3 = new Sphere("sph3", Vec3f(-3, -2, -7), 1);
    sph3->materialType = DIFFUSE_AND_GLOSSY;
    sph3->diffuseColor = Vec3f(0.6, 0.7, 0.8);

    Sphere *sph4 = new Sphere("sph4", Vec3f(3, 2, -7), 1);
    sph4->materialType = DIFFUSE_AND_GLOSSY;
    sph4->diffuseColor = Vec3f(0.6, 0.7, 0.8);

    Sphere *sph5 = new Sphere("sph5", Vec3f(-3, -2, -15), 2);
    sph5->materialType = DIFFUSE_AND_GLOSSY;
    sph5->diffuseColor = Vec3f(0.6, 0.7, 0.8);

    Sphere *sph6 = new Sphere("sph6", Vec3f(9, -2, -8), 1);
    sph6->materialType = DIFFUSE_AND_GLOSSY;
    sph6->diffuseColor = Vec3f(0.6, 0.7, 0.8);
*/

/*
    Sphere *sph2 = new Sphere(Vec3f(-1, 0, -12), 2);
    sph2->materialType = DIFFUSE_AND_GLOSSY;
    sph2->diffuseColor = Vec3f(0.6, 0.7, 0.8);
*/

/*
    Sphere *sph3 = new Sphere(Vec3f(0, 0, -8), 2);
    sph3->ior = 1.5;
    sph3->materialType = REFLECTION_AND_REFRACTION;
*/
    
    objects.push_back(std::unique_ptr<Sphere>(sph1));
/*
    objects.push_back(std::unique_ptr<Sphere>(sph2));
    objects.push_back(std::unique_ptr<Sphere>(sph3));
    objects.push_back(std::unique_ptr<Sphere>(sph4));
    objects.push_back(std::unique_ptr<Sphere>(sph5));
    objects.push_back(std::unique_ptr<Sphere>(sph6));
*/
/*
    objects.push_back(std::unique_ptr<Sphere>(sph2));
    objects.push_back(std::unique_ptr<Sphere>(sph3));
*/

/*
    //Vec3f verts[4] = {{-5,-3,-6}, {5,-3,-6}, {5,-3,-16}, {-5,-3,-16}};
    Vec3f verts[4] = {{-10,-3,-6}, {10,-3,-6}, {10,-3,-16}, {-10,-3,-16}};
    uint32_t vertIndex[6] = {0, 1, 3, 1, 2, 3};
    Vec2f st[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    MeshTriangle *mesh = new MeshTriangle("mesh1", verts, vertIndex, 2, st);
    mesh->materialType = DIFFUSE_AND_GLOSSY;
    
    objects.push_back(std::unique_ptr<MeshTriangle>(mesh));
*/

//    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-20, 70, 20), 0.5)));
//    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(1, 1, 10), 1)));
//    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-30, 50, 12), 1)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(30, 50, 12), 1)));
//    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(30, -50, 12), 0.3)));
//    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-30, -50, 12), 0.3)));
//    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(0, 0, 12), 1)));
//    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(30, -50, 12), 1)));
    //lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(0, 0, -12), 5)));
/*
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(10, 50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(0, 50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-10, 50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-20, 50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-30, 50, -12), 0.3)));

    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(30, 5, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(20, 5, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(10, 5, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(0, 5, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-10, 5, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-20, 5, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-30, 5, -12), 0.3)));
    
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(30, -50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(20, -50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(10, -50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(0, -50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-10, -50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-20, -50, -12), 0.3)));
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-30, -50, -12), 0.3)));
*/

    // setting up options
    Options options[100];
    std::memset(options, 0, sizeof(options));
    // no diffuse at all
    options[0].diffuseSpliter = 0;
    options[0].maxDepth = 1;
    options[0].spp = 1;
    options[0].width = VIEW_WIDTH*options[0].spp;
    options[0].height = VIEW_HEIGHT*options[0].spp;
    options[0].fov = 90;
    //options[0].backgroundColor = Vec3f(0.235294, 0.67451, 0.843137);
    options[0].backgroundColor = Vec3f(0.0);
    //options[0].bias = 0.001;
    options[0].bias = 0.001;
    options[0].viewpoint = Vec3f(0.0);
    //Vec3f viewpoints[4] = {{0,0,0}, {0,0,1}, {1,1,1}, {6,6,1}};

#if 0
    // 10 diffuse at each first hitpoint, no further diffuse
    options[1].diffuseSpliter = 0;
    options[1].maxDepth = 0;
    options[1].spp = 1;
    options[1].width = VIEW_WIDTH*options[1].spp;
    options[1].height = VIEW_HEIGHT*options[1].spp;
    options[1].fov = 90;
    options[1].backgroundColor = Vec3f(0.0);
    options[1].bias = 0.0001;
    options[1].viewpoint = Vec3f(0.0);

    // 10 diffuse at each first hitpoint, 1 depth further diffuse
    options[2].diffuseSpliter = 1;
    options[2].maxDepth = 1;
    options[2].spp = 1;
    options[2].width = VIEW_WIDTH*options[2].spp;
    options[2].height = VIEW_HEIGHT*options[2].spp;
    options[2].fov = 90;
    options[2].backgroundColor = Vec3f(0.0);
    options[2].bias = 0.0001;
    options[2].viewpoint = Vec3f(0.0);

    // 10 diffuse at each first hitpoint, 3 depth further diffuse
    options[3].diffuseSpliter = 1;
    options[3].maxDepth = 3;
    options[3].spp = 1;
    options[3].width = VIEW_WIDTH*options[3].spp;
    options[3].height = VIEW_HEIGHT*options[3].spp;
    options[3].fov = 90;
    options[3].backgroundColor = Vec3f(0.0);
    options[3].bias = 0.0001;
    options[3].viewpoint = Vec3f(0.0);

    // 10 diffuse at each first hitpoint, 3 depth further diffuse
    options[4].diffuseSpliter = 1;
    options[4].maxDepth = 9;
    options[4].spp = 1;
    options[4].width = VIEW_WIDTH*options[4].spp;
    options[4].height = VIEW_HEIGHT*options[4].spp;
    options[4].fov = 90;
    options[4].backgroundColor = Vec3f(0.0);
    options[4].bias = 0.0001;
    options[4].viewpoint = Vec3f(0.0);

    // 10 diffuse at each first hitpoint, 3 depth further diffuse
    options[5].diffuseSpliter = 100;
    options[5].maxDepth = 1;
    options[5].spp = 1;
    options[5].width = VIEW_WIDTH*options[5].spp;
    options[5].height = VIEW_HEIGHT*options[5].spp;
    options[5].fov = 90;
    options[5].backgroundColor = Vec3f(0.0);
    options[5].bias = 0.0001;
    options[5].viewpoint = Vec3f(0.0);

    // 10 diffuse at each first hitpoint, 3 depth further diffuse
    options[6].diffuseSpliter = 1000;
    options[6].maxDepth = 1;
    options[6].spp = 1;
    options[6].width = VIEW_WIDTH*options[6].spp;
    options[6].height = VIEW_HEIGHT*options[6].spp;
    options[6].fov = 90;
    options[6].backgroundColor = Vec3f(0.0);
    options[6].bias = 0.0001;
    options[6].viewpoint = Vec3f(0.0);
#endif

    // setting up ray store
    RayStore rayStore;

    char outfile[256];

    std::printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", 
                "split", "depth", 
                "origin", "reflect", "refract", "diffuse", 
                "nohit", "invis", "overflow", "CPU(Rays)", "TIME(S)", "MEM(GB)");
    time_t start, end;
    //std::printf("split\t depth\t total\t origin\t reflect\t refract\t diffuse\t nohit\t invis\t overflow\t CPUConsumed\n");
    for (int i =0; i<sizeof(options)/sizeof(struct Options); i++){
        if(options[i].width == 0) break;

        // do prerender
        std::memset(&rayStore, 0, sizeof(rayStore));
        // caculate time consumed
        start = time(NULL);
        prerender(rayStore, options[i], objects, lights);
        end = time(NULL);
        std::printf("#%-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10.0f %-10.2f\n",
                    options[i].diffuseSpliter, options[i].maxDepth,
                    rayStore.originRays, rayStore.reflectionRays, rayStore.refractionRays, rayStore.diffuseRays,
                    rayStore.nohitRays, rayStore.invisibleRays, rayStore.overflowRays, 
                    rayStore.totalRays, difftime(end, start), rayStore.totalMem*1.0/(1024.0*1024.0*1024.0));


        std::memset(&rayStore, 0, sizeof(rayStore));
        std::sprintf(outfile,
            "map_x.%d_y.%d_z.%d_fov.%d_dep.%d_spp.%d_split.%d.ppm", (int)options[i].viewpoint.x,
            (int)options[i].viewpoint.y, (int)options[i].viewpoint.z, (int)options[i].fov, options[i].maxDepth, options[i].spp,
            options[i].diffuseSpliter);
        // caculate time consumed
        start = time(NULL);
        // finally, render
        std::memset(&rayStore, 0, sizeof(rayStore));
        render(rayStore, outfile, options[i], objects, lights);
        end = time(NULL);
        std::printf("%-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10.0f %-10.2f\n",
                    options[i].diffuseSpliter, options[i].maxDepth,
                    rayStore.originRays, rayStore.reflectionRays, rayStore.refractionRays, rayStore.diffuseRays,
                    rayStore.nohitRays, rayStore.invisibleRays, rayStore.overflowRays, 
                    rayStore.totalRays, difftime(end, start), rayStore.totalMem*1.0/(1024.0*1024.0*1024.0));
    }

/*
    rayStore.invalidRays = rayStore.overflowRays + rayStore.nohitRays;
    std::printf("Total rays(%d) = Origin rays(%d) + Reflection rays(%d) + Refraction rays(%d) + Diffuse rays(%d)\n", 
                 rayStore.totalRays, rayStore.originRays, rayStore.reflectionRays, rayStore.refractionRays, rayStore.diffuseRays);
    std::printf("Tracing result is:\n\
                 Valid rays: %d, Nohit rays: %d, Invisible rays: %d, OverflowRays: %d\n", 
                 rayStore.validRays, rayStore.nohitRays, rayStore.invisibleRays, rayStore.overflowRays);
*/
/*
    std::printf("Invalid rays reasons\n");
    for (uint32_t j = 0; j < options.height; ++j) {
        for (uint32_t i = 0; i < options.width; ++i) {
            if(rayStore.origViewLink[i][j][0] != nullptr && rayStore.origViewLink[i][j][0]->hitObject != nullptr) {
                std::printf("ray(%d,%d) hit object: %s\n", i, j, rayStore.origViewLink[i][j][0]->hitObject->name.c_str());
            }
        }
    }
*/
//    dumpPix(rayStore, 222, 340);

    return 0;
}