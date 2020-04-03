/*********************************************************
    Leo, lili 
    Prototype to verify cloud ray tracing
    Usage: c++ -O0 -g -std=c++11 -o cloudray.cpp cloudray 
*********************************************************/

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
#include <assert.h>
#include "geometry.h"

#define MY_UINT64_T     uint64_t
#define VIEW_WIDTH      640
#define VIEW_HEIGHT     480
#define RAY_CAST_DESITY 0.1
/*
#define LIGHT_NUM_MAX 3
#define OBJ_NUM_MAX   5
#define V_RES_MAX     3000
#define H_RES_MAX     3000
*/

#define OVERSTACK_PROTECT_DEPTH 9 
//#define INTENSITY_TOO_WEAK   0.01*0.01
#define INTENSITY_TOO_WEAK   0.001*0.001
const float kInfinity = std::numeric_limits<float>::max();

enum ObjectType { OBJECT_TYPE_NONE, OBJECT_TYPE_MESH, OBJECT_TYPE_SPHERE };
enum MaterialType { DIFFUSE_AND_GLOSSY, REFLECTION_AND_REFRACTION, REFLECTION };
enum RayStatus { VALID_RAY, NOHIT_RAY, INVISIBLE_RAY, OVERFLOW_RAY };
enum RayType { RAY_TYPE_ORIG, RAY_TYPE_REFLECTION, RAY_TYPE_REFRACTION, RAY_TYPE_DIFFUSE };
char RayTypeString[10][20] = {"orig", "reflect", "refract", "diffuse"};


/*
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
*/

// Leo: create a matrix by using lookat function
Matrix44f lookAt(const Vec3f& from, const Vec3f& to, const Vec3f& tmp = Vec3f(1, 0, 0)) 
{ 
    Vec3f up = (to - from); 
    up = up.normalize();
    Vec3f forward = tmp.crossProduct(up);
    Vec3f right = up.crossProduct(forward); 
    Matrix44f x; 
    x[0][0] = right.x; 
    x[0][1] = right.y; 
    x[0][2] = right.z; 
    x[1][0] = up.x; 
    x[1][1] = up.y; 
    x[1][2] = up.z; 
    x[2][0] = forward.x; 
    x[2][1] = forward.y; 
    x[2][2] = forward.z; 
     
    x[3][0] = from.x; 
    x[3][1] = from.y; 
    x[3][2] = from.z; 
    return x;
} 
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
{ 
    float deg = rad * 180 / M_PI; 
    if (deg < 0) 
        deg += 360.0;
    return deg;
}

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
    bool  doTraditionalRender;
    bool  doRenderAfterDiffusePreprocess;
    bool  doRenderAfterDiffuseAndReflectPreprocess;
    // a list of viewpoint to cast the original rays
    Vec3f viewpoints[100];
};

class Ray
{
public:
    Ray(const RayType rayType, const Vec3f &orig, const Vec3f &dir, const Vec3f &leftIntensity = -1) : orig(orig), dir(dir){
        hitObject = nullptr;
        status = NOHIT_RAY;
        type   = rayType;
        intensity = leftIntensity;
        inside    = false;
        validCount = nohitCount = invisibleCount = overflowCount = weakCount = 0;
    }
    Vec3f orig;
    Vec3f dir;
    // Hitted object of current ray
    void *hitObject;
    Vec3f  hitPoint;
    // The ray is inside or outside the object
    bool   inside;
    Vec3f  intensity;
    // child reflection rays
    std::vector<std::unique_ptr<Ray>> reflectionLink;
    // child refraction rays
    std::vector<std::unique_ptr<Ray>> refractionLink;
    // child diffuse rays
    std::vector<std::unique_ptr<Ray>> diffuseLink;

    // status of current ray
    enum RayStatus status;
    // type of current ray
    enum RayType type;

    // Counter of valid child rays
    uint32_t validCount;
    // Counter of invalid rays as per nohit
    uint32_t nohitCount;
    // Counter of invisible diffuse reflection rays
    uint32_t invisibleCount;
    // Counter of invalid rays as per overflow
    uint32_t overflowCount;
    // Counter of too weak ingnored rays
    uint32_t weakCount;
};


class Light
{
public:
    Light(const Vec3f &p, const Vec3f &i) : position(p), intensity(i) {}
    Vec3f position;
    Vec3f intensity;
};

// Surface light cast to a specific angle
struct SurfaceAngle {
    Vec3f angleColor;
};

// shade point on each object
class Surface {
public:
    Surface(const float surfaceAngleRatio) : angleRatio(surfaceAngleRatio) {
        /* caculate the hit angle refer to sphere Normal on the surface */
        /* each surface will cast rays into a half sphere space which express as theta[0,90),phi[0,360) */
        vAngleRes = (90.+1.)*angleRatio;
        hAngleRes = (360.)*angleRatio;
        if (angleRatio > 0.) {
            MY_UINT64_T size = (MY_UINT64_T)sizeof(SurfaceAngle)*vAngleRes*hAngleRes;
            angles = (SurfaceAngle *)malloc(size);
            std::memset(angles, 0, size);
        }
    }
    void reset(uint32_t index, Vec3f &normal, Vec3f center) {
        idx = index;
        N = normal;
        if (angleRatio > 0.0) {
            local2World = lookAt(center, center+N);
            world2Local = local2World.inverse();
            //std::cout << center << normal << std::endl << local2World << std::endl;
        }
        //std::cout << center << center+N << std::endl;
        MY_UINT64_T size = (MY_UINT64_T)sizeof(SurfaceAngle)*vAngleRes*hAngleRes;
        std::memset(angles, 0, size);
    }

    SurfaceAngle* getSurfaceAngleByVH(const uint32_t v, const uint32_t h, Vec3f * relPoint=nullptr) const
    {
        SurfaceAngle *angle = nullptr;
        if(angles == nullptr) return angle;
        angle = angles + v%vAngleRes*hAngleRes + h%hAngleRes;
        if (relPoint != nullptr) {
            float theta = deg2rad(v*91.0/vAngleRes);
            float phi = deg2rad(h*360.0/hAngleRes);
            assert (theta>=0. && theta<=90.);
            assert (phi>=0. && phi<360.);
            // modified by xinhou @20200205
            //*relPoint = Vec3f(sin(phi)*sin(theta), cos(theta), cos(phi)*sin(theta));
            //Vec3f relPointN = Vec3f(sin(phi)*sin(theta), cos(theta), cos(phi)*sin(theta));
            Vec3f relPointN = Vec3f(cos(phi)*sin(theta), cos(theta), sin(phi)*sin(theta));
            local2World.multDirMatrix(relPointN, *relPoint);
/*
            if (relPointN != *relPoint)
                std::cout << relPointN << *relPoint << std::endl << local2World << std::endl;
*/
        }
        return angle;
    }    

    SurfaceAngle* getSurfaceAngleByPolar(const float theta, const float phi) const
    {
        if(angles == nullptr) return nullptr;
        /* caculate the hit angle refer to sphere Normal on the surface */
        /* each surface will cast rays into a half sphere space which express as theta[0,90),phi[0,360) */
        uint32_t v,h;
        v = floor(theta/91.0*vAngleRes);
        h = floor(phi/360.0*hAngleRes);
        return angles + v*hAngleRes + h;
    }

    SurfaceAngle* getSurfaceAngleByDir(const Vec3f &dir, uint32_t *angleV = nullptr, uint32_t *angleH = nullptr) const
    {
        Vec3f dirWorld;
        if(angles == nullptr) return nullptr;
        world2Local.multDirMatrix(dir, dirWorld);
        float theta = rad2deg(acos(dirWorld.y));
        //float phi = rad2deg(atan2(dirWorld.x, dirWorld.z));
        float phi = rad2deg(atan2(dirWorld.z, dirWorld.x));
#if 0
        /* caculate the hit angle refer to sphere Normal on the surface */
        /* each surface will cast rays into a half sphere space which express as theta[0,90),phi[0,360) */
        float thetaD = rad2deg(acos(-dir.y));
        float phiD = rad2deg(atan2(-dir.x, -dir.z));
        float thetaN = rad2deg(acos(N.y));
        float phiN = rad2deg(atan2(N.x, N.z));
        float theta = fabs(thetaD - thetaN);
        float phi = 90.+(phiD - phiN);
        if (phi < 0) phi += 360.;
#endif
        /* caculate the hit angle refer to sphere Normal on the surface */
        /* each surface will cast rays into a half sphere space which express as theta[0,90),phi[0,360) */
        uint32_t v,h;
        v = floor(theta/91.0*vAngleRes);
        h = floor(phi/360.0*hAngleRes);
        if (angleV != nullptr) *angleV = v;
        if (angleH != nullptr) *angleH = h;
        return angles + v*hAngleRes + h;
    }

    // there will be 90*angleRatio*360*angleRatio angles to cast rays
    float angleRatio = 0.0;
    uint32_t vAngleRes, hAngleRes;

    // hitColor = diffuseColor*diffuseAmt + specularColor * specularAmt;
    // diffuseAmt = SUM(diffuseAmt from each light)
    Vec3f diffuseAmt;
    // diffuseColor = hitObject->evalDiffuseColor(st)
    //Vec3f diffuseColor;
    // specularAmt = SUM(specularAmt from each light)
    // each light = powf(std::max(0.f, -dotProduct(reflectionDirection, dir)), hitObject->specularExponent) * lights[i]->intensity;
    //Vec3f specularAmt;
    // now set it to hitObject->Ks;
    //Vec3f specularColor;
    Vec3f N; // normal
    // Change local matrix system to world matrix system
    Matrix44f local2World;
    Matrix44f world2Local;
    /* TBD: index of current surface inside object, it can be caculated instead of using memory */
    uint32_t   idx;
    // store relfect and refract color to each angles
    struct SurfaceAngle *angles = nullptr;
};


class Object
{
 public:
    Object() :
        type(OBJECT_TYPE_NONE),
        name("NO NAME"),
        materialType(DIFFUSE_AND_GLOSSY),
        // LEO: TBD
        ior(1.3), Kd(0.1), Ks(0.2), diffuseColor(0.2), specularExponent(25),
        //ior(1.3), Kd(0.4), Ks(0.2), diffuseColor(0.2), specularExponent(25),
        vRes(0), hRes(0) {}
    virtual ~Object() {}
    virtual bool intersect(const Vec3f &, const Vec3f &, float &, Vec3f &, Vec2f &, Surface **, SurfaceAngle **) const = 0;
    virtual Surface* getSurfaceByVH(const uint32_t &, const uint32_t &, Vec3f * =nullptr) const = 0;
    virtual Vec3f evalDiffuseColor(const Vec2f &) const { return diffuseColor; }
    virtual Vec3f pointRel2Abs(const Vec3f &) const =0;
    virtual Vec3f pointAbs2Rel(const Vec3f &) const =0;
    virtual void reset(void) {};
    void enableRecorder(void)
    {
        if (traceLinks == nullptr) {
            // creating the ray links to record the rays
            traceLinks = (std::vector<std::unique_ptr<Ray>> *)malloc((MY_UINT64_T)sizeof(std::vector<std::unique_ptr<Ray>>)*vRes*hRes);
            assert(traceLinks != nullptr);
            std::memset(traceLinks, 0, sizeof(std::vector<std::unique_ptr<Ray>>)*vRes*hRes);
        }
        if (traceLinks != nullptr)
            recorderEnabled = true;
        else {
            recorderEnabled = false;
            std::printf("failed to enable rays recorder for object:%s\n", name.c_str());
        }
    }
    void disableRecorder(void) { recorderEnabled = false; }

    void setType(ObjectType objType) { type = objType; }
    void setName(std::string objName)
    {
        name = objName;
    }
    std::string getName(void) {return name;}
    void setResolution(uint32_t verticalRes, uint32_t horizonRes)
    {
        vRes  = verticalRes;
        hRes  = horizonRes;
    }
    void dumpSurface(const Options &option) const
    {
        char outfile[256];
        std::sprintf(outfile,
            "obj[%s]_density.%.2f_dep.%d_spp.%d_split.%d.ppm", name.c_str(), RAY_CAST_DESITY, option.maxDepth, option.spp,
            option.diffuseSpliter);
        // save framebuffer to file
        std::ofstream ofs;
        /* text file for compare */
        ofs.open(outfile);
        ofs << "P3\n" << hRes << " " << vRes << "\n255\n";
        Surface *curr;
        for (uint32_t v = 0; v < vRes; ++v) {
            for (uint32_t h = 0; h < hRes; ++h) {
                curr = getSurfaceByVH(v, h);
                int r = (int)(255 * clamp(0, 1, curr->diffuseAmt.x));
                int g = (int)(255 * clamp(0, 1, curr->diffuseAmt.y));
                int b = (int)(255 * clamp(0, 1, curr->diffuseAmt.z));
                ofs << r << " " << g << " " << b << "\n ";
            }
        }
        ofs.close();
    }

    void dumpSurfaceAngles(const Options &option) const
    {
        char outfile[256];
        std::sprintf(outfile,
            "objangle[%s]_density.%.2f_dep.%d_spp.%d_split.%d.ppm", name.c_str(), RAY_CAST_DESITY, option.maxDepth, option.spp,
            option.diffuseSpliter);
        // save framebuffer to file
        std::ofstream ofs;
        /* text file for compare */
        ofs.open(outfile);
        ofs << "P3\n" << hRes << " " << vRes << "\n255\n";
        Surface *curr;
/*
        for (uint32_t v = 0; v < vRes; ++v) {
            for (uint32_t h = 0; h < hRes; ++h) {
                curr = getSurfaceByVH(v, h);
*/
                curr = getSurfaceByVH(0, 0);
                for (uint32_t vAngle=0; vAngle<curr->vAngleRes; vAngle++) {
                    for (uint32_t hAngle=0; hAngle<curr->hAngleRes; hAngle++) {
                        SurfaceAngle *angle = curr->getSurfaceAngleByVH(vAngle, hAngle);
                        int r = (int)(255 * clamp(0, 1, angle->angleColor.x));
                        int g = (int)(255 * clamp(0, 1, angle->angleColor.y));
                        int b = (int)(255 * clamp(0, 1, angle->angleColor.z));
                        ofs << r << " " << g << " " << b << "\n ";
                    }
                }
/*
            }
        }
*/
        ofs.close();
    }


    ObjectType type;
    // material properties
    MaterialType materialType;
    float ior;
    float Kd, Ks;
    Vec3f diffuseColor;
    float specularExponent;
    std::string  name;
    // vertical resolution is the factor to split x from [min, max] or THETA from [0, 180]
    // we set the vertical resolution as r/10 by now.
    uint32_t vRes;
    // horizontal resolution is the factor to split y from [min, max] or PHI from [0, 360)
    // we set the horizontal resolution as r/10 by now.
    uint32_t hRes;
    // ratio determine the object and light field datas
    float ratio = RAY_CAST_DESITY;

    // link stack to record the rays
    std::vector<std::unique_ptr<Ray>> * traceLinks = nullptr;
    bool recorderEnabled = false;
    // there will be vRes*ampRatio*hRes*ampRatio blocks of amp value
    float ampRatio = 1.0;
    // there will be vAngleRes*ampRatio*hAngleRes*ampRatio blocks of amp value
    float surfaceAngleRatio = 0.0;
    // the number point is vRes * hRes
    //struct Surface * pSurfaces;
    std::vector<std::unique_ptr<Surface>> pSurfaces; 
    // the diffuse color the object by itself
    Vec3f  localDiffuseColor = -1.;
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

class RayStore
{
public:
    RayStore(const Options &currOption) : option(currOption) 
    {
        currPixel = 0;
        currRay = nullptr;

//#define RAY_TRACE_LINK_RECORDER
#ifdef RAY_TRACE_LINK_RECORDER
        // creating the rays from eye tracker
        eyeTraceLinks = (std::vector<std::unique_ptr<Ray>> *)malloc((MY_UINT64_T)sizeof(std::vector<std::unique_ptr<Ray>>)*VIEW_HEIGHT*VIEW_WIDTH);
        assert(eyeTraceLinks != nullptr);
        std::memset(eyeTraceLinks, 0, sizeof(std::vector<std::unique_ptr<Ray>>)*VIEW_HEIGHT*VIEW_WIDTH);
#endif
        totalMem = 0;
        totalRays = 0;
        originRays = 0;
        reflectionRays = 0;
        refractionRays = 0;
        diffuseRays = 0;
        invisibleRays = 0;
        weakRays = 0;
        overflowRays = 0;
        loopInternalRays = 0;
        validRays = 0;
        invalidRays = 0;
        nohitRays = 0;
    }
    Ray * record(const RayType type, std::vector<std::unique_ptr<Ray>> *links, const uint32_t index, 
                    const Vec3f &orig, const Vec3f &dir, const Vec3f &intensity = -1)
    {
        if (links == nullptr || links+index == nullptr)
            return nullptr;
        // tracker the ray
        Ray * currRay = new Ray(type, orig, dir);
        totalMem += sizeof(Ray);
        currRay = currRay;
        links[index].push_back(std::unique_ptr<Ray>(currRay));
        return currRay;
    }
    void dumpRay(const std::vector<std::unique_ptr<Ray>> &curr, uint32_t idx, uint32_t depth)
    {
        #define MAX_DUMP_DEPTH 256 
        char prefix[MAX_DUMP_DEPTH*2] = "";
        if(depth >= MAX_DUMP_DEPTH) {
            std::printf("dumpRay out of stack\n");
            return;
        }

        std::printf("%*s%d from(%f,%f,%f)-%d->to(%f,%f,%f), intensity(%f,%f,%f)*\n", depth, "#", depth,
                    curr[idx]->orig.x, curr[idx]->orig.y, curr[idx]->orig.z, 
                    curr[idx]->inside,
                    curr[idx]->dir.x, curr[idx]->dir.y, curr[idx]->dir.z,
                    curr[idx]->intensity.x, curr[idx]->intensity.y, curr[idx]->intensity.z);
        if(curr[idx]->hitObject != nullptr) {
            std::printf("%*s%d hit object: %p, point(%f, %f, %f)\n", depth, "#", depth, 
                    ((Object *)(curr[idx]->hitObject))->name.c_str(), 
                    curr[idx]->hitPoint.x, curr[idx]->hitPoint.y, curr[idx]->hitPoint.z);

            for(uint32_t i=0; i<curr[idx]->reflectionLink.size(); i++) {
                std::printf("%*s%d reflect[%d]:\n", depth+1, "#", depth+1, i);
                dumpRay(curr[idx]->reflectionLink, i, depth+1);
            }

            for(uint32_t i=0; i<curr[idx]->refractionLink.size(); i++) {
                std::printf("%*s%d refract[%d]:\n", depth+1, "#", depth+1, i);
                dumpRay(curr[idx]->refractionLink, i, depth+1);
            }

            for(uint32_t i=0; i<curr[idx]->diffuseLink.size(); i++) {
                std::printf("%*s%d diffuse[%d]:\n", depth+1, "#", depth+1, i);
                dumpRay(curr[idx]->diffuseLink, i, depth+1);
            }
        }
        else
            std::printf("%*s%d nohit\n", depth, "#", depth);
    }

    void dumpObjectTraceLink(
        const std::vector<std::unique_ptr<Object>> &objects,
        uint32_t objIdx,
        uint32_t vertical,
        uint32_t horizon)
    {
        if (objects[objIdx]->traceLinks == nullptr) return;
        std::printf("***dump light trace rays of object-vertical-horizon(%s,%d,%d)***\n", objects[objIdx]->name.c_str(), vertical, horizon);
        uint32_t index = vertical*objects[objIdx]->hRes + horizon;
        dumpRay(objects[objIdx]->traceLinks[index], 0, 1);
    }

    void dumpEyeTraceLink(
        uint32_t vertical,
        uint32_t horizon) 
    {
        if (eyeTraceLinks == nullptr) return;
        std::printf("***********dump eye trace rays of pixel(%d,%d)**************\n", vertical, horizon);
        assert( vertical < VIEW_HEIGHT && horizon < VIEW_WIDTH );
        uint32_t index = vertical*VIEW_WIDTH + horizon;
        dumpRay(eyeTraceLinks[index], 0, 1);
    }

    Options option;
    // Pixel point of current processing original ray
    Vec3f currPixel;
    Ray *currRay;

    std::vector<std::unique_ptr<Ray>> *eyeTraceLinks = nullptr;

    // Counter of total memory to record rays
    MY_UINT64_T totalMem;
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
    // Counter of ignored weak rays
    uint32_t weakRays;
    // Counter of invalid rays as per overflow
    uint32_t overflowRays;
    // Rays loop inside object
    uint32_t loopInternalRays;
    // Counter of valid rays
    uint32_t validRays;
    // Counter of invalid rays
    uint32_t invalidRays;
    // Counter of invalid rays as per nohit
    uint32_t nohitRays;
};

class Sphere : public Object
{
public:
    Sphere(const std::string name, const MaterialType type, const Vec3f &c, const float &r) : center(c), radius(r), radius2(r * r) 
    {
        materialType = type;
        switch (materialType) {
            case DIFFUSE_AND_GLOSSY:
                ampRatio = ratio;
                surfaceAngleRatio = 0.0;
                break;
            default:
                ampRatio = 4.*ratio;
                surfaceAngleRatio = ratio;
                break;
        }
        // vertical range is [0,180], horizon range is [0,360)
        uint32_t vRes = (180.+1.)*ampRatio*r, hRes = 360.*ampRatio*r;
        setType(OBJECT_TYPE_SPHERE);
        setName(name);
        setResolution(vRes, hRes);
        //MY_UINT64_T size = (MY_UINT64_T)sizeof(Surface) * vRes * hRes;
        //pSurfaces = (Surface *)malloc(size);
        uint32_t vAngleRes = 0;
        uint32_t hAngleRes = 0;
        for (uint32_t i=0; i<vRes*hRes; i++) {
            Surface *surface = new Surface(surfaceAngleRatio);
            pSurfaces.push_back(std::unique_ptr<Surface>(surface));
            if (vAngleRes < surface->vAngleRes) vAngleRes = surface->vAngleRes;
            if (hAngleRes < surface->hAngleRes) hAngleRes = surface->hAngleRes;
        }
        uint64_t raysNum = vRes*hRes + vRes*hRes*vAngleRes*hAngleRes;
        std::printf("sphere:%s, shadePoint:%d (vRes:%d, hRes:%d), pointAngle:%d (vAngle:%d, hAngle:%d), rays:%lu\n", 
                    name.c_str(), vRes*hRes, vRes, hRes, vAngleRes*hAngleRes, vAngleRes, hAngleRes, raysNum);
        reset();
    }
    // store the pre-caculated shade value of each point
    void reset(void)
    {
        float theta, phi;
        Surface *curr;
        // DEBUG
        uint32_t idx = 0;
        for (uint32_t v = 0; v < vRes; ++v) {
            for (uint32_t h = 0; h < hRes; ++h) {
                curr = getSurfaceByVH(v, h);
                // v(0, 1, 2, ... , 17) ==> theta(0, 10, 20, ..., 180)
                // TBD: bugfix when v is 180, theta is only 180/181*180
                theta = deg2rad(181.f * v/vRes);
                phi = deg2rad(360.f * h/hRes);
                //Vec3f normal = Vec3f(sin(phi)*sin(theta), cos(theta), cos(phi)*sin(theta));
                Vec3f normal = Vec3f(cos(phi)*sin(theta), cos(theta), sin(phi)*sin(theta));
                Vec3f surfaceCenter = center+normal*radius;
                curr->reset(idx++, normal, surfaceCenter);
/*
                if (v >= 360 )
                    std::printf("&&&&v=%d, vRes=%d, theta=%f, N.y=%f\n",v, vRes, theta, curr->N.y);
*/
            }
        }
    }
    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear, Vec3f &point, Vec2f &mapIdx, Surface **surface, SurfaceAngle **angle) const
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

        point = orig + dir * tnear;
        Vec3f N = normalize(point - center);
        /* caculate the hit point refer to sphere center on the surface */
        float theta = rad2deg(acos(N.y));
        //float phi = rad2deg(atan2(N.x, N.z));
        float phi = rad2deg(atan2(N.z, N.x));
        /* set the bitmap index */
        mapIdx.x = theta;
        mapIdx.y = phi;
        uint32_t v,h;
        v = floor(theta/181.0*vRes);
        h = floor(phi/360.0*hRes);
        Surface *pSurface = getSurfaceByVH(v, h);
        assert(surface != nullptr);
        *surface = pSurface;
        /* caculate the hit angle refer to sphere center on the surface */
        assert(angle != nullptr);
        if (pSurface != nullptr)
            *angle = pSurface->getSurfaceAngleByDir(-dir);
        else
            *angle = nullptr;

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

    Surface* getSurfaceByVH(const uint32_t &v, const uint32_t &h, Vec3f *worldPoint = nullptr) const
    {
        //assert( v < vRes && h < hRes);
        Surface *surface;
        surface = pSurfaces.at(v%vRes*hRes + h%hRes).get();
        if (surface != nullptr && worldPoint != nullptr)
            *worldPoint = center + surface->N*radius;
        return surface;
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
    if (fabs(det) < 0.000001) return false;

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

    return tnear < 0? false: true;
}

class MeshTriangle : public Object
{
public:
    MeshTriangle(
        const std::string name, 
        const MaterialType type,
        const Vec3f *verts,
        const uint32_t *vertsIndex,
        const uint32_t &numTris,
        const Vec2f *st)
    {
        materialType = type;
        switch (materialType) {
            case DIFFUSE_AND_GLOSSY:
                ampRatio = ratio;
                surfaceAngleRatio = 0.0;
                break;
            default:
                ampRatio = 10.*ratio;
                surfaceAngleRatio = 5.*ratio;
                break;
        }
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
        
        const Vec3f &v0 = vertices[vertexIndex[0]];
        const Vec3f &v1 = vertices[vertexIndex[1]];
        const Vec3f &v2 = vertices[vertexIndex[2]];
        Vec3f e0 = (v1 - v0);
        Vec3f e1 = (v2 - v0);

        uint32_t vRes = (uint32_t)(ampRatio * dotProduct(e0, e0));
        uint32_t hRes = (uint32_t)(ampRatio * dotProduct(e1, e1));
        setType(OBJECT_TYPE_MESH);
        setName(name);
        setResolution(vRes, hRes);
        //MY_UINT64_T size = (MY_UINT64_T)sizeof(Surface) * vRes * hRes;
        //pSurfaces = (Surface *)malloc(size);
        uint32_t vAngleRes = 0;
        uint32_t hAngleRes = 0;
        for (uint32_t i=0; i<vRes*hRes; i++) {
            Surface *surface = new Surface(surfaceAngleRatio);
            pSurfaces.push_back(std::unique_ptr<Surface>(surface));
            if (vAngleRes < surface->vAngleRes) vAngleRes = surface->vAngleRes;
            if (hAngleRes < surface->hAngleRes) hAngleRes = surface->hAngleRes;
        }
        reset();

        uint64_t raysNum = vRes*hRes + vRes*hRes*vAngleRes*hAngleRes;
        std::printf("mesh:%s, shadePoint:%d (vRes:%d, hRes:%d), pointAngle:%d (vAngle:%d, hAngle:%d), rays:%lu\n", 
                    name.c_str(), vRes*hRes, vRes, hRes, vAngleRes*hAngleRes, vAngleRes, hAngleRes, raysNum);
    }

    Surface* getSurfaceByVH(const uint32_t &v, const uint32_t &h, Vec3f *worldPoint = nullptr) const
    {
        //assert( v < vRes && h < hRes);
        Surface *surface;
        surface = pSurfaces.at(v%vRes*hRes + h%hRes).get();
        if (surface != nullptr && worldPoint != nullptr) {
            const Vec3f &v0 = vertices[vertexIndex[0]];
            const Vec3f &v1 = vertices[vertexIndex[1]];
            const Vec3f &v2 = vertices[vertexIndex[2]];
            Vec3f e0 = (v1 - v0) * (1.0/hRes);
            Vec3f e1 = (v2 - v0) * (1.0/vRes);
            *worldPoint = v0 + h*e0 + v*e1;
            //std::printf("h/v(%d,%d), P(%f,%f,%f)\n", h, v, worldPoint->x, worldPoint->y, worldPoint->z);
        }
        return surface;
    }

    void reset(void)
    {
        uint32_t size = sizeof(Surface) * vRes * hRes;
        float theta, phi;
        Surface *curr;

        const Vec3f &v0 = vertices[vertexIndex[0]];
        const Vec3f &v1 = vertices[vertexIndex[1]];
        const Vec3f &v2 = vertices[vertexIndex[2]];
        Vec3f e0 = normalize(v1 - v0);
        Vec3f e1 = normalize(v2 - v0);
        Vec3f N = normalize(crossProduct(e0, e1));

        uint32_t idx = 0;
        Vec3f center;
        for (uint32_t v = 0; v < vRes; ++v) {
            for (uint32_t h = 0; h < hRes; ++h) {
                curr = getSurfaceByVH(v, h, &center);
                //TBD, LEO, center of a mesh is not v1 need to be justified, v1-(v2+v1)/2 ?
                curr->reset(idx++, N, center);
            }
        }
    }

    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear, Vec3f &point, Vec2f &mapIdx, Surface **surface, SurfaceAngle **angle) const
    {
        bool intersect = false;
        uint32_t index = 0;
        float u = 0, v = 0;
        for (uint32_t k = 0; k < numTriangles; ++k) {
            const Vec3f & v0 = vertices[vertexIndex[k * 3]];
            const Vec3f & v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vec3f & v2 = vertices[vertexIndex[k * 3 + 2]];
            float tK, uK, vK;
            if (rayTriangleIntersect(v0, v1, v2, orig, dir, tK, uK, vK) && tK < tnear) {
                tnear = tK;
                index = k;
                u = uK;
                v = vK;
                intersect = true;
            }
        }

        if ( intersect ) {
            const Vec2f &st0 = stCoordinates[vertexIndex[index * 3]];
            const Vec2f &st1 = stCoordinates[vertexIndex[index * 3 + 1]];
            const Vec2f &st2 = stCoordinates[vertexIndex[index * 3 + 2]];
            Vec2f st = st0 * (1 - u - v) + st1 * u + st2 * v;
            point = orig + dir * tnear;
            assert ( 0 <= st.x <= 1.0 && 0 <= st.y <= 1.0);
            uint32_t v = round(st.y*vRes);
            uint32_t h = round(st.x*hRes);
            if (v >= vRes) v = vRes - 1;
            if (h >= hRes) h = hRes - 1;
            Surface *pSurface = getSurfaceByVH(v, h);
            assert(surface != nullptr);
            *surface = pSurface;
            /* set the bitmap index */
            mapIdx = st;
            /* caculate the hit angle refer to sphere center on the surface */
            assert(angle != nullptr);
            if (pSurface != nullptr)
                *angle = pSurface->getSurfaceAngleByDir(-dir);
            else
                *angle = nullptr;
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

    Vec3f evalDiffuseColor(const Vec2f &mapIdx) const
    {
        if (localDiffuseColor == -1.) {
            //float pattern = (fmodf(st.x * mapRatio, 1) > 0.5) ^ (fmodf(st.y * mapRatio, 1) > 0.5);
            float pattern = (fmodf(mapIdx.x * mapRatio, 1) > 0.5) ^ (fmodf(mapIdx.y * mapRatio, 1) > 0.5);
            return mix(Vec3f(0.815, 0.235, 0.031), Vec3f(0.937, 0.937, 0.231), pattern);
        }
        return localDiffuseColor;
    }

    std::unique_ptr<Vec3f[]> vertices;
    uint32_t numTriangles;
    std::unique_ptr<uint32_t[]> vertexIndex;
    std::unique_ptr<Vec2f[]> stCoordinates;
    /* there will be mapRatio*mapRatio*4 blocks of different color */
    uint32_t mapRatio = 5;
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
    float &tNear, Vec3f &hitPoint, Vec2f &mapIdx, Surface **hitSurface, SurfaceAngle **hitAngle, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        Vec3f pointK = 0;
        Vec2f mapK = 0;
        Surface *surfaceK = nullptr;
        SurfaceAngle *angleK = nullptr;
        if (objects[k]->intersect(orig, dir, tNearK, pointK, mapK, &surfaceK, &angleK) && tNearK < tNear) {
            *hitObject = objects[k].get();
            tNear = tNearK;
            *hitSurface = surfaceK;
            *hitAngle = angleK;
            hitPoint = pointK;
            mapIdx = mapK;
        }
    }
    
    bool hitted = (*hitObject != nullptr);
    return hitted;

//    return (*hitObject != nullptr);
}

/* precast ray from light to object*/
Vec3f forwordCastRay(
    RayStore &rayStore,
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    const Vec3f intensity,
    const Options &options,
    uint32_t depth,
    // index of object which is direct casted
    Object *targetObject=nullptr,
    Surface *targetSurface=nullptr,
    Vec3f targetPoint = 0,
    Vec2f targetMapIdx = 0)
{
    Ray * newRay = nullptr;
    Ray * currRay = nullptr;

    if (depth > OVERSTACK_PROTECT_DEPTH) {
        rayStore.overflowRays++;
        if(rayStore.currRay != nullptr) {
            rayStore.currRay->status = OVERFLOW_RAY;
            rayStore.currRay->overflowCount++;
        }
        return options.backgroundColor;
    }

    rayStore.totalRays++;
    
    Vec3f hitColor = options.backgroundColor;
    Vec3f leftIntensity = -1;
    float tnear = kInfinity;
    Object *hitObject = nullptr;
    Surface *hitSurface = nullptr;
    SurfaceAngle *hitAngle = nullptr;
    Vec3f hitPoint = 0;
    Vec2f mapIdx = 0;
    bool hitted = trace(orig, dir, objects, tnear, hitPoint, mapIdx, &hitSurface, &hitAngle, &hitObject);
    bool insideObject = false;
/*
    if(hitted && depth >=1)
        std::printf("this should not happen.depth=%d, orig(%f,%f,%f)-->dir(%f,%f,%f), hitObject(%s)\n", 
                    depth, orig.x, orig.y, orig.z, dir.x, dir.y, dir.z, hitObject->name.c_str());
    if (hitted && depth >= 3)
        std::printf("o(%f,%f), type(%d), depth:%d, tnear: %f, hitObject:%s, intensity:%f\n",
                    rayStore.currPixel.x, rayStore.currPixel.y, rayStore.currType,
                    depth, tnear, hitObject==nullptr?"nothing":hitObject->name.c_str(), intensity.x);
*/
    if (targetObject != nullptr) {
/*
        if (hitted && targetObject->name == "mesh1" && hitObject->name == "mesh1") {
            std::printf("o(%f,%f), point(%f,%f,%f), tnear: %f\n", rayStore.currPixel.x, rayStore.currPixel.y, 
                        hitPoint.x, hitPoint.y, hitPoint.z, tnear);
        }
*/
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
    }
    if (hitSurface == nullptr) {
        std::printf("BUG: ####targetSurface should not be NULL here\n");
        return hitColor;
    }
    Vec3f N = hitSurface->N; // normal

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


    if(rayStore.currRay != nullptr) {
        rayStore.currRay->status = VALID_RAY;
        rayStore.currRay->validCount++;
        rayStore.currRay->hitObject = hitObject;
        rayStore.currRay->hitPoint = hitPoint;
    }

    switch (hitObject->materialType) {
        case REFLECTION_AND_REFRACTION:
        {
            float kr;
            fresnel(dir, N, hitObject->ior, kr);
            leftIntensity = intensity*kr;
            float leftSqureValue = dotProduct(leftIntensity, leftIntensity);
            if (leftSqureValue < INTENSITY_TOO_WEAK) {
                rayStore.weakRays ++;
                if(rayStore.currRay != nullptr)
                    rayStore.currRay->weakCount ++;
                break;
            }
            Vec3f reflectionDirection = normalize(reflect(dir, N));
            insideObject = (dotProduct(reflectionDirection, N) < 0);
            Vec3f reflectionRayOrig = insideObject ?
                hitPoint - N * options.bias :
                hitPoint + N * options.bias;
            Vec3f reflectionColor = 0;
/*
            Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                hitPoint - N * options.bias :
                hitPoint + N * options.bias;
            Vec3f refractionRayOrig = (dotProduct(refractionDirection, N) < 0) ?
                hitPoint - N * options.bias :
                hitPoint + N * options.bias;
*/
            /* don't trace reflection ray inside object */
            if(!insideObject) {
                rayStore.reflectionRays++;
                // tracker the ray
                if(rayStore.currRay != nullptr) {
                    newRay = rayStore.record(RAY_TYPE_REFLECTION, &rayStore.currRay->reflectionLink, 0, reflectionRayOrig, reflectionDirection, leftIntensity);
                    newRay->inside = insideObject;
                    currRay = rayStore.currRay;
                    rayStore.currRay = newRay;
                }
                reflectionColor = forwordCastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, leftIntensity, options, depth + 1);
                rayStore.currRay = currRay;
                //Vec3f reflectionColor = forwordCastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, intensity*kr, options, depth + 1);
            }

            leftIntensity = intensity*(1-kr);
            Vec3f refractionDirection = normalize(refract(dir, N, hitObject->ior));
            insideObject = (dotProduct(refractionDirection, N) < 0);
            Vec3f refractionRayOrig = insideObject ?
                hitPoint - N * options.bias :
                hitPoint + N * options.bias;
            rayStore.refractionRays++;
            // tracker the ray
            if(rayStore.currRay != nullptr) {
                newRay = rayStore.record(RAY_TYPE_REFRACTION, &rayStore.currRay->refractionLink, 0, refractionRayOrig, refractionDirection, leftIntensity);
                newRay->inside = insideObject;
                currRay = rayStore.currRay;
                rayStore.currRay = newRay;
            }
            Vec3f refractionColor = forwordCastRay(rayStore, refractionRayOrig, refractionDirection, objects, leftIntensity, options, depth + 1);
            rayStore.currRay = currRay;
            hitColor = reflectionColor * kr + refractionColor * (1 - kr);
            break;
        }
        case REFLECTION:
        {
            float kr = 0.9;
            //fresnel(dir, N, hitObject->ior, kr);
            leftIntensity = intensity*kr;
            float leftSqureValue = dotProduct(leftIntensity, leftIntensity);
            if (leftSqureValue < INTENSITY_TOO_WEAK) {
                rayStore.weakRays ++;
                if(rayStore.currRay != nullptr)
                    rayStore.currRay->weakCount ++;
                break;
            }
            Vec3f reflectionDirection = reflect(dir, N);
            insideObject = (dotProduct(reflectionDirection, N) < 0);
            Vec3f reflectionRayOrig = insideObject ?
                hitPoint - N * options.bias :
                hitPoint + N * options.bias;
            rayStore.reflectionRays++;
            // tracker the ray
            if(rayStore.currRay != nullptr) {
                newRay = rayStore.record(RAY_TYPE_REFLECTION, &rayStore.currRay->reflectionLink, 0, reflectionRayOrig, reflectionDirection, leftIntensity);
                newRay->inside = insideObject;
                currRay = rayStore.currRay;
                rayStore.currRay = newRay;
            }
           // hitColor = forwordCastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, intensity*(1-kr), options, depth + 1) * kr;
            hitColor = forwordCastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, leftIntensity, options, depth + 1) * kr;
            rayStore.currRay = currRay;
            break;
        }
        default:
        {
            // TBD , leo
            break;
            // Diffuse relfect
            // each relfect light will share part of the light
            // how many diffuse relfect light will be traced
            uint32_t count = options.diffuseSpliter;
            // Prevent memory waste before overflow
            if (depth+1 > OVERSTACK_PROTECT_DEPTH) {
                rayStore.overflowRays += count;
                if(rayStore.currRay != nullptr)
                    rayStore.currRay->overflowCount += count;
            }
            //leftIntensity = intensity*(1.0-kr)*(1.0/(count+1));
            leftIntensity = intensity*hitObject->Kd*(1.0/(count+1));
            float leftSqureValue = dotProduct(leftIntensity, leftIntensity);
            if (leftSqureValue < INTENSITY_TOO_WEAK) {
                rayStore.weakRays ++;
                if(rayStore.currRay != nullptr)
                    rayStore.currRay->weakCount ++;
                break;
            }
            else {
                for (int32_t i=count*(-1./2); i<=count/2.; i+=1) {
                //for (uint32_t i=1; i<=count; i++) {
                    Vec3f reflectionDirection = diffuse(dir, N, i*4.0/(count+1));
                    //Vec3f reflectionDirection = normalize(reflect(dir, N));
                    //Vec3f reflectionRayOrig = hitPoint + N * options.bias;
                    insideObject = (dotProduct(reflectionDirection, N) < 0);
                    Vec3f reflectionRayOrig = insideObject ?
                        hitPoint - N * options.bias :
                        hitPoint + N * options.bias;
                    rayStore.diffuseRays++;
                    //std::printf("DEBUG--(%d, %d)-->Total rays(%d) = Origin rays(%d) + Reflection rays(%d) + Refraction rays(%d) + Diffuse rays(%d)\n", i, depth, rays.totalRays, rays.originRays, rays.reflectionRays, rays.refractionRays, rays.diffuseRays);
                    // tracker the ray
                    if(rayStore.currRay != nullptr) {
                        newRay = rayStore.record(RAY_TYPE_DIFFUSE, &rayStore.currRay->diffuseLink, 0, reflectionRayOrig, reflectionDirection, leftIntensity);
                        newRay->inside = insideObject;
                        currRay = rayStore.currRay;
                        rayStore.currRay = newRay;
                    }
                    
                    forwordCastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, leftIntensity, options, depth + 1);
                    rayStore.currRay = currRay;
                }
            }
            
            break;
        }
    }

    /* pre-caculate diffuse amt */
    float Kd = hitObject->Kd;
    Vec3f lightDir = hitPoint - orig;
    lightDir = normalize(lightDir);
    float LdotN = std::max(0.f, dotProduct(-lightDir, N));
    if (intensity.x < 0 || intensity.y <0 || intensity.z < 0 || Kd <0)
        std::printf("ERROR: intensity=(%f,%f,%f), Kd=%f\n", intensity.x, intensity.y, intensity.z, Kd);
    /* pre-caculate diffuse amt */
    hitSurface->diffuseAmt += intensity * LdotN * Kd;
    /* pre-caculate specular amt */
    /*
    Vec3f reflectionDirection = reflect(lightDir, N);
    hitSurface->specularAmt += powf(std::max(0.f, -dotProduct(reflectionDirection, dir)), hitObject->specularExponent) * intensity;
    */
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
// by calling the backwardCastRay() function recursively. When the surface is transparent, we mix
// the reflection and refraction color using the result of the fresnel equations (it computes
// the amount of reflection and refractin depending on the surface normal, incident view direction
// and surface refractive index).
//
// If the surface is duffuse/glossy we use the Phong illumation model to compute the color
// at the intersection point.
// [/comment]
Vec3f backwardCastRay(
    RayStore &rayStore,
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    const std::vector<std::unique_ptr<Light>> &lights,
    const Options &options,
    uint32_t depth,
    bool withLightRender = false,
    bool withObjectRender = false,
    Vec3f *pDeltaAmt = nullptr)
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
    float tnear = kInfinity;
    Object *hitObject = nullptr;
    Surface * hitSurface = nullptr;
    SurfaceAngle *hitAngle = nullptr;
    Vec3f hitPoint = 0;
    Vec2f mapIdx = 0;
    Vec3f globalAmt = 0, localAmt = 0, specularColor = 0;
    bool  insideObject = false;
    if (trace(orig, dir, objects, tnear, hitPoint, mapIdx, &hitSurface, &hitAngle, &hitObject)) {
        Vec3f N = hitSurface->N; // normal
//        std::printf("%*s%d hit[%s]:\n", depth+1, "#", depth+1, hitObject->name.c_str());
//        Vec3f testColor = hitObject->evalDiffuseColor(mapIdx);
//        std::printf("#hitPoint(%f, %f, %f), color(%f, %f, %f) \n", hitPoint.x, hitPoint.y, hitPoint.z, testColor.x, testColor.y, testColor.z);
        if(rayStore.currRay != nullptr) {
            rayStore.currRay->status = VALID_RAY;
            rayStore.currRay->validCount++;
            rayStore.currRay->hitObject = hitObject;
            rayStore.currRay->hitPoint = hitPoint;
        }

        if (withObjectRender) {
            if (hitAngle == nullptr) {
                globalAmt = hitSurface->diffuseAmt;
                localAmt = 0;
                hitColor = (globalAmt + localAmt) * hitObject->evalDiffuseColor(mapIdx) + specularColor * hitObject->Ks;
            }
            else
                hitColor = hitAngle->angleColor;
            return hitColor;
        }

            
/*
        if (rayStore.currPixel.x == 235. && rayStore.currPixel.y == 304.)
            std::printf("stop here.\n");
*/
        switch (hitObject->materialType) {
            case REFLECTION_AND_REFRACTION:
            {
                Vec3f reflectionDirection = normalize(reflect(dir, N));
                insideObject = (dotProduct(reflectionDirection, N) < 0);
                Vec3f reflectionRayOrig = insideObject ?
                    hitPoint - N * options.bias :
                    hitPoint + N * options.bias;
                Vec3f reflectionColor = 0;
                Vec3f diffuseColor = 0;
                float kr;
                fresnel(dir, N, hitObject->ior, kr);
                /* don't trace reflection ray inside object */
                if(!insideObject) {
                    rayStore.reflectionRays++;
                    // tracker the ray
                    if(rayStore.currRay != nullptr) {
                        newRay = rayStore.record(RAY_TYPE_REFLECTION, &rayStore.currRay->reflectionLink, 0, reflectionRayOrig, reflectionDirection);
                        newRay->inside = insideObject;
                        currRay = rayStore.currRay;
                        rayStore.currRay = newRay;
                    }
                    reflectionColor = backwardCastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1, withLightRender);
                    rayStore.currRay = currRay;
                }

                Vec3f refractionDirection = normalize(refract(dir, N, hitObject->ior));
                insideObject = (dotProduct(refractionDirection, N) < 0);
                Vec3f refractionRayOrig = insideObject ?
                    hitPoint - N * options.bias :
                    hitPoint + N * options.bias;

                rayStore.refractionRays++;
                // tracker the ray
                if(rayStore.currRay != nullptr) {
                    newRay = rayStore.record(RAY_TYPE_REFRACTION, &rayStore.currRay->refractionLink, 0, refractionRayOrig, refractionDirection);
                    newRay->inside = insideObject;
                    currRay = rayStore.currRay;
                    rayStore.currRay = newRay;
                }
                Vec3f refractionColor = backwardCastRay(rayStore, refractionRayOrig, refractionDirection, objects, lights, options, depth + 1, withLightRender);
                rayStore.currRay = currRay;
                if (withLightRender)
                    diffuseColor = hitSurface->diffuseAmt * hitObject->evalDiffuseColor(mapIdx);
                hitColor = reflectionColor * kr + refractionColor * (1 - kr) + diffuseColor;
                break;
            }
            case REFLECTION:
            {
                float kr = 0.5;
                Vec3f reflectionColor = 0;
                Vec3f diffuseColor = 0;
                //fresnel(dir, N, hitObject->ior, kr);
                // LEO: do i need normolization here.
                Vec3f reflectionDirection = reflect(dir, N);
                insideObject = (dotProduct(reflectionDirection, N) < 0);
                Vec3f reflectionRayOrig = insideObject ?
                    hitPoint - N * options.bias :
                    hitPoint + N * options.bias;
                rayStore.reflectionRays++;
                // tracker the ray
                if(rayStore.currRay != nullptr) {
                    newRay = rayStore.record(RAY_TYPE_REFLECTION, &rayStore.currRay->reflectionLink, 0, reflectionRayOrig, reflectionDirection);
                    newRay->inside = insideObject;
                    currRay = rayStore.currRay;
                    rayStore.currRay = newRay;
                }
                reflectionColor = backwardCastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1, withLightRender) * kr;
                if (withLightRender)
                    diffuseColor = hitSurface->diffuseAmt * hitObject->evalDiffuseColor(mapIdx);
                hitColor = reflectionColor + diffuseColor;
                rayStore.currRay = currRay;
                break;
            }
            default:
            {
                // [comment]
                // We use the Phong illumation model int the default case. The phong model
                // is composed of a diffuse and a specular reflection component.
                // [/comment]
                Vec3f shadowPointOrig = (dotProduct(dir, N) < 0) ?
                    hitPoint + N * options.bias :
                    hitPoint - N * options.bias;

                if (false) {
                //if (!withLightRender) {
                    // Diffuse relfect
                    // each relfect light will share part of the light
                    // how many diffuse relfect light will be traced
                    uint32_t count = options.diffuseSpliter;
                    float weight = 1./(count+1);
                    //float weight = 0.001;
                    hitColor = 0;
                    // Prevent memory waste before overflow
                    if (depth+1 > options.maxDepth) {
                        rayStore.overflowRays += count;
                        if(rayStore.currRay != nullptr)
                            rayStore.currRay->overflowCount += count;
                    }
                    else {
                        //for (uint32_t i=1; i<=count; i++) {
                        for (int32_t i=count*(-1./2); i<=count/2.; i+=1) {
                            Vec3f reflectionDirection = normalize(diffuse(dir, N, i*4.0/count));
                            insideObject = (dotProduct(reflectionDirection, N) < 0);
                            Vec3f reflectionRayOrig = insideObject ?
                                hitPoint - N * options.bias :
                                hitPoint + N * options.bias;
                            rayStore.diffuseRays++;
                            //std::printf("DEBUG--(%d, %d)-->Total rays(%d) = Origin rays(%d) + Reflection rays(%d) + Refraction rays(%d) + Diffuse rays(%d)\n", i, depth, rays.totalRays, rays.originRays, rays.reflectionRays, rays.refractionRays, rays.diffuseRays);
                            // tracker the ray
                            if(rayStore.currRay != nullptr) {
                                newRay = rayStore.record(RAY_TYPE_DIFFUSE, &rayStore.currRay->diffuseLink, 0, reflectionRayOrig, reflectionDirection);
                                newRay->inside = insideObject;
                                currRay = rayStore.currRay;
                                rayStore.currRay = newRay;
                            }
                            Vec3f deltaAmt = 0;
                            backwardCastRay(rayStore, reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1, false, false, &deltaAmt);
                            rayStore.currRay = currRay;
                            globalAmt += deltaAmt * powf(weight, depth+1);
                        }
                    }
                }

                // [comment]
                // Loop over all lights in the scene and sum their contribution up
                // We also apply the lambert cosine law here though we haven't explained yet what this means.
                // [/comment]
                Vec3f tmpAmt = 0;
                for (uint32_t i = 0; i < lights.size(); ++i) {
                    Vec3f lightDir = lights[i]->position - hitPoint;
                    // square of the distance between hitPoint and the light
                    float lightDistance2 = dotProduct(lightDir, lightDir);
                    lightDir = normalize(lightDir);
                    if (!withLightRender) {
                        float LdotN = std::max(0.f, dotProduct(lightDir, N));
                        Object *shadowHitObject = nullptr;
                        Surface *shadowHitSurface = nullptr;
                        SurfaceAngle *shadowHitAngle = nullptr;
                        Vec3f shadowHitPoint = 0;
                        Vec2f shadowMapIdx = 0;
                        float tNearShadow = kInfinity;
                        // is the point in shadow, and is the nearest occluding object closer to the object than the light itself?
                        bool inShadow = trace(shadowPointOrig, lightDir, objects, tNearShadow, shadowHitPoint, shadowMapIdx, 
                              &shadowHitSurface, &shadowHitAngle, &shadowHitObject) && tNearShadow * tNearShadow < lightDistance2;

/*
                        if (inShadow)
                            std::printf("inShadow: obj(%s), point(%f)\n", shadowHitObject->name.c_str(), tNearShadow);
*/
                        tmpAmt = (1 - inShadow) * lights[i]->intensity * LdotN * hitObject->Kd;
                        if (pDeltaAmt != nullptr)
                            *pDeltaAmt += tmpAmt;
                        else
                            localAmt += tmpAmt;
                            
                    }
                    Vec3f reflectionDirection = reflect(-lightDir, N);
                    specularColor += powf(std::max(0.f, -dotProduct(reflectionDirection, dir)), hitObject->specularExponent) * lights[i]->intensity;
                }
                if (withLightRender) {
                    globalAmt = hitSurface->diffuseAmt;
                    localAmt = 0;
                }

                if (tmpAmt == 0) rayStore.invisibleRays++;
                else rayStore.validRays++;

                hitColor = (globalAmt + localAmt) * hitObject->evalDiffuseColor(mapIdx) + specularColor * hitObject->Ks;

/*
                if(rayStore.currPixel.x < 240.)
                    std::printf("o(%f,%f), object(%s), point(%f,%f,%f), mapidx(%f,%f)\n", 
                                rayStore.currPixel.x, rayStore.currPixel.y, hitObject->name.c_str(),
                                hitPoint.x, hitPoint.y, hitPoint.z, mapIdx.x, mapIdx.y);
*/
                            
                break;
            }
        }
    }
    else {
        rayStore.nohitRays++;
        //std::printf("%*s%d nohit\n", depth+1, "#", depth+1);
        if(rayStore.currRay != nullptr) {
            rayStore.currRay->status = NOHIT_RAY;
            rayStore.currRay->nohitCount++;
        }
    }

    return hitColor;
}

/* objectRender the object from Surface angles */
void objectRender(
    RayStore &rayStore,
    const Options &options,
    const std::vector<std::unique_ptr<Object>> &objects,
    const std::vector<std::unique_ptr<Light>> &lights)
{
    Object *targetObject;
    Surface *targetSurface;
    Vec3f   target;
    Vec3f   dir = 0;
    Vec3f   orig = 0;
    uint32_t v=0, h=0;

#define DEBUG_ANGLE_ZERO

    for (uint32_t i=0; i<objects.size(); i++) {
        targetObject = objects[i].get();

        if (targetObject->surfaceAngleRatio <= 0.) continue;
        for (v=0; v<objects[i]->vRes; v++) {
            for (h=0; h<objects[i]->hRes; h++) {
                targetSurface = targetObject->getSurfaceByVH(v, h, &target);
                if (targetSurface == nullptr) continue;

                // LEO: debug a angle color
#ifdef DEBUG_ANGLE_ZERO
                Vec3f debugDir = normalize(Vec3f(0) - target);
                uint32_t vAngleTarget = 0, hAngleTarget = 0;
                targetSurface->getSurfaceAngleByDir(debugDir, &vAngleTarget, &hAngleTarget);
#endif

                for (uint32_t vAngle=0; vAngle<targetSurface->vAngleRes; vAngle++) {
                    for (uint32_t hAngle=0; hAngle<targetSurface->hAngleRes; hAngle++) {
                        SurfaceAngle *angle = targetSurface->getSurfaceAngleByVH(vAngle, hAngle, &dir);
                        if (angle == nullptr) continue;

#ifdef DEBUG_ANGLE_ZERO
                        if (abs(vAngleTarget-vAngle)<5*ceil(targetSurface->angleRatio) && \
                            abs(hAngleTarget-hAngle)<5*ceil(targetSurface->angleRatio)) {
#endif
                            // dir of forwordCastRay is relative to orig
                            rayStore.originRays++;
                            // tracker the ray
                            if (objects[i]->recorderEnabled)
                                rayStore.record(RAY_TYPE_ORIG, objects[i]->traceLinks, v*objects[i]->hRes + h, orig, dir);
                            orig = target + dir;
                            rayStore.currPixel = {(float)v, (float)h, 0};
                            angle->angleColor = backwardCastRay(rayStore, orig, -dir, objects, lights, options, 0);
                            //std::cout << angle->angleColor <<  std::endl;
#ifdef DEBUG_ANGLE_ZERO
                        }
#endif
                    }
                }
                // rayStore.dumpObjectTraceLink(objects, i, 0, 0);
                // dump object shadepoint as ppm file
                //objects[i]->dumpSurfaceAngles(options);
            }
        }
        objects[i]->dumpSurfaceAngles(options);
    }

    // LEO: debug a angle color
    //ofs.close();
}


/* lightRender the object from light */
void lightRender(
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
    uint32_t v=0, h=0;

    for (uint32_t l=0; l<lights.size(); l++) {
        orig = lights[l]->position;
        for (uint32_t i=0; i<objects.size(); i++) {
            targetObject = objects[i].get();
            for (v=0; v<objects[i]->vRes; v++) {
                for (h=0; h<objects[i]->hRes; h++) {
                targetSurface = targetObject->getSurfaceByVH(v, h, &targetPoint);
                if (targetSurface == nullptr)
                    continue;
                // dir of forwordCastRay is relative to orig
                // dir = center + P.rel(theta, phi)*radius - orig
                // set the test point a little bit far away the center of sphere. test point is rel address from orig.
                testPoint = normalize(targetPoint + targetSurface->N*options.bias - orig);
                rayStore.originRays++;
                // tracker the ray
                if (objects[i]->recorderEnabled)
                    rayStore.record(RAY_TYPE_ORIG, objects[i]->traceLinks, v*objects[i]->hRes + h, orig, testPoint);
/*
                assert( l < LIGHT_NUM_MAX && i < OBJ_NUM_MAX && v < V_RES_MAX && h < H_RES_MAX );
                rayStore.record(RAY_TYPE_ORIG, rayStore.lightTraceLinks, 
                                l*OBJ_NUM_MAX*V_RES_MAX*H_RES_MAX + i*V_RES_MAX*H_RES_MAX + v*H_RES_MAX + h,
                                orig, testPoint);
*/
    /*
                std::printf("v/h(%d,%d): targetPoint[%f,%f,%f], testPoint[%f,%f,%f]\n",
                            v, h, targetPoint.x, targetPoint.y, targetPoint.z,
                            testPoint.x,testPoint.y, testPoint.z);
    */
                rayStore.currPixel = {(float)v, (float)h, 0};
                forwordCastRay(rayStore, orig, testPoint, objects, lights[l]->intensity, options, 0, targetObject, targetSurface, targetPoint);
                //std::printf("light[%d]:%.0f%%\r",l, (h*vRes+v)*100.0/(vRes*hRes));
                }
            }
            rayStore.dumpObjectTraceLink(objects, i, 0, 0);
            // dump object shadepoint as ppm file
            objects[i]->dumpSurface(options);
        }
    }
}


// [comment]
// The main eyeRender function. This where we iterate over all pixels in the image, generate
// primary rays and cast these rays into the scene. The content of the framebuffer is
// saved to a file.
// [/comment]
void eyeRender(
    RayStore &rayStore,
    char * outfile,
    const Options &options,
    const Vec3f &viewpoint,
    const std::vector<std::unique_ptr<Object>> &objects,
    const std::vector<std::unique_ptr<Light>> &lights,
    const bool withLightRender = false,
    const bool withObjectRender = false)
{
    Vec3f orig = viewpoint;
    // change the camera to world
#ifdef CAMERATOWORLD
    Vec3f origWorld, dirWorld;
    Matrix44f cameraToWorld(orig, orig+Vec3f{0.,0.,-1.});
    std::cout << cameraToWorld << std::endl;
#endif
    Vec3f *framebuffer = new Vec3f[options.width * options.height];
    Vec3f *pix = framebuffer;
    float scale = tan(deg2rad(options.fov * 0.5));
    float imageAspectRatio = options.width / (float)options.height;
    //Vec3f orig(0);
#if 1
    for (uint32_t j = 0; j < options.height; ++j) {
        for (uint32_t i = 0; i < options.width; ++i) {
            // generate primary ray direction
            float x = (2 * (i + 0.5) / (float)options.width - 1) * imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / (float)options.height) * scale;
            Vec3f dir = normalize(Vec3f(x, y, -1));
            rayStore.originRays++;
            rayStore.currPixel = {(float)j, (float)i, -1.0};
            // tracker the ray
            rayStore.record(RAY_TYPE_ORIG, rayStore.eyeTraceLinks, j*VIEW_WIDTH+i, orig, dir);

//DEBUG by LEO to compare backward tracing and forward tracing
#else
    for (uint32_t i = 0; i < options.width; ++i) {
        for (uint32_t j = 0; j < options.height; ++j) {
            Object *obj = objects[1].get();
            uint32_t h = (uint32_t)((float)j * obj->hRes / options.height);
            uint32_t v = (uint32_t)((float)i * obj->vRes / options.width);
            Vec3f  worldTarget;
            obj->getSurfaceByVH(v, h, &worldTarget);
            Vec3f dir = normalize(worldTarget-orig);


#endif

#ifdef CAMERATOWORLD
            cameraToWorld.multVecMatrix(orig, origWorld);
            cameraToWorld.multDirMatrix(dir, dirWorld);
            dirWorld.normalize();
            *(pix++) = backwardCastRay(rayStore, origWorld, dirWorld, objects, lights, options, 0, withLightRender, withObjectRender);
#else
            *(pix++) = backwardCastRay(rayStore, orig, dir, objects, lights, options, 0, withLightRender, withObjectRender);
#endif

#if 0
            std::cout << "oooo===" << v << "," << h << "," << 0 << "," << 0 << *(pix-1) << "===" << std::endl;
            std::cout << dir << std::endl;
//            std::printf("%.0f%%\r",(j*options.width+i)*100.0/(options.width * options.height));
#endif
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
// as well as set the options for the eyeRender (image widht and height, maximum recursion
// depth, field-of-view, etc.). We then call the eyeRender function().
// [/comment]
int main(int argc, char **argv)
{
    // creating the scene (adding objects and lights)
    std::vector<std::unique_ptr<Object>> objects;
    std::vector<std::unique_ptr<Light>> lights;
    
    
    Sphere *sph1 = new Sphere("sph1", DIFFUSE_AND_GLOSSY, Vec3f(-4, 0, -8), 2);
    sph1->ior = 1.3;
    sph1->Kd  = 0.8;
    sph1->diffuseColor = Vec3f(0.6, 0.7, 0.8);
    //sph1->enableRecorder();
    objects.push_back(std::unique_ptr<Sphere>(sph1));


/*
    Sphere *sph2 = new Sphere("sph2", REFLECTION_AND_REFRACTION, Vec3f(4, 0, -8), 2);
    sph2->ior = 1.7;
    sph2->Kd  = 0.0;
    sph2->diffuseColor = Vec3f(0.6, 0.7, 0.8);
    //sph2->enableRecorder();
    objects.push_back(std::unique_ptr<Sphere>(sph2));
*/


    Vec3f verts[4] = {{-10,-2,0}, {10,-2,0}, {10,-2,-14}, {-10,-2,-14}};
    uint32_t vertIndex[6] = {0, 1, 3, 1, 2, 3};
    Vec2f st[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    MeshTriangle *mesh1 = new MeshTriangle("mesh1", REFLECTION, verts, vertIndex, 2, st);
    mesh1->ior = 1.5;
    mesh1->Kd  = 0.1;
    mesh1->localDiffuseColor = Vec3f{0.3843, 0.3569, 0.3412};
    objects.push_back(std::unique_ptr<MeshTriangle>(mesh1));

    

    Vec3f verts2[4] = {{-10,-2,-14}, {10,-2,-14}, {10,18,-14}, {-10,18,-14}};
    uint32_t vertIndex2[6] = {0, 1, 3, 1, 2, 3};
    Vec2f st2[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    MeshTriangle *mesh2 = new MeshTriangle("mesh2", DIFFUSE_AND_GLOSSY, verts2, vertIndex2, 2, st2);
    mesh2->ior = 1.3;
    mesh2->Kd  = 0.8;
    mesh2->localDiffuseColor = -1.;
    objects.push_back(std::unique_ptr<MeshTriangle>(mesh2));



    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(20, 25, 8), 1)));
//    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(1, 1, 10), 1)));

    // setting up options
    Options options[100];
    std::memset(options, 0, sizeof(options));
    // no diffuse at all
    options[0].diffuseSpliter = 3;
    options[0].maxDepth = 5;
    options[0].spp = 1;
    options[0].width = VIEW_WIDTH*options[0].spp;
    options[0].height = VIEW_HEIGHT*options[0].spp;
    options[0].fov = 60;
    //options[0].backgroundColor = Vec3f(0.235294, 0.67451, 0.843137);
    options[0].backgroundColor = Vec3f(0.95, 0.95, 0.95);
    //options[0].backgroundColor = Vec3f(0.0);
    //options[0].bias = 0.001;
    options[0].bias = 0.001;
    options[0].doTraditionalRender = true;
    options[0].doRenderAfterDiffusePreprocess = true;
    options[0].doRenderAfterDiffuseAndReflectPreprocess = true;
    options[0].viewpoints[0] = Vec3f(0, 0, 10);

/*
    options[0].viewpoints[0] = Vec3f(0, 5, 0);
    options[0].viewpoints[1] = Vec3f(-5, 0, -4);
    //options[0].viewpoints[2] = Vec3f(5, 0, 0);
    options[0].viewpoints[3] = Vec3f(2, 0, 0);
    options[0].viewpoints[4] = Vec3f(-2, 0, 0);
*/
    //Vec3f viewpoints[4] = {{0,0,0}, {0,0,1}, {1,1,1}, {6,6,1}};

#if 0
    // 10 diffuse at each first hitpoint, no further diffuse
    options[1].diffuseSpliter = 10;
    options[1].maxDepth = 1;
    options[1].spp = 1;
    options[1].width = VIEW_WIDTH*options[1].spp;
    options[1].height = VIEW_HEIGHT*options[1].spp;
    options[1].fov = 90;
    options[1].backgroundColor = Vec3f(0.0);
    options[1].bias = 0.001;

    // 10 diffuse at each first hitpoint, 1 depth further diffuse
    options[2].diffuseSpliter = 1;
    options[2].maxDepth = 1;
    options[2].spp = 1;
    options[2].width = VIEW_WIDTH*options[2].spp;
    options[2].height = VIEW_HEIGHT*options[2].spp;
    options[2].fov = 90;
    options[2].backgroundColor = Vec3f(0.0);
    options[2].bias = 0.0001;

    // 10 diffuse at each first hitpoint, 3 depth further diffuse
    options[3].diffuseSpliter = 1;
    options[3].maxDepth = 3;
    options[3].spp = 1;
    options[3].width = VIEW_WIDTH*options[3].spp;
    options[3].height = VIEW_HEIGHT*options[3].spp;
    options[3].fov = 90;
    options[3].backgroundColor = Vec3f(0.0);
    options[3].bias = 0.0001;

    // 10 diffuse at each first hitpoint, 3 depth further diffuse
    options[4].diffuseSpliter = 1;
    options[4].maxDepth = 9;
    options[4].spp = 1;
    options[4].width = VIEW_WIDTH*options[4].spp;
    options[4].height = VIEW_HEIGHT*options[4].spp;
    options[4].fov = 90;
    options[4].backgroundColor = Vec3f(0.0);
    options[4].bias = 0.0001;

    // 10 diffuse at each first hitpoint, 3 depth further diffuse
    options[5].diffuseSpliter = 100;
    options[5].maxDepth = 1;
    options[5].spp = 1;
    options[5].width = VIEW_WIDTH*options[5].spp;
    options[5].height = VIEW_HEIGHT*options[5].spp;
    options[5].fov = 90;
    options[5].backgroundColor = Vec3f(0.0);
    options[5].bias = 0.0001;

    // 10 diffuse at each first hitpoint, 3 depth further diffuse
    options[6].diffuseSpliter = 1000;
    options[6].maxDepth = 1;
    options[6].spp = 1;
    options[6].width = VIEW_WIDTH*options[6].spp;
    options[6].height = VIEW_HEIGHT*options[6].spp;
    options[6].fov = 90;
    options[6].backgroundColor = Vec3f(0.0);
    options[6].bias = 0.0001;
#endif

    // setting up ray store
    //RayStore rayStore;

    char outfile[256];
    RayStore *rayStore;

    std::printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", 
                "split", "depth", 
                "origin", "reflect", "refract", "diffuse", 
                "nohit", "invis", "weak", "overflow", "CPU(Rays)", "TIME(S)", "MEM(GB)");
    time_t start, end;
    //std::printf("split\t depth\t total\t origin\t reflect\t refract\t diffuse\t nohit\t invis\t overflow\t CPUConsumed\n");
    for (int i =0; i<sizeof(options)/sizeof(struct Options); i++){
        if(options[i].width == 0) break;

        for (uint32_t i=0; i<objects.size(); i++) {
            objects[i]->reset();
        }

        if (options[i].doRenderAfterDiffusePreprocess == true || options[i].doRenderAfterDiffuseAndReflectPreprocess == true) {
            // do lightRender
            // setting up ray store
            rayStore = new RayStore(options[i]);
            // caculate time consumed
            start = time(NULL);
            lightRender(*rayStore, options[i], objects, lights);
            end = time(NULL);
            std::printf("###pre render for doRenderAfterDiffusePreprocess & doRenderAfterDiffuseAndReflectPreprocess from light###\n");
            std::printf("%-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10.0f %-10.2f\n",
                        options[i].diffuseSpliter, options[i].maxDepth,
                        rayStore->originRays, rayStore->reflectionRays, rayStore->refractionRays, rayStore->diffuseRays,
                        rayStore->nohitRays, rayStore->invisibleRays, rayStore->weakRays, rayStore->overflowRays, 
                        rayStore->totalRays, difftime(end, start), rayStore->totalMem*1.0/(1024.0*1024.0*1024.0));
            delete rayStore;
        }
        if (options[i].doRenderAfterDiffusePreprocess == true) {
            // do post render from eyes after lightRender
            // calcule all the viewpoints with same options 
            std::printf("###post render for doRenderAfterDiffusePreprocess###\n");
            for (int j =0; j<sizeof(options[i].viewpoints)/sizeof(Vec3f); j++) {
                rayStore = new RayStore(options[i]);
                //std::memset(&rayStore, 0, sizeof(rayStore));
                std::sprintf(outfile,
                    "afterDiffusePreprocess_x.%d_y.%d_z.%d_density.%.2f_dep.%d_spp.%d_split.%d.ppm", (int)options[i].viewpoints[j].x,
                    (int)options[i].viewpoints[j].y, (int)options[i].viewpoints[j].z, RAY_CAST_DESITY, options[i].maxDepth, options[i].spp,
                    options[i].diffuseSpliter);
                // caculate time consumed
                start = time(NULL);
                // finally, eyeRender
                //std::memset(&rayStore, 0, sizeof(rayStore));
                /* start eyeRender after lightRender */
                eyeRender(*rayStore, outfile, options[i], options[i].viewpoints[j], objects, lights, true, false);
                end = time(NULL);
                std::printf("%-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10.0f %-10.2f\n",
                            options[i].diffuseSpliter, options[i].maxDepth,
                            rayStore->originRays, rayStore->reflectionRays, rayStore->refractionRays, rayStore->diffuseRays,
                            rayStore->nohitRays, rayStore->invisibleRays, rayStore->weakRays, rayStore->overflowRays, 
                            rayStore->totalRays, difftime(end, start), rayStore->totalMem*1.0/(1024.0*1024.0*1024.0));

                delete rayStore;
                // (0,0,0) is the default viewpoint, and it means the end of the list
                if (options[i].viewpoints[j] == 0)
                    break;
            }
        }

        if (options[i].doRenderAfterDiffuseAndReflectPreprocess == true) {
            // do objectRender
            // setting up ray store
            rayStore = new RayStore(options[i]);
            // caculate time consumed
            std::printf("###pre render for doRenderAfterDiffuseAndReflectPreprocess from object surface angle###\n");
            start = time(NULL);
            objectRender(*rayStore, options[i], objects, lights);
            end = time(NULL);
            std::printf("%-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10.0f %-10.2f\n",
                        options[i].diffuseSpliter, options[i].maxDepth,
                        rayStore->originRays, rayStore->reflectionRays, rayStore->refractionRays, rayStore->diffuseRays,
                        rayStore->nohitRays, rayStore->invisibleRays, rayStore->weakRays, rayStore->overflowRays, 
                        rayStore->totalRays, difftime(end, start), rayStore->totalMem*1.0/(1024.0*1024.0*1024.0));
            delete rayStore;

            // do post render from eyes after lightRender
            // calcule all the viewpoints with same options 
            std::printf("###post render for doRenderAfterDiffuseAndReflectPreprocess###\n");
            for (int j =0; j<sizeof(options[i].viewpoints)/sizeof(Vec3f); j++) {
                rayStore = new RayStore(options[i]);
                //std::memset(&rayStore, 0, sizeof(rayStore));
                std::sprintf(outfile,
                    "afterReflectPreprocess_x.%d_y.%d_z.%d_density.%.2f_dep.%d_spp.%d_split.%d.ppm", (int)options[i].viewpoints[j].x,
                    (int)options[i].viewpoints[j].y, (int)options[i].viewpoints[j].z, RAY_CAST_DESITY, options[i].maxDepth, options[i].spp,
                    options[i].diffuseSpliter);
                // caculate time consumed
                start = time(NULL);
                // finally, eyeRender
                //std::memset(&rayStore, 0, sizeof(rayStore));
                /* start eyeRender after lightRender */
                eyeRender(*rayStore, outfile, options[i], options[i].viewpoints[j], objects, lights, true, true);
                end = time(NULL);
                std::printf("%-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10.0f %-10.2f\n",
                            options[i].diffuseSpliter, options[i].maxDepth,
                            rayStore->originRays, rayStore->reflectionRays, rayStore->refractionRays, rayStore->diffuseRays,
                            rayStore->nohitRays, rayStore->invisibleRays, rayStore->weakRays, rayStore->overflowRays, 
                            rayStore->totalRays, difftime(end, start), rayStore->totalMem*1.0/(1024.0*1024.0*1024.0));

                delete rayStore;
                // (0,0,0) is the default viewpoint, and it means the end of the list
                if (options[i].viewpoints[j] == 0)
                    break;
            }
        }

        if (options[i].doTraditionalRender == true) {
            // do pure render from eyes without lightRender
            // calcule all the viewpoints with same options 
            std::printf("###traditional render from eye###\n");
            for (int j =0; j<sizeof(options[i].viewpoints)/sizeof(Vec3f); j++) {
                //std::memset(&rayStore, 0, sizeof(rayStore));
                rayStore = new RayStore(options[i]);
                std::sprintf(outfile,
                    "traditional_x.%d_y.%d_z.%d_density.%.2f_dep.%d_spp.%d_split.%d.ppm", (int)options[i].viewpoints[j].x,
                    (int)options[i].viewpoints[j].y, (int)options[i].viewpoints[j].z, RAY_CAST_DESITY, options[i].maxDepth, options[i].spp,
                    options[i].diffuseSpliter);
                // caculate time consumed
                start = time(NULL);
                // finally, eyeRender
                //std::memset(&rayStore, 0, sizeof(rayStore));
                /* start eyeRender after lightRender */
                eyeRender(*rayStore, outfile, options[i], options[i].viewpoints[j], objects, lights, false, false);
                end = time(NULL);
                std::printf("%-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10u %-10.0f %-10.2f\n",
                            options[i].diffuseSpliter, options[i].maxDepth,
                            rayStore->originRays, rayStore->reflectionRays, rayStore->refractionRays, rayStore->diffuseRays,
                            rayStore->nohitRays, rayStore->invisibleRays, rayStore->weakRays, rayStore->overflowRays, 
                            rayStore->totalRays, difftime(end, start), rayStore->totalMem*1.0/(1024.0*1024.0*1024.0));

                rayStore->dumpEyeTraceLink(222, 340);

                delete rayStore;
                // (0,0,0) is the default viewpoint, and it means the end of the list
                if (options[i].viewpoints[j] == 0)
                    break;
            }
        }
    }

    return 0;
}
