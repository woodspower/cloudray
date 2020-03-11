/*********************************************************
    Wang Xinhou 
    Optimization algorithms for cloud ray tracing
    Usage: TBD, for now copy these functions when needed
*********************************************************/

// [comment]
// Calc the min sqr value for point located between [a, b]
// if a * b < 0 then 0 is in [a, b], the min sql is 0
// if a * b > 0 then 0 is not in [a, b], the min sql is min(a*a, b*b)
// [/comment]
inline
float minAbs(const float &a, const float &b)
{ 
    if (a*b < 0) return 0;
    if (std::abs(a) < std::abs(b)) return a;
    return b;
}

inline
float maxAbs(const float &a, const float &b)
{
    if (std::abs(a) < std::abs(b)) return b;
    return a;
}

// [comment]
// inputs include the viewpoints, reflectPoints, number of viewpoints vNums, 
// number of reflectPoints pNums.
// use the limts of viewpoints and reflectPoints to calc the range of theta and phi
// theta is in [acos(yMax/rMin), acos(yMin/rMax)]
// phi is in [min(atan2(z,x)), max(atan2(z,x))], special case, x can be zero
// [/comment]
void limitVHAngle(
    Vec3f *viewpoints,
    Vec3f *reflectPoints,
    uint32_t vNums,
    uint32_t pNums,
    float &thetaMin, 
    float &thetaMax, 
    float &phiMin, 
    float &phiMax)
{
    Vec3f reflectPointMin = Vec3f(100000.);
    Vec3f reflectPointMax = Vec3f(-100000.);
    Vec3f viewpointMin = Vec3f(100000.);
    Vec3f viewpointMax = Vec3f(-100000.);
    for(uint32_t i=0; i<vNums; i++) {
        if (viewpoints[i].x > viewpointMax.x) viewpointMax.x = viewpoints[i].x;
        if (viewpoints[i].y > viewpointMax.y) viewpointMax.y = viewpoints[i].y;
        if (viewpoints[i].z > viewpointMax.z) viewpointMax.z = viewpoints[i].z;
        if (viewpoints[i].x < viewpointMin.x) viewpointMin.x = viewpoints[i].x;
        if (viewpoints[i].y < viewpointMin.y) viewpointMin.y = viewpoints[i].y;
        if (viewpoints[i].z < viewpointMin.z) viewpointMin.z = viewpoints[i].z;
    }
    for(uint32_t i=0; i<pNums; i++) {
        if (reflectPoints[i].x > reflectPointMax.x) reflectPointMax.x = reflectPoints[i].x;
        if (reflectPoints[i].y > reflectPointMax.y) reflectPointMax.y = reflectPoints[i].y;
        if (reflectPoints[i].z > reflectPointMax.z) reflectPointMax.z = reflectPoints[i].z;
        if (reflectPoints[i].x < reflectPointMin.x) reflectPointMin.x = reflectPoints[i].x;
        if (reflectPoints[i].y < reflectPointMin.y) reflectPointMin.y = reflectPoints[i].y;
        if (reflectPoints[i].z < reflectPointMin.z) reflectPointMin.z = reflectPoints[i].z;
    }
    Vec3f PVMin = viewpointMin - reflectPointMax;
    Vec3f PVMax = viewpointMax - reflectPointMin;
    float minX, minY, minZ, maxX, maxY, maxZ;
    minX = minAbs(PVMin.x, PVMax.x);
    minY = minAbs(PVMin.y, PVMax.y);
    minZ = minAbs(PVMin.z, PVMax.z);
    maxX = maxAbs(PVMin.x, PVMax.x);
    maxY = maxAbs(PVMin.y, PVMax.y);
    maxZ = maxAbs(PVMin.z, PVMax.z);
    const float rMin = sqrt(minX*minX+minY*minY+minZ*minZ);
    const float rMax = sqrt(maxX*maxX+maxY*maxY+maxZ*maxZ);
    thetaMin = acos(PVMax.y/rMin);
    thetaMax = acos(PVMin.y/rMax);
    phiMin = std::min(
             std::min(std::min(atan2(PVMax.z,PVMax.x),atan2(PVMax.z, PVMin.x)),
                      std::min(atan2(PVMin.z,PVMax.x),atan2(PVMin.z, PVMin.x))),
             std::min(atan2(PVMin.z,minX),atan2(PVMax.z,minX)));
    phiMax = std::max(
             std::max(std::max(atan2(PVMax.z,PVMax.x),atan2(PVMax.z, PVMin.x)),
                      std::max(atan2(PVMin.z,PVMax.x),atan2(PVMin.z, PVMin.x))),
             std::max(atan2(PVMin.z,minX),atan2(PVMax.z,minX)));
}

