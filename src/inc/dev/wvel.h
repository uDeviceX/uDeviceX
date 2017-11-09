namespace k_wvel {
inline __device__ float vell0(float y, float z, float gd) {
    float v, *r;
    enum {X, Y, Z};
    r = glb::r0;
    v = 0.0;
#if   WVEL_PAR_Z
    v = glb::gd * (z - r[Z]);
#elif WVEL_PAR_Y
    v = glb::gd * (y - r[Y]);
#endif
    return v;
}

inline __device__ void vell(float x, float y, float z,
                            float *vx, float *vy, float *vz) {
    *vx = vell0(y, z, glb::gd);
    *vy = 0;
    *vz = 0;
}

inline __device__ void bounce_vel(float   xw, float   yw, float   zw, /* wall */
                                  float* vxp, float* vyp, float* vzp) {
    float vx = *vxp, vy = *vyp, vz = *vzp;
    float vxw, vyw, vzw;
    vell(xw, yw, zw, &vxw, &vyw, &vzw);
    /* go to velocity relative to the wall; bounce; and go back */
    vx -= vxw; vx = -vx; vx += vxw;
    vy -= vyw; vy = -vy; vy += vyw;
    vz -= vzw; vz = -vz; vz += vzw;

    *vxp = vx; *vyp = vy; *vzp = vz;
}

} /* namespace k_wvel */
