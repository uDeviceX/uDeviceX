namespace k_wvel {
inline __device__ void vell(float x, float y, float z,
                            float *vx, float *vy, float *vz) {
    enum {X, Y, Z};
    *vx = *vy = *vz = 0;
#if shear_z
    float *r = glb::r0;
    *vx = gamma_dot * (z - r[Z]);
#elif shear_y
    float *r = glb::r0;    
    *vx = gamma_dot * (y - r[Y]);
#endif
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
