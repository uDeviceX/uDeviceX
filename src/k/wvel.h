namespace k_wvel
{
    /* wall velocity */
    __device__ void vell(float x, float y, float z,
                         float *vxw, float *vyw, float *vzw)
    {
#if defined( shear_z )
        float z0 = glb::r0[2];
        *vxw = gamma_dot * (z - z0); *vyw = 0; *vzw = 0; /* velocity of the wall; */
#elif defined( shear_y )
        float y0 = glb::r0[1];
        *vxw = gamma_dot * (y - y0); *vyw = 0; *vzw = 0; /* velocity of the wall; */
#endif
    }

  __device__ void bounce_vel(float   xw, float   yw, float   zw, /* wall */
			     float* vxp, float* vyp, float* vzp) {
    float vx = *vxp, vy = *vyp, vz = *vzp;
    float vxw, vyw, vzw;
    vell(xw, yw, zw, &vxw, &vyw, &vzw);
    /* go to velocity relative to the wall; bounce; and go back */
    vx -= vxw; vx = -vx; vx += vxw;
    vy -= vyw; vy = -vy; vy += vyw;
    vz -= vzw; vz = -vz; vz += vzw;

    lastbit::Preserver up1(*vxp);
    *vxp = vx; *vyp = vy; *vzp = vz;
  }

} /* namespace k_wvel */
