namespace k_wvel { /* wall velocity */
  __device__ void vell(float x, float y, float z,
		       float *vxw, float *vyw, float *vzw) {
    float z0 = glb::r0[2];
    *vxw = gamma_dot * (z - z0); *vyw = 0; *vzw = 0; /* velocity of the wall; */
  }

  __device__ void bounce_vel(float   xw, float   yw, float   zw, /* wall */
			     float* vxp, float* vyp, float* vzp) {
    float vx = *vxp, vy = *vyp, vz = *vzp;
    float vxw, vyw, vzw; vell(xw, yw, zw, &vxw, &vyw, &vzw);
    /* go to velocity relative to the wall; bounce; and go back */
    vx -= vxw; vx = -vx; vx += vxw;
    vy -= vyw; vy = -vy; vy += vyw;
    vz -= vzw; vz = -vz; vz += vzw;

    lastbit::Preserver up1(*vxp);
    *vxp = vx; *vyp = vy; *vzp = vz;
  }

} /* namespace k_wvel */
