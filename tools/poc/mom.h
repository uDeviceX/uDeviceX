void bb_lin(float* V0, float* V1,
	    float dt , /**/ float* f) {
  enum {X, Y, Z};
  const float m = 1;
  float fx, fy, fz;

  float V0x = V0[X], V0y = V0[Y], V0z = V0[Z];
  float V1x = V1[X], V1y = V1[Y], V1z = V1[Z];

  fx = -((V1x-V0x)*m)/dt;
  fy = -((V1y-V0y)*m)/dt;
  fz = -((V1z-V0z)*m)/dt;

  f[X] =  fx;  f[Y] =  fy;  f[Z] =  fz;
}

void bb_ang(float* R0, float* R1, float* V0, float* V1,
	    float dt , /**/ float* to) {
  const float m = 1;
  enum {X, Y, Z};

  float R0x = R0[X], R0y = R0[Y], R0z = R0[Z];
  float R1x = R1[X], R1y = R1[Y], R1z = R1[Z];

  float V0x = V0[X], V0y = V0[Y], V0z = V0[Z];
  float V1x = V1[X], V1y = V1[Y], V1z = V1[Z];

  float tox, toy, toz;
  tox = -((R1y*V1z-R1z*V1y-R0y*V0z+R0z*V0y)*m)/dt;
  toy =  ((R1x*V1z-R1z*V1x-R0x*V0z+R0z*V0x)*m)/dt;
  toz = -((R1x*V1y-R1y*V1x-R0x*V0y+R0y*V0x)*m)/dt;

  to[X] = tox; to[Y] = toy; to[Z] = toz;
}
