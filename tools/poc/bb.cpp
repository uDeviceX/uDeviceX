
#include "math.h"
#include "intersect.h"
#include "mom.h"

void cross(float* a, float* b, /**/ float* c) {
  enum {X, Y, Z};
  c[X] = a[Y]*b[Z]-b[Y]*a[Z];
  c[Y] = b[X]*a[Z]-a[X]*b[Z];
  c[Z] = a[X]*b[Y]-b[X]*a[Y];
}

void wavg(float* R0, float* R1, float t, /**/ float* Rt) {
  enum {X, Y, Z};
  Rt[X] = R0[X]*(1-t) + R1[X]*t;
  Rt[Y] = R0[Y]*(1-t) + R1[Y]*t;
  Rt[Z] = R0[Z]*(1-t) + R1[Z]*t;
}

/* bounce velocity of the particles
   V0: initial velocity,
   Vw: wall velocity */
void bounce_vel(float* V0, float* Vw, /**/ float* V1) {
  for (int c = 0; c < 3; c++) {
    V1[c]  =  V0[c];
    
    V1[c] -=  Vw[c];
    V1[c]  = -V1[c];
    V1[c] +=  Vw[c];
  }
}

void  bounce_pos(float* R1, float* Rt, /**/ float* Rn) {
  for (int c = 0; c < 3; c++)
    Rn[c] = Rt[c]-(R1[c]-Rt[c]);
}

/* angular velocity and position to linear velocity */
void om2lin(float* om, float* r, /**/ float* v) {
  cross(om, r, /**/ v);
}

int main() {
/*float* R1 = &pp.r[ip];
  float* V0 = &pp.v[ip]; */
  float R1[] = {0, 0, 0}; /* current positon of a particle */
  float V0[] = {1, 1, 1}; /* current velocity of a particle */
  
  float om[] = {0, 0, 1};
  float  f[] = {0, 0, 0};
  float to[] = {0, 0, 0};
  
  float dt = 0.1; /* timestep */
  float radius = 1.0; /* sphere radius */
  
  float t; /* parameter of the intersection [0, 1] */
  
  int c;
  float R0[3];
  for (c = 0; c < 3; c++) R0[c] = R1[c] - dt*V0[c];

  bool ok = intersect(R0, R1, radius, /**/ &t);
  if (!ok) return 0;

  float Rt[3];
  wavg(R0, R1, t, /**/ Rt);

  float Vw[3];
  om2lin(om, Rt, /**/ Vw);

  float V1[3];
  bounce_vel(V0, Vw, /**/ V1);

  float Rn[3];
  bounce_pos(R1, Rt, /**/ Rn);

  float f0[3], to0[3];
  bb_lin(        V0, V1, dt, /**/  f0);
  bb_ang(R0, R1, V0, V1, dt, /**/ to0);

  for (c = 0; c < 3; c++) {
     f[c] +=  f0[c];
    to[c] += to0[c];
  }

  for (c = 0; c < 3; c++) { /* update velocity and positon */
    V0[c] = V1[c]; R1[c] = Rn[c];
  }

}
