#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

const int NX = %NX%;
const int NY = %NY%;
const int NZ = %NZ%;

const float xextent = %xextent%;
const float yextent = %yextent%;
const float zextent = %zextent%;

const float xlo      = 0;
const float ylo      = 0;
const float zlo      = 0;

const float xhi      = xextent;
const float yhi      = yextent;
const float zhi      = zextent;

const float xc      = 0.5*xextent;
const float yc      = 0.5*yextent;
const float zc      = 0.5*zextent;

const float x14      = 0.25*xextent;
const float y14      = 0.25*yextent;
const float z14      = 0.25*zextent;

const float x34      = 0.75*xextent;
const float y34      = 0.75*yextent;
const float z34      = 0.75*zextent;

const float Lx      = xextent;
const float Ly      = yextent;
const float Lz      = zextent;

const float pi      = 3.141592653589793;

const float OBJ_MARGIN = %OBJ_MARGIN%;

/* minus infinity */
const float MINF = - std::numeric_limits<float>::max();

using namespace std;

/* local variables */
float sdf[NZ][NY][NX];
float x, y, z;
float h, r0, r2, R2, xp, yp, zp;
float nx, ny, nz, n_abs;
float ax, ay, az, a2;
float dX2, dY2, dZ2, dR2, dR;
float D, rx, ry, ang;
float eg;
float xo, yo, zo; /* center of rotation */

float xorg, yorg, zorg, s;

void rotX(float* x, float* y, float al) {
  float x0 = *x, y0 = *y;
  *x = x0*cos(al) - y0*sin(al);
  *y = x0*sin(al) + y0*cos(al);
}

void rot0(float xo, float yo, float zo, float phix, float phiy, float phiz) {
  x += xo; y += yo; z += yo;

  rotX(&x, &y, phiz);
  rotX(&x, &z, phiy);
  rotX(&y, &z, phix);

  x -= xo; y -= yo; z -= yo;
}

void rot(float phix, float phiy, float phiz) {
  /* update `x', `y', `z' */
  rot0(-xo, -yo, -zo, -phix, -phiy, -phiz);
}

//int   wrap(int i, int n) {return i == n - 1 ? 0 : i;}
float i2x (int i)  {return xextent/(float)(NX-1)*i;}
float i2y (int i)  {return yextent/(float)(NY-1)*i;}
float i2z (int i)  {return zextent/(float)(NZ-1)*i;}

float in_interval(float d, float dlo, float dhi) {
  return
	d < dlo ? 0 :
	d > dhi ? 0 :
	          1
      ;
}

float di(float d, float dlo, float dhi)  {
  return
    d < dlo ? dlo - d :
    d > dhi ? d - dhi :
    0;
}

float de(float d, float dlo, float dhi) {
  float dc = dlo + 0.5*(dhi - dlo);
  return d > dc ? abs(d - dhi) : abs(d - dlo);
}

float min3(float a, float b, float c) {
  return fmin(a, fmin(b, c));
}

float sq(float x) {
  return x*x;
}


float in_range(float s) {
  return
    s >  OBJ_MARGIN ?  OBJ_MARGIN :
    s < -OBJ_MARGIN ? -OBJ_MARGIN :
                                s
  ;
}

float in_box(float x, float y, float z,
		float xlo, float xhi,
		float ylo, float yhi,
		float zlo, float zhi) {
    return
      in_interval(x, xlo, xhi) &&
      in_interval(y, ylo, yhi) &&
      in_interval(z, zlo, zhi);
}

bool in_wall(float s) {
  return s>0;
}

bool in_void(float s) {
  return !in_wall(s);
}

float wall_wins(float so, float sn) {
  // prefer one inside the wall
  if (in_wall(so) && in_void(sn))
    return so;

  // prefer one inside the wall  
  if (in_wall(sn) && in_void(so))
    return sn;

  // `so' and `sn' are positive : prefer smaller (closer to the wall)
  if (in_wall(so) && in_wall(sn))
    return so < sn ? so : sn;

  // `so' and `sn' are negative : prefer bigger (closer to the wall,
  // with smaller abs)
  return so < sn ? sn : so;
}

float void_wins(float so, float sn) {
  // prefer one inside the void
  if (in_wall(so) && in_void(sn))
    return sn;

  // prefer one inside the void
  if (in_wall(sn) && in_void(so))
    return so;

  // `so' and `sn' are positive : prefer smaller (closer to the wall)
  if (in_wall(so) && in_wall(sn))
    return so < sn ? so : sn;

  // `so' and `sn' are negative : prefer bigger (closer to the wall,
  // with smaller abs)
  return so < sn ? sn : so;
}

int main(int /*argc */, char **argv) {
  
  FILE * f = fopen(argv[1], "w");
  fprintf(f, "%f %f %f\n", xextent, yextent, zextent);
  fprintf(f, "%d %d %d\n", NX, NY, NZ);

  for (int i = 0; i < NX; i++) {
    xorg = i2x(i);
    for (int j = 0; j < NY; j++) {
      yorg = i2y(j);
      for (int k = 0; k < NZ; k++) {
	float x0, y0, z0;
	zorg = i2z(k);
	s = MINF; // assume we are very far from the walls (sdf =
		  // -inf)
	//%update_sdf%

	s = in_range(s);
	sdf[k][j][i] = s;
      }
    }
  }

  fwrite(sdf[0], sizeof(float), NX*NY*NZ, f);
  fclose(f);
}
