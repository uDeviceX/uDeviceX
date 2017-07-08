#include <mpi.h>
#include "common.h"
#include "io.h"
#include <conf.h>
#include "m.h"
#include "field.h"

namespace field {
static float spl(float x) { /* b-spline (see tools/bspline.mac) */
  return  \
    x <= 0 ? 0.0 :
    x <= 1 ? x*x*x/6 :
    x <= 2 ? (x*((12-3*x)*x-12)+4)/6 :
    x <= 3 ? (x*(x*(3*x-24)+60)-44)/6 :
    x <= 4 ? (x*((12-x)*x-48)+64)/6 :
    0.0;
}

static void skip_line(FILE *f) {
  char l[BUFSIZ];
  fgets(l, sizeof(l), f);
}

void ini_dims(const char *path, /**/ int N[3], float ext[3]) {
    FILE *f;
    char l[BUFSIZ];
    f = fopen(path, "r");
    fgets(l, sizeof(l), f);
    sscanf(l, "%f %f %f", &ext[0], &ext[1], &ext[2]);
    fgets(l, sizeof(l), f);
    sscanf(l, "%d %d %d", &N[0], &N[1], &N[2]);
    fclose(f);
}
  
void ini_data(const char *path, int n, /**/ float *D) { /* read sdf file */
  FILE *f;
  f = fopen(path, "r");
  skip_line(f); skip_line(f);
  fread(D, sizeof(D[0]), n, f);
  fclose(f);
}

void sample(const float org[3], const float spa[3], const int N1[3], const int N0[3], const float *D0, float *D1) {
    enum {X, Y, Z};
#define OOO(ix, iy, iz) (D1 [ix + N1[X] * (iy + N1[Y] * iz)])
#define DDD(ix, iy, iz) (D0 [ix + N0[X] * (iy + N0[Y] * iz)])
#define i2r(i, d) (org[d] + (i + 0.5) * spa[d] - 0.5)
#define i2x(i)    i2r(i,X)
#define i2y(i)    i2r(i,Y)
#define i2z(i)    i2r(i,Z)
    int iz, iy, ix, i, c, sx, sy, sz;
    float s;
    for (iz = 0; iz < N1[Z]; ++iz)
    for (iy = 0; iy < N1[Y]; ++iy)
    for (ix = 0; ix < N1[X]; ++ix) {
        float r[3] = {(float) i2x(ix), (float) i2y(iy), (float) i2z(iz)};

        int anchor[3];
        for (c = 0; c < 3; ++c) anchor[c] = (int)floor(r[c]);

        float w[3][4];
        for (c = 0; c < 3; ++c)
        for (i = 0; i < 4; ++i)
	  w[c][i] = spl(r[c] - (anchor[c] - 1 + i) + 2);

        float tmp[4][4];
        for (sz = 0; sz < 4; ++sz)
        for (sy = 0; sy < 4; ++sy) {
            s = 0;
            for (sx = 0; sx < 4; ++sx) {
                int l[3] = {sx, sy, sz};
                int g[3];
                for (c = 0; c < 3; ++c)
                g[c] = (l[c] - 1 + anchor[c] + N0[c]) % N0[c];

                s += w[0][sx] * DDD(g[X], g[Y], g[Z]);
            }
            tmp[sz][sy] = s;
        }
        float partial[4];
        for (sz = 0; sz < 4; ++sz) {
            s = 0;
            for (sy = 0; sy < 4; ++sy) s += w[1][sy] * tmp[sz][sy];
            partial[sz] = s;
        }
        float val = 0;
        for (sz = 0; sz < 4; ++sz) val += w[2][sz] * partial[sz];
        OOO(ix, iy, iz) = val;
    }
#undef DDD
#undef OOO
}

static void scale0(int N[3], float s, int x, int y, int z, /**/ float *D) {
  enum {X, Y};
  int i;
  i = x + N[X] * (y + N[Y] * z);
  D[i] *= s;
}

void scale(int N[3], float s, /**/ float *D) {
  enum {X, Y, Z};
  int x, y, z;
  for (z = 0; z < N[Z]; ++z) for (y = 0; y < N[Y]; ++y) for (x = 0; x < N[X]; ++x)
    scale0(N, s, x, y, z, /**/ D);
}

static void dump0(const int N[3], const float ext[3], const float* D0, /**/ float* D1) {
  int c, L[3] = {XS, YS, ZS};
  float org[3], spa[3], ampl;
  for (c = 0; c < 3; ++c) {
    org[c] = m::coords[c] * L[c] / (float)(m::dims[c] * L[c]) * N[c];
    spa[c] = N[c] / (float)(m::dims[c] * L[c]);
  }
  ampl = L[0] / (ext[0] / (float) m::dims[0]);
  sample(org, spa, L, N, D0, /**/ D1);
  scale(L, ampl, /**/ D1);
}

static void dump1(const int N[3], const float ext[3], const float* D, /*w*/ float* W) {
  dump0(N, ext, D, /**/ W);
  H5FieldDump dump;
  dump.scalar(W, "wall");
}

void dump(const int N[], const float ext[], const float* D) {
  float *W = new float[XS * YS * ZS];
  dump1(N, ext, D, /*w*/ W);
  delete[] W;
}
} /* namespace field */
