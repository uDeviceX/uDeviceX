#include <vector>
#include <cstdio>
#include <cstdlib>

typedef float real;

int NX, NY, NZ; // grid size
real xextent, yextent, zextent; // domain
real sx, sy, sz;                // spacing

real xl, yl, zl;                // region to compute volume
real xh, yh, zh;

void usage() {
  printf("OVERVIEW: report information about the volume occupaied by wall and void in sdf file\n");
  printf("USAGE: sdf2volume <input sdf file> <xl> <yl> <zl> <xh> <yh> <zh>\n");
}

real i2x (int i)  {return i*sx;} // index to coordinate
real i2y (int i)  {return i*sy;}
real i2z (int i)  {return i*sz;}

int safe_minus(int i) {return i>1 ? i - 1 : 1;}

bool in_range(real x, real y, real z) {
  return x>=xl && y>=yl && z>=zl && x<xh && y<yh && z<zh;
}

int main(int argc, char ** argv) {
  if (argc != 8) {
    usage();
    return EXIT_FAILURE;
  }
  
  FILE * f = fopen(argv[1], "r");
  if (f == NULL ) {
    printf("(sdf2volume) cannot open file %s\n", argv[1]);
    return EXIT_FAILURE;
  }

  xl = atof(argv[2]); yl = atof(argv[3]); zl = atof(argv[4]);
  xh = atof(argv[5]); yh = atof(argv[6]); zh = atof(argv[7]);

  fscanf(f, "%f %f %f\n", &xextent, &yextent, &zextent);
  fscanf(f, "%d %d %d\n", &NX, &NY, &NZ);

  sx = xextent/safe_minus(NX); // spacing
  sy = yextent/safe_minus(NY);
  sz = zextent/safe_minus(NZ);
  
  fprintf(stderr, "(sdf2volume) sx sy sz: %g %g %g\n", sx, sy, sz);
  fprintf(stderr, "(sdf2volume) xl yl zl: %g %g %g\n", xl, yl, zl);
  fprintf(stderr, "(sdf2volume) xh yh zh: %g %g %g\n", xh, yh, zh);  
  
  std::vector<real> sdf_data(NX*NY*NZ);
  int n_total = NX*NY*NZ;
  fread(&sdf_data[0], sizeof(real), n_total, f);
  fclose(f);

  int n_wall = 0;
  int n_void = 0;
  int idx = 0;
  for (int k = 0; k < NZ; k++) {
    real z = i2z(k);
    for (int j = 0; j < NY; j++) {
      real y = i2y(j);
      for (int i = 0; i < NX; i++) {
	real x = i2x(i);

	real wall = sdf_data[idx++];
	if (!in_range(x, y, z)) continue;
	if (wall<0) n_void++; else n_wall++;
      }
    }
  }
	
  real Vcell = sx*sy*sz;

  printf("total: %g\n", Vcell*(n_void+n_wall));
  printf("void: %g\n", Vcell*n_void);
  printf("wall: %g\n", Vcell*n_wall);
}

