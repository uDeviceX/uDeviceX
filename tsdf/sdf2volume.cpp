#include <vector>
#include <limits>
#include <cstdio>
#include <cstdlib>

typedef float real;

/* origin in vtk file */
const real orgx = 0.0;
const real orgy = 0.0;
const real orgz = 0.0;

void usage() {
  printf("OVERVIEW: report information about the volume occupaied by wall and void in sdf file\n");
  printf("USAGE: sdf2volume <input sdf file>\n");
  printf("Output: <volume of void>\n<volume of wall>\n");
}

int main(int argc, char ** argv) {
  if (argc != 2) {
    usage();
    return EXIT_FAILURE;
  }
  
  FILE * f = fopen(argv[1], "r");
  if (f == NULL ) {
    perror("Error");
    return EXIT_FAILURE;
  }

  // grid size
  int NX, NY, NZ;
  // domain
  real xextent, yextent, zextent;
  fscanf(f, "%f %f %f\n", &xextent, &yextent, &zextent);
  fscanf(f, "%d %d %d\n", &NX, &NY, &NZ);
  int n_total = NX*NY*NZ;
  
  std::vector<real> sdf_data(NX*NY*NZ);
  fread(&sdf_data[0], sizeof(real), n_total, f);
  fclose(f);

  int n_wall = 0;
  int n_void = 0;
  for (int i=0; i<n_total; i++) {
    if (sdf_data[i]<=0)
      n_void++;
    else
      n_wall++;
  }

  real Vtotal = xextent*yextent*zextent;
  printf("%g\n%g\n", (Vtotal*n_void)/n_total, (Vtotal*n_wall)/n_total);
}

