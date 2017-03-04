#include <vector>
#include <limits>
#include <cstdio>
#include <cstdlib>

typedef float real;

/* name of scalar field variable in vtk file */
const char* VTK_SC_NAME = "wall";
const char* VTK_TYPE    = "Float64";

/* origin in vtk file */
const real orgx = 0.0;
const real orgy = 0.0;
const real orgz = 0.0;

// grid size
int NX, NY, NZ;

// domain
real xextent, yextent, zextent;

std::vector<real> data1;
std::vector<real> data2;
std::vector<real> data_out;

bool in_wall(float s) {return s>0;}

bool in_void(float s) {return !in_wall(s);}

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

void usage() {
  printf("OVERVIEW: merge two sdf files\n");
  printf("USAGE: mergesdf <sdf file1> <sdf file2> <output sdf file>\n");
}

void read_file(char const* fname, std::vector<real>& data) {
  fprintf(stderr, "(mergesdf) Reading file %s\n", fname);
  FILE * f = fopen(fname, "r");
  if (f == NULL ) {
    perror("Error");
    exit(EXIT_FAILURE);
  }
  
  fscanf(f, "%f %f %f\n", &xextent, &yextent, &zextent);
  fscanf(f, "%d %d %d\n", &NX, &NY, &NZ);
  fprintf(stderr, "(mergesdf) Extent: [%g, %g, %g]. Grid size: [%d, %d, %d]\n",
	 xextent, yextent, zextent, NX, NY, NZ);

  data.resize(NX*NY*NZ);
  fread(&data[0], sizeof(real), NX*NY*NZ, f);
  fclose(f);
}

void write_file(const char* fname, std::vector<real>& data) {
  fprintf(stderr, "(mergesdf) Writing file %s\n", fname);
  FILE * f = fopen(fname, "w");
  if (f == NULL ) {
    perror("Error");
    exit(EXIT_FAILURE);
  }
  
  fprintf(f, "%g %g %g\n", xextent, yextent, zextent);
  fprintf(f, "%d %d %d\n", NX, NY, NZ);

  fwrite(&data[0], sizeof(real), NX*NY*NZ, f);
  fclose(f);
}

void req_size() {
  if (data1.size() == data2.size()) return;

  fprintf(stderr, "(mergesdf.cpp)(ERROR) files has different number of nodes\n");
  fprintf(stderr, "(mergesdf.cpp)(ERROR) data1.size(): %lu\n", data1.size());
  fprintf(stderr, "(mergesdf.cpp)(ERROR) data2.size(): %lu\n", data2.size());

  exit(EXIT_FAILURE);
}

int main(int argc, char ** argv) {
  if (argc != 4) {
    usage();
    return EXIT_FAILURE;
  }

  read_file(argv[1], data1);
  read_file(argv[2], data2);

  req_size();
  
  data_out.resize(NX*NY*NZ);
  for (size_t i=0; i<data_out.size(); i++)
    data_out[i] = wall_wins(data1[i], data2[i]); // TODO: make an option for `void_wins'

  write_file(argv[3], data_out);
}
