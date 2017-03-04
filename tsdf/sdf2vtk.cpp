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

void usage() {
  printf("OVERVIEW: convert sdf file format to xml vtk (vti)\n");
  printf("USAGE: sdf2vtk <input sdf file> <output vti file>\n");
}

int main(int argc, char ** argv) {
  if (argc != 3) {
    usage();
    return EXIT_FAILURE;
  }
  
  fprintf(stderr, "(sdf2vtk) Reading file %s\n", argv[1]);
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
  fprintf(stderr, "(sdf2vtk) Extent: [%g, %g, %g]. Grid size: [%d, %d, %d]\n",
	 xextent, yextent, zextent, NX, NY, NZ);

  std::vector<real> sdf_data(NX*NY*NZ);
  fread(&sdf_data[0], sizeof(real), NX*NY*NZ, f);
  fclose(f);
  
  fprintf(stderr, "(sdf2vtk) Writing file %s \n", argv[2]);
  FILE * v = fopen(argv[2], "w");
  if (v == NULL ) {
    perror("Error");
    return EXIT_FAILURE;
  }

  // spacing
  const real spx = NX>1 ? xextent/(NX-1) : 0;
  const real spy = NY>1 ? yextent/(NY-1) : 0;
  const real spz = NZ>1 ? zextent/(NZ-1) : 0;

  fprintf(v, "<?xml version=\"1.0\"?>\n");
  fprintf(v, "<VTKFile byte_order=\"LittleEndian\" version=\"1.0\" type=\"ImageData\">\n");
  fprintf(v, "<ImageData Origin=\"%-1.16e %-1.16e %-1.16e\" WholeExtent=\"0 %d 0 %d 0 %d\" Spacing=\"%-1.16e %-1.16e %-1.16e\">\n",
	  orgx, orgy, orgz, NX-1, NY-1, NZ-1, spx, spy, spz);
  fprintf(v, "<Piece Extent=\"0 %d 0 %d 0 %d\">\n", NX-1, NY-1, NZ-1);
  fprintf(v, "<PointData Scalars=\"%s\">\n", VTK_SC_NAME);
  fprintf(v, "<DataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"1\" format=\"ascii\"/>", VTK_TYPE, VTK_SC_NAME);
  for (int i=0; i<NX*NY*NZ; i++) {
    if (sdf_data[i] != sdf_data[i]) {
      fprintf(stderr, "(sdf2vtk) see nan in position %d\n", i);
      exit(EXIT_FAILURE);
    }
    fprintf(v, "%-1.6e\n", sdf_data[i]);
  }

  fprintf(v, "</PointData>\n");
  fprintf(v, "</Piece>\n");
  fprintf(v, "</ImageData>\n");
  fprintf(v, "</VTKFile>\n");
  fclose(v);
}

