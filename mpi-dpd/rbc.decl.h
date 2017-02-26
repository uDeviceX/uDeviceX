namespace rbc {
struct Extent {
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;
};

float *orig_xyzuvw;
float *host_av;
float *devtrs4;

int *triplets;

float *addfrc;
int maxCells;
}
