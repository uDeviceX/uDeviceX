enum {frag_bulk = 26};

struct int3;

namespace fraghst {

int i2dx(int i);
int i2dy(int i);
int i2dz(int i);
void i2d3(int i, /**/ int d[3]);

int d2i(int x, int y, int z);
int d32i(const int d[3]);

int ncell(int3 L, int i);

int antid2i(int x, int y, int z);
int anti(int i);

void estimates(int3 L, int nfrags, float maxd, /**/ int *cap);

} // fraghst
