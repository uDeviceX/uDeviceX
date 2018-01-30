enum {frag_bulk = 26};

struct int3;

namespace fraghst {

int i2dx(int i);
int i2dy(int i);
int i2dz(int i);
int i2d(int i, int dir);
void i2d3(int i, /**/ int d[3]);

int d2i(int x, int y, int z);
int d32i(const int d[3]);

int frag_ncell(int3 L, int i);

int frag_ad2i(int x, int y, int z);
int frag_anti(int i);

void frag_estimates(int nfrags, float maxd, /**/ int *cap);

} // fraghst
