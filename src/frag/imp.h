enum {frag_bulk = 26};

struct int3;

namespace fraghst {

int frag_i2dx(int i);
int frag_i2dy(int i);
int frag_i2dz(int i);
int frag_i2d(int i, int dir);
void frag_i2d3(int i, /**/ int d[3]);

int frag_d2i(int x, int y, int z);
int frag_d32i(const int d[3]);

int frag_ncell(int3 L, int i);

int frag_ad2i(int x, int y, int z);
int frag_anti(int i);

void frag_estimates(int nfrags, float maxd, /**/ int *cap);

} // fraghst
