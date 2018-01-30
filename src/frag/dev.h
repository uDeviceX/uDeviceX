#if HOST
  #define _I_
  #define _S_ static
#else
  #define _I_ static __device__
  #define _S_ static __device__
#endif

enum {frag_bulk = 26};
enum {BAD_DIR = -2};

/* fragment id to direction                    */

_I_ int frag_i2dx(int i) { return (i     + 2) % 3 - 1; }
_I_ int frag_i2dy(int i) { return (i / 3 + 2) % 3 - 1; }
_I_ int frag_i2dz(int i) { return (i / 9 + 2) % 3 - 1; }

_I_ int frag_i2d(int i, int dir) {
    enum {X, Y, Z};
    switch (c) {
    case X:
        return frag_i2dx(i);
        break;
    case Y:
        return frag_i2dy(i);
        break;
    case Z:
        return frag_i2dz(i);
        break;
    default:
        return BAD_DIR;
        break;
    };
}

_I_ void frag_i2d3(int i, /**/ int d[3]) {
    enum {X, Y, Z};
    d[X] = frag_i2dx(i);
    d[Y] = frag_i2dy(i);
    d[Z] = frag_i2dz(i);
}

/* direction to fragment id                    */
_I_ int frag_d2i(int x, int y, int z) {
    return    ((x + 2) % 3)
        + 3 * ((y + 2) % 3)
        + 9 * ((z + 2) % 3);
}

_I_ int frag_d32i(const int d[3])  {
    enum {X, Y, Z};
    return frag_d2i(d[X], d[Y], d[Z]);
}

/* number of cells in direction x, y, z        */
_S_ int frag_ncell0(int3 L, int x, int y, int z) {
    int nx, ny, nz;
    nx = x ? 1 : L.x;
    ny = y ? 1 : L.y;
    nz = z ? 1 : L.z;
    return nx * ny * nz; 
}

/* number of cells in fragment i               */
_I_ int frag_ncell(int3 L, int i) {
    int x, y, z;
    x = frag_i2dx(i);
    y = frag_i2dy(i);
    z = frag_i2dz(i);
    return frag_ncell0(L, x, y, z);
}

/* anti direction to fragment id                */
_I_ int frag_ad2i(int x, int y, int z) {
    return frag_d2i(-x, -y, -z);
}

/* anti fragment                                */
_I_ int frag_anti(int i) {
    int x, y, z;
    x = frag_i2dx(i);
    y = frag_i2dy(i);
    z = frag_i2dz(i);
    return frag_ad2i(x, y, z);
}

/* [f]ragment [id] : where is `i' in sorted a[27]? */
static __device__ int frag_get_fid(const int a[], const int i) {
    int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

#undef _I_
#undef _S_
