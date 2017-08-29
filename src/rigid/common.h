namespace rig {

enum {X, Y, Z};
enum {XX, XY, XZ, YY, YZ, ZZ};
enum {YX = XY, ZX = XZ, ZY = YZ};

#define _HD_ __host__ __device__

} // rig
