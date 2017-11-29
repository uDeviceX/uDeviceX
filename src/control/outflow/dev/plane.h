/* plane is described as a*x + b*y + c*z + d = 0 */
struct Params {
    float a, b, c, d;
};

__device__ int predicate(float3 o, Params p, const float r[3]) {
    enum {X, Y, Z};
    float x, y, z, s;
    x = r[X] - o.x;
    y = r[Y] - o.y;
    z = r[Z] - o.z;

    s = p.a * x + p.b * y + p.c * z + p.d;
    
    return s > 0;
}
