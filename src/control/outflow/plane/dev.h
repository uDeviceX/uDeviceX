/* plane is described as a*x + b*y + c*z + d = 0 */
struct ParamsPlane {
    float a, b, c, d;
};

__device__ int predicate(ParamsPlane p, const float r[3]) {
    enum {X, Y, Z};
    float s;
    
    s = p.a * r[X] + p.b * r[Y] + p.c * r[Z] + p.d;
    
    return s > 0;
}
