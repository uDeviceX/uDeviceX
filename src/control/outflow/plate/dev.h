/* plane is described as a*x + b*y + c*z + d = 0 */
struct ParamsPlate {
    float a, b, c, d;
};

__device__ int predicate(ParamsPlate p, const float r[3]) {
    enum {X, Y, Z};
    float s;
    
    s = p.a * r[X] + p.b * r[Y] + p.c * r[Z] + p.d;
    
    return s > 0;
}
