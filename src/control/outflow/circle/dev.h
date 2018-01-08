struct ParamsCircle {
    float Rsq;  /* radius squared  */
    float3 c;   /* center          */
};

__device__ int predicate(ParamsCircle p, const float r[3]) {
    enum {X, Y, Z};
    float x, y, rsq;
    x = r[X] - p.c.x;
    y = r[Y] - p.c.y;

    rsq = x*x + y*y;
    return rsq > p.Rsq;
}
