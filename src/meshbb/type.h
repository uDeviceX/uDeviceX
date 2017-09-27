typedef float real;
typedef float3 real3;

struct rPa {
    real3 r, v;
};

__device__ float3 make_real3(float a, float b, float c) {
    return make_float3(a, b, c);
}

__device__ double3 make_real3(double a, double b, double c) {
    return make_double3(a, b, c);
}
