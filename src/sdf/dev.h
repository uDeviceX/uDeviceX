__device__ __forceinline__
float fetch(const tex3Dca sdf, const float i, const float j, const float k) {
    return Ttex3D(float, sdf.to, i, j, k);
}
