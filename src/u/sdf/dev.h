__global__ void main(Sdf_v sdf_v, float x, float y, float z) {
    float s;
    s = sdf(&sdf_v, x, y, z);
    printf("%g %g %g %g\n", x, y, z, s);
}
