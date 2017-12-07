__global__ void main(const tex3Dca texsdf,
                     float x, float y, float z) {
    float s;
    s = sdf(texsdf, x, y, z);
    printf("%g %g %g %g\n", x, y, z, s);
}
