__global__ void main(const sdf::tex3Dca texsdf,
                     float x, float y, float z) {
    float s;
    s = sdf::sub::dev::sdf(texsdf, x, y, z);
    printf("%g %g %g %g\n", x, y, z, s);
}
