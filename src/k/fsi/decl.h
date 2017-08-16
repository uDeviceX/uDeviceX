namespace k_fsi {
namespace t {
texture<float2, cudaTextureType1D> pp;
texture<int, cudaTextureType1D> start, count;
}
namespace g {
__constant__ int starts[27], counts[26];
__constant__ Particle *pp[26];
__constant__ Force *ff[26];
}
}
