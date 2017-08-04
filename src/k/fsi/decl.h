namespace k_fsi {
texture<float2, cudaTextureType1D> texSolventParticles;
texture<int, cudaTextureType1D> texCellsStart, texCellsCount;
namespace g {
__constant__ int starts[27], counts[26];
__constant__ Particle *pp[26];
__constant__ Force *ff[26];
}
}
