namespace RedistPartKernels {
  __constant__ RedistPart::PackBuffer pack_buffers[27];
  __constant__ RedistPart::UnpackBuffer unpack_buffers[27];
  __device__   int pack_count[27], pack_start_padded[28];
  __constant__ int unpack_start[28], unpack_start_padded[28];
  __device__ bool failed;

  int ntexparticles = 0;
  float2 * texparticledata;
  texture<float, cudaTextureType1D> texAllParticles;
  texture<float2, cudaTextureType1D> texAllParticlesFloat2;
}
