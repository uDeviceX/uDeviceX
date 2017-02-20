namespace SolutePUP {
  __constant__ int ccapacities[26], *scattered_indices[26];
  __device__ bool failed;
  __constant__ int coffsets[26];
  __constant__ int ccounts[26], cbases[27], cpaddedstarts[27];
}
