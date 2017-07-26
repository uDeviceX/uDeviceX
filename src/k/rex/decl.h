namespace k_rex { namespace g { /* globals */
__constant__ int capacities[26], *scattered_indices[26];
__device__ bool failed;
__constant__ int offsets[26];
__constant__ int counts[26], tstarts[27], starts[27];
__constant__ float *recvbags[26];
}}
