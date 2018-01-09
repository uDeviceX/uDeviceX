static __device__ bool valid_cell(int3 L, int3 c) {
    return c.x < L.x && c.y < L.y && c.z < L.z;
}

static __device__ int get_cid(int3 L, int3 c) {
    return c.x + L.x * (c.y + L.y * c.z);
}

__global__ void sample(Coords coords, int3 L, const int *cellsstart, const int *cellscount, const Particle *pp,
                       /**/ float3 *gridv) {
    Particle p;
    float3 u;
    int pid, cid, num;
    const int3 c = make_int3(threadIdx.x + blockIdx.x * blockDim.x,
                             threadIdx.y + blockIdx.y * blockDim.y,
                             threadIdx.z + blockIdx.z * blockDim.z);

    if (valid_cell(L, c)) {
        cid = get_cid(L, c);
        num = cellscount[cid];
        
        for (pid = cellsstart[cid]; pid < cellsstart[cid] + num; pid++) {
            p = pp[pid];
            u = transform(coords, p);

            gridv[cid].x += u.x / num;
            gridv[cid].y += u.y / num;
            gridv[cid].z += u.z / num;
        }
    }
}
