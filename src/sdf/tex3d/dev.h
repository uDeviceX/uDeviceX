static __device__ float fetch(Tex3d q, float i, float j, float k) {
    return Ttex3D(float, q->t, i, j, k);
}
