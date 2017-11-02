static __device__ void fetch(Cloud c, int i, /**/ forces::Pa *p) {
    cloud_get_p(c, i, /**/ p);
}
