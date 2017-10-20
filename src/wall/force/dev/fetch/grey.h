static __device__ void fetch(hforces::Cloud c, int i, /**/ forces::Pa *p) {
    hforces::dev::cloud_get_p(c, i, /**/ p);
}
