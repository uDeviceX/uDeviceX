struct Map {
    int *cc;
    int color;
};
static __device__ int validp(const Map m, int i) {
    return m.cc[i] == m.color;
}
