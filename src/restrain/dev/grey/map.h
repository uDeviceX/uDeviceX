struct Map {
    int *cc;
    int color;
};
struct __device__ int validp(const Map m, int i) { return m.cc[i] == color; }
