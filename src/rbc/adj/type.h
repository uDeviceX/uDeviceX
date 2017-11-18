namespace rbc { namespace adj {
struct Map { /* one edge info */
    int i0, i1, i2, i3, i4;
    int rbc; /* cell id */
};

struct AdjHst { /* adjacency lists */
    int *adj0;
    int *adj1;
};

}} /* namespace */
