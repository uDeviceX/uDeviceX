namespace clist {
void build(int n, int xcells, int ycells, int zcells,
           /**/ Particle *pp, int *starts, int *counts);

class Clist {
    int LX, LY, LZ;
    void buildn(Particle *const pp, const int n) {
        clist::build(n, LX, LY, LZ, /**/ pp, start, count);
    }

    void build0() {
        CC(cudaMemsetAsync(start, 0, sizeof(start[0]) * ncells));
        CC(cudaMemsetAsync(count, 0, sizeof(count[0]) * ncells));
    }

public:
    int ncells;
    int *start, *count;
    Clist(int X, int Y, int Z)
    {
        LX = X; LY = Y; LZ = Z;
        ncells = LX * LY * LZ + 1;
        Dalloc(&start, ncells);
        Dalloc(&count, ncells);
    }

    void build(Particle *const pp, int n) {
        if (n) buildn(pp, n); else build0();
    }

    ~Clist() { CC(cudaFree(start)); CC(cudaFree(count)); }
};

}
