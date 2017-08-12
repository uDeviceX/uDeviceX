namespace clist {
void build(int n, int xcells, int ycells, int zcells,
           float xstart, float ystart, float zstart,
           /**/ Particle *pp, int *starts, int *counts);

/* container for the cell lists, which contains only two integer
   vectors of size ncells.  the start[cell-id] array gives the entry in
   the particle array associated to first particle belonging to cell-id
   count[cell-id] tells how many particles are inside cell-id.  building
   the cell lists involve a reordering of the particle array (!) */

class Clist {
    const int LX, LY, LZ;

    void buildn(Particle *const pp, const int n) {
        clist::build(n, LX, LY, LZ, -LX/2, -LY/2, -LZ/2, /**/ pp, start, count);
    }

    void build0() {
        CC(cudaMemsetAsync(start, 0, sizeof(start[0]) * ncells));
        CC(cudaMemsetAsync(count, 0, sizeof(count[0]) * ncells));
    }

public:
    const int ncells;
    int *start, *count;
    Clist(const int X, const int Y, const int Z)
        : ncells(X*Y*Z + 1), LX(X), LY(Y), LZ(Z) {
        MSG("%s:%d: ncells: %n", __FILE__, __LINE__, ncells);
        CC(cudaMalloc(&start, sizeof(start[0]) * ncells));
        CC(cudaMalloc(&count, sizeof(count[0]) * ncells));
    }

    void build(Particle *const pp, const int n) {
        if (n) buildn(pp, n); else build0();
    }

    ~Clist() { CC(cudaFree(start)); CC(cudaFree(count)); }
};

}
