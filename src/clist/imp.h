namespace clist {
struct Clist0 {
    int LX, LY, LZ;
    int ncells;
    int *start, *count;
};

class Clist {
    int LX, LY, LZ;
public:
    int ncells;
    int *start, *count;
    Clist(int X, int Y, int Z);
    void build(Particle *const pp, int n);
    ~Clist();
};
} /* namespace */
