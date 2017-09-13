namespace clist {
struct Clist0 {
    int LX, LY, LZ;
    int ncells;
    int *start, *count;
};

class Clist {
    int LX, LY, LZ;
    void buildn(Particle *const pp, const int n);
    void build0();
public:
    int ncells;
    int *start, *count;
    Clist(int X, int Y, int Z);
    void build(Particle *const pp, int n);
    ~Clist();
};
} /* namespace */
