namespace clist {
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
