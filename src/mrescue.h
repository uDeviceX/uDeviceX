namespace mrescue
{
    void init(int n);
    void close();

    void rescue_hst(const int *tt, const int nt, const Particle *ipp, const int n, /**/ Particle *pp);
    //void rescue_dev(const int *tt, const int nt, const Particle *ipp, const int n, /**/ Particle *pp);
}
