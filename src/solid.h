namespace solid
{
    void init(Particle *pp, int n, float pmass, float *com,
              /**/ float *rr0, float *Iinv, float *e0, float *e1, float *e2, float *v, float *om);


    void reinit_ft_hst(const int nsolid, /**/ Solid *ss);
    void reinit_ft_dev(const int nsolid, /**/ Solid *ss);

    void update_hst(Force *ff, float *rr0, int n, int nsolid, /**/ Particle *pp, Solid *shst);    
    void update_dev(Force *ff, float *rr0, int n, int nsolid, /**/ Particle *pp, Solid *sdev);

    void generate_hst(const Solid *ss_hst, const int ns, const float *rr0, const int nps, /**/ Particle *pp);
    void generate_dev(const Solid *ss_dev, const int ns, const float *rr0, const int nps, /**/ Particle *pp);

    void mesh2pp_hst(const Solid *ss_hst, const int ns, const Mesh m, /**/ Particle *pp);

    void update_mesh_hst(const Solid *ss_hst, const int ns, const Mesh m, /**/ Particle *pp);
    void update_mesh_dev(const Solid *ss_dev, const int ns, const Mesh m, /**/ Particle *pp);

    void dump(const int it, const Solid *ss, const Solid *ssbb, int nsolid, const int *mcoords);
}
