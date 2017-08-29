namespace rig
{
void ini(const Particle *pp, int n, float pmass, const float *com, const Mesh mesh, /**/ float *rr0, Solid *s);
    
void reinit_ft(const int nsolid, /**/ Solid *ss);

void update(const Force *ff, const float *rr0, int n, int nsolid, /**/ Particle *pp, Solid *sdev);

void generate(const Solid *ss, const int ns, const float *rr0, const int nps, /**/ Particle *pp);

void mesh2pp(const Solid *ss_hst, const int ns, const Mesh m, /**/ Particle *pp);

void update_mesh(const Solid *ss, const int ns, const Mesh m, /**/ Particle *pp);

void dump(const int it, const Solid *ss, const Solid *ssbb, int nsolid, const int *mcoords);
}
