namespace rig {
namespace sub {

void load_solid_mesh(const char *fname, Mesh *m_dev, Mesh *m_hst);

void gen_from_solvent(const Mesh m_hst,  /* io */ Particle *opp, int *on,
                      /* o */ int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst, Particle *pp_hst);

void gen_from_strt(const int id, int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst);
void gen_pp_hst(const int ns, const float *rr0_hst, const int nps, /**/ Solid *ss_hst, Particle *pp_hst);
void gen_ipp_hst(const Solid *ss_hst, const int ns, const Mesh m_hst, Particle *i_pp_hst);

void set_ids(const int ns, Solid *ss_hst, Solid *ss_dev);

void strt_dump_templ(const int nps, const float *rr0_hst);
void strt_dump(const int id, const int ns, const Solid *ss);

} // sub
} // rig
