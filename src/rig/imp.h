namespace rig {

struct Quants {
    int n, ns, nps;              /* number of particles (total), solid, particle per solid        */
    Particle *pp_hst, *pp;       /* particles on hst and device                                   */
    Solid    *ss_hst, *ss;       /* rigid strutures                                               */
    float   *rr0_hst, *rr0;      /* frozen particle templates                                     */

    /* mesh related quantities */
    int nt, nv;                   /* number of [t]riangles and [v]ertices                          */
    int4 *htt;                    /* triangle indices of [h]ost and [d]evice                       */
    int4 *dtt;
    float *hvv, *dvv;             /* vertices of [h]ost and [d]evice (template)                    */
    Particle *i_pp_hst, *i_pp;    /* particles representing all meshes of all solids of that node  */

    Solid *ss_dmp, *ss_dmp_bb;
};

void ini(Quants *q);
void fin(Quants *q);

void gen_quants(Coords coords, MPI_Comm comm, Particle *opp, int *on, Quants *q);
void strt_quants(const int id, Quants *q);

void set_ids(MPI_Comm comm, Quants q);

void strt_dump_templ(const Quants q);
void strt_dump(const int id, const Quants q);

} // rig
