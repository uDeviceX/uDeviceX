namespace rig {

struct RigQuants {
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

void ini(RigQuants *q);
void fin(RigQuants *q);

void gen_quants(Coords coords, MPI_Comm comm, Particle *opp, int *on, RigQuants *q);
void strt_quants(Coords coords, const int id, RigQuants *q);

void set_ids(MPI_Comm comm, RigQuants q);

void strt_dump_templ(Coords coords, const RigQuants q);
void strt_dump(Coords coords, const int id, const RigQuants q);

} // rig
