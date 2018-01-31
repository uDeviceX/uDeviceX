struct RigQuants {
    int n, ns, nps;              /* number of particles (total), solid, particle per solid        */
    Particle *pp_hst, *pp;       /* particles on hst and device                                   */
    Solid    *ss_hst, *ss;       /* rigid strutures                                               */
    float   *rr0_hst, *rr0;      /* frozen particle templates                                     */

    /* mesh related quantities */
    int nt, nv;                   /* number of [t]riangles and [v]ertices                          */
    int4 *htt, *dtt;              /* triangle indices of [h]ost and [d]evice                       */
    float *hvv, *dvv;             /* vertices of [h]ost and [d]evice (template)                    */
    Particle *i_pp_hst, *i_pp;    /* particles representing all meshes of all solids of that node  */

    Solid *ss_dmp, *ss_dmp_bb;

    int maxp; /* maximum particle number */
};

struct Coords;

void rig_ini(int maxp, RigQuants *q);
void rig_fin(RigQuants *q);

void rig_gen_quants(const Coords *coords, MPI_Comm comm, Particle *opp, int *on, RigQuants *q);
void rig_strt_quants(const Coords *coords, const int id, RigQuants *q);

void rig_set_ids(MPI_Comm comm, RigQuants *q);

void rig_strt_dump_templ(const Coords *coords, const RigQuants *q);
void rig_strt_dump(const Coords *coords, const int id, const RigQuants *q);
