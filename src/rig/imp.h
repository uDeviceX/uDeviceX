// tag::quants[]
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
// end::quants[]

struct Coords;
struct RigPinInfo;

// tag::mem[]
void rig_ini(int maxp, RigQuants *q);
void rig_fin(RigQuants *q);
// end::mem[]

// tag::gen[]
void rig_gen_quants(const Coords *coords, bool empty_pp, float rig_mass, const RigPinInfo *pi, MPI_Comm comm, Particle *opp, int *on, RigQuants *q);
void rig_strt_quants(const Coords *coords, const int id, RigQuants *q);
// end::gen[]

// tag::genid[]
void rig_set_ids(MPI_Comm comm, RigQuants *q);
// end::genid[]

// tag::io[]
void rig_strt_dump_templ(const Coords *coords, const RigQuants *q);         // <1>
void rig_strt_dump(const Coords *coords, const int id, const RigQuants *q); // <2>
// tag::io[]
