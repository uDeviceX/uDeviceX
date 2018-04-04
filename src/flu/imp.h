// tag::struct[]
struct FluQuants {
    Particle *pp, *pp0;    /* particles on device  */
    int       n;           /* particle number      */
    Clist      cells;      /* cell lists           */
    ClistMap *mcells;      /* cell lists map       */
    Particle *pp_hst;      /* particles on host    */

    /* optional data */

    bool ids, colors;
    
    int *ii, *ii0;  /* global ids on device */
    int *ii_hst;    /* global ids on host   */

    int *cc, *cc0;  /* colors on device */
    int *cc_hst;    /* colors on host   */

    int maxp; /* maximum particle number */
}; 
// end::struct[]

struct Coords;
struct GenColor;

// tag::mem[]
void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q);
void flu_fin(FluQuants *q);
// end::mem[]

// tag::gen[]
void flu_gen_quants(const Coords*, int numdensity, const GenColor *gc, FluQuants *q); // <1>
void flu_gen_ids(MPI_Comm, const int n, FluQuants *q); // <2>
// end::gen[]

// tag::start[]
void flu_strt_quants(MPI_Comm, const int id, FluQuants *q);     // <1>
void flu_strt_dump(MPI_Comm, const int id, const FluQuants *q); // <2>
// end::start[]

// tag::tools[]
void flu_txt_dump(const Coords*, const FluQuants *q); // <1>

/* build cells only from one array of particles fully contained in the domain */
/* warning: this will delete particles which are outside                      */
void flu_build_cells(/**/ FluQuants *q); // <2>
// end::tools[]
