/* used in forces.h */
enum {SOLVENT_KIND, SOLID_KIND, WALL_KIND};

#define N_COLOR (4)
enum {BLUE_COLOR, RED_COLOR, /* solvent colors */
      SOLID_COLOR, WALL_COLOR};

/* maximum particle number per one processor for static allocation */
#define MAX_PART_NUM (3*XS*YS*ZS*numberdensity)

/* maximum number of particles per solid */
#define MAX_PSOLID_NUM 12000

/* maximum number of solids per node */
#define MAX_SOLIDS 40

/* maximum number of object types (solid, rbc, ...) */
#define MAX_OBJ_TYPES 2

/* maximum number density of particles of the objects */
#define MAX_OBJ_DENSITY (30)

/* maximum number of red blood cells per node */
#define MAX_CELL_NUM 1500

/* maximum texture size in bytes */
#define MAX_TEXO_SIZE (100000000)

/* safety factor for dpd halo interactions */
#define HSAFETY_FACTOR 10.f

/* safety factor for odist fragments */
#define ODSTR_FACTOR (3)

/* write ascii/bin in l/ply.cu */
#define PLY_WRITE_ASCII
