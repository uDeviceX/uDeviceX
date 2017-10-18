/* used in forces.h */
enum {SOLVENT_KIND, SOLID_KIND, WALL_KIND};

enum {BLUE_COLOR, RED_COLOR};

/* maximum particle number per one processor for static allocation */
#define MAX_PART_NUM (3*XS*YS*ZS*numberdensity)

/* maximum number of particles per solid */
#define MAX_PSOLID_NUM 30000

/* maximum number of solids per node */
#define MAX_SOLIDS 40

/* maximum number of object types (solid, rbc, ...) */
#define MAX_OBJ_TYPES 2

/* maximum number density of particles of the objects */
#define MAX_OBJ_DENSITY (100)

/* maximum number of faces and vertices per one RBC */
#define MAX_FACE_NUM 1000
#define MAX_VERT_NUM 2000

/* maximum number of red blood cells per node */
#define MAX_CELL_NUM 500

/* maximum texture size in bytes */
#define MAX_TEXO_SIZE 5000000

/* safety factor for dpd halo interactions */
#define HSAFETY_FACTOR 10.f

/* safety factor for odist fragments */
#define ODSTR_FACTOR (3)

/* write ascii/bin in l/ply.cu */
#define PLY_WRITE_ASCII
