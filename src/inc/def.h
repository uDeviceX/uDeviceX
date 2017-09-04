enum { /* used in forces.h */
    SOLVENT_KIND = 0,
    SOLID_KIND   = 1,
    WALL_KIND    = 2,
    RBC_KIND     = 3,
};

/* maximum particle number per one processor for static allocation */
#define MAX_PART_NUM 1000000

/* maximum number of particles per solid */
#define MAX_PSOLID_NUM 30000

/* maximum number of solids per node */
#define MAX_SOLIDS 20

/* maximum number of object types (solid, rbc, ...) */
#define MAX_OBJ_TYPES 10

/* maximum number denstion of particles of the objects */
#define MAX_OBJ_DENSITY (100)

/* maximum number of faces and vertices per one RBC */
#define MAX_FACE_NUM 5000
#define MAX_VERT_NUM 10000

/* maximum number of red blood cells per node */
#define MAX_CELL_NUM 100

/* maximum texture size in bytes */
#define MAX_TEXO_SIZE 5000000

/* safety factor for dpd halo interactions */
#define HSAFETY_FACTOR 10.f

/* safety factor for odist fragments */
#define ODSTR_FACTOR (3)

/* write ascii/bin in l/ply.cu */
#define PLY_WRITE_ASCII
