/* used in forces.h */
enum {SOLVENT_KIND, SOLID_KIND, WALL_KIND};

enum {BLUE_COLOR, RED_COLOR, /* solvent colors */
      SOLID_COLOR,
      WALL_COLOR,
      N_COLOR};

enum {
    /* maximum number of particles per solid */
    MAX_PSOLID_NUM = 12000,

    /* maximum number of solids per node */
    MAX_SOLIDS = 40,

    /* maximum number of object types (solid, rbc, ...) */
    MAX_OBJ_TYPES = 2,

    /* maximum number density of particles of the objects */
    MAX_OBJ_DENSITY = 30,

    /* maximum number of red blood cells per node */
    MAX_CELL_NUM = 1500,

    /* maximum texture size in bytes */
    MAX_TEXO_SIZE = 100000000,

    /* safety factor for dpd halo interactions */
    HSAFETY_FACTOR = 10,

    /* safety factor for odist fragments */
    ODSTR_FACTOR = 3,
};


/* write ascii/bin in l/ply.cu */
#define PLY_WRITE_ASCII
