enum { /* used in forces.h */
    SOLVENT_TYPE = 0,
    SOLID_TYPE   = 1,
    WALL_TYPE    = 2,
    RBC_TYPE     = 3,
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
#define MAX_OBJ_DENSITY 20

/* maximum number of faces and vertices per one RBC */
#define MAX_FACE_NUM 5000
#define MAX_VERT_NUM 10000

/* maximum number of red blood cells per node */
#define MAX_CELL_NUM 100

/* safety factor for dpd halo interactions */
#define HSAFETY_FACTOR 10.f

/* write ascii/bin in l/ply.cu */
#define PLY_WRITE_ASCII

#define MSG00(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#define MSG(fmt, ...) MSG00("%03d: ", m::rank), MSG00(fmt, ##__VA_ARGS__), MSG00("\n")
#define MSG0(fmt, ...) do { if (m::rank == 0) MSG(fmt, ##__VA_ARGS__); } while (0)

#define ERR(fmt, ...) do { fprintf(stderr, "%03d: ERROR: %s:%d: " fmt, m::rank, __FILE__, __LINE__, ##__VA_ARGS__); exit(1); } while(0)
