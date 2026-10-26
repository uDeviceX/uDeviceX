// Grouped editable compilation unit for RBC force and bending force code.
#include "udx.rbc-force-all.cu"

/* The half-edge implementation is owned by the I/O unit. */
struct HeRead;
struct He;
int he_read_int4_ini(int, int, const int4 *, HeRead **);
int he_read_fin(HeRead *);
int he_ini(HeRead *, He **);
int he_fin(He *);
int he_ne(He *);
int he_nxt(He *, int);
int he_flp(He *, int);
int he_ver(He *, int);
int he_hdg_edg(He *, int);

/* Force units provide the shared infrastructure used by bending units. */
#define UDX_RBC_BENDING_COMMON
#include "udx.rbc-force-bending-all.cu"
