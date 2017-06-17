namespace sim
{
bool solids0;

H5FieldDump *dump_field;


namespace o /* s[o]lvent */
{

int       n; /* Quants for sol:: */
Particle *pp;
Clist *cells;

sol::TicketZ tz;
sol::TicketD td;
sol::Work w;

Force    *ff;

float4  *zip0; /* "zipped" version of Solvent array */
ushort4 *zip1;

Particle *pp0; /* Solvent (temporal buffer) */

Particle  pp_hst[MAX_PART_NUM]; /* solvent on host           */
Force     ff_hst[MAX_PART_NUM]; /* solvent forces on host    */
}

namespace r /* [r]bc */
{
int n = 0, nc = 0, nt = RBCnt, nv = RBCnv;
Particle *pp;
Force    *ff;

Particle pp_hst[MAX_PART_NUM];
int faces[MAX_FACE_NUM];
float *av;
}

/*** see int/wall.h ***/
namespace w {
wall::Quants q;
wall::Ticket t;
}
/***  ***/

namespace a /* all */
{
Particle pp_hst[3*MAX_PART_NUM]; /* particles on host */
}
}

/* functions defined in dev/ and hst/ */
void distr_solid();
void update_solid0();
void bounce_solid(int);
