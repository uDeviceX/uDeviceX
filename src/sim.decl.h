namespace sim
{
bool solids0;

H5FieldDump *dump_field;


namespace o /* s[o]lvent */
{

int       n; /* Quants for flu:: */
Particle *pp;
Clist *cells;

flu::TicketZ tz;
flu::TicketD td;
flu::Work w;
Force    *ff;

Particle  pp_hst[MAX_PART_NUM]; /* solvent on host           */
Force     ff_hst[MAX_PART_NUM]; /* solvent forces on host    */
}

namespace r /* [r]bc */
{
rbc::Quants q;
rbc::TicketT tt;
Force     *ff;
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
