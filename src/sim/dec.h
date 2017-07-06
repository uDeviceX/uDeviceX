bool solids0;

H5FieldDump *dump_field;


namespace o /* s[o]lvent */
{
flu::Quants q;
flu::TicketZ tz;   /* [z]ip */
flu::TicketRND trnd;
odstr::TicketD td;   /* [d]istribute */
odstr::Work w;

/* [h]alo interactions */
namespace h {
dpdr::TicketCom tc;
dpdr::Ticketrnd trnd;
dpdr::TicketShalo ts;
dpdr::TicketRhalo tr;
}

Force    *ff;
Force     ff_hst[MAX_PART_NUM]; /* solvent forces on host    */
}

namespace r /* [r]bc */
{
rbc::Quants q;
rbc::TicketT tt;
Force     *ff;
}

namespace s /* rigid bodies */
{
rig::Quants q;
rig::TicketBB t;
Force *ff, *ff_hst;
}

/*** see int/wall.h ***/
namespace w {
sdf::Quants qsdf;
wall::Quants q;
wall::Ticket t;
}
/***  ***/

namespace a /* all */
{
Particle pp_hst[3*MAX_PART_NUM]; /* particles on host */
}

/* functions defined in 00dev/ and 00hst/ */
void distr_solid();
void update_solid0();
void bounce_solid(int);
