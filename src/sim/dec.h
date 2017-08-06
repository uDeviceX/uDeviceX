bool solids0;

dump::Ticket dumpt;
basetags::TagGen tag_gen;

namespace o /* s[o]lvent */
{
flu::Quants       q;
flu::TicketZ     tz; /* [z]ip             */
flu::TicketRND trnd; /* random            */
odstr::TicketD   td; /* [d]istribute      */
odstr::TicketU   tu; /* [u]npack          */
odstr::Work       w; /* work              */

flu::QuantsI     qi; /* global [i]ds      */
odstr::TicketI   ti; /* global [i]ds      */
odstr::TicketUI tui; /* [u]npack [i]ds    */

flu::QuantsI     qt; /* [t]ags            */
odstr::TicketI   tt; /* [t]ags            */
odstr::TicketUI tut; /* [u]npack [t]ags   */

/* [h]alo interactions */
namespace h {
dpdr::TicketCom tc;
dpdr::TicketRnd trnd;
dpdr::TicketShalo ts;
dpdr::TicketRhalo tr;

/* optional: flags */
dpdr::TicketICom tic;
dpdr::TicketSIhalo tsi;
dpdr::TicketRIhalo tri;
}

Force *ff;
Force  ff_hst[MAX_PART_NUM]; /* solvent forces on host    */
}

namespace r /* [r]bc */
{
rbc::Quants q;
rbc::TicketT tt;

/* [d]istribute [t]ickets */
rdstr::TicketC tdc;
rdstr::TicketP tdp;
rdstr::TicketE tde;
rdstr::TicketS tds;
rdstr::TicketR tdr;

Force     *ff;
}

namespace s /* rigid bodies */
{
rig::Quants q;
rig::TicketBB t;
scan::Work ws; /* work for scan */
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

namespace mc /* mesh communication */
{
mcomm::TicketCom tc;
mcomm::TicketM   tm;
mcomm::TicketS   ts;
mcomm::TicketR   tr;
}

namespace bb /* bounce back */
{
tcells::Quants qtc;
mbounce::TicketM tm;
}

/* defined in 0dev/ and 0hst/ */
void distr_solid();
void update_solid0();
void bounce_solid(int);
