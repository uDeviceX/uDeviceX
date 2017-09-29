bool solids0;

bop::Ticket dumpt;
basetags::TagGen tag_gen;

namespace o /* s[o]lvent */
{
flu::Quants       q;
flu::TicketZ     tz; /* [z]ip             */
flu::TicketRND trnd; /* random            */

flu::QuantsI     qi; /* global [i]ds      */
flu::QuantsI     qc; /* [c]olors          */

Distr d;

/* [h]alo interactions : local halos : see type.h */
H h;

Force *ff;
Force  ff_hst[MAX_PART_NUM]; /* solvent forces on host    */
}

namespace r /* [r]bc */
{
rbc::Quants q;
rbc::TicketT tt;

RbcDistr d;

Force     *ff;
}

namespace s /* rigid bodies */
{
rig::Quants q;
rig::TicketBB t;
scan::Work ws; /* work for scan */
Force *ff, *ff_hst;

RigDistr d;
BBexch e;
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

namespace rs /* objects */
{
Sexch e;
}

namespace mc /* mesh communication */
{
Mexch e;
Particle *pp;
}

namespace bb /* bounce back */
{
tcells::Quants qtc;
mbounce::TicketM tm;

meshbb::BBdata bbd;
Momentum *mm;
}

