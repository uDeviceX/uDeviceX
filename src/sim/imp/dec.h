bool solids0;

bop::Ticket dumpt;
basetags::TagGen tag_gen;


Flu flu;

namespace o /* s[o]lvent */
{
/* velocity controller */
PidVCont vcont;
}

namespace r /* [r]bc */
{
rbc::Quants q;
rbc::force::TicketT tt;

RbcDistr d;

Force     *ff;

rbc::com::Helper  com;      /* helper to compute center of masses */
rbc::stretch::Fo *stretch;  /* helper to apply stretching [fo]rce to cells */
}

namespace s /* rigid bodies */
{
rig::Quants q;
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
Objexch e;
cnt::Contact c;
}

Colorer colorer;

namespace bb /* bounce back */
{
meshbb::BBdata bbd;
Momentum *mm;
}
