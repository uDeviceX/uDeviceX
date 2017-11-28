bool solids0;

bop::Ticket dumpt;
basetags::TagGen tag_gen;

/* solvent */
Flu flu;
Rbc rbc;

namespace o /* s[o]lvent */
{
/* velocity controller */
PidVCont vcont;
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
