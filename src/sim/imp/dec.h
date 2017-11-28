bool solids0;

bop::Ticket dumpt;
basetags::TagGen tag_gen;

Flu flu;
Rbc rbc;
Rig rig;

BounceBack bb;
Colorer colorer;
PidVCont vcont;

/*** see int/wall.h ***/
namespace w {
sdf::Quants qsdf;
wall::Quants q;
wall::Ticket t;
}
/***  ***/

namespace a /* all */
{
Particle *pp_hst; /* particles on host */
}

namespace rs /* objects */
{
ObjExch e;
cnt::Contact c;
}


