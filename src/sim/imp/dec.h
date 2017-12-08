bool solids0;

bop::Ticket dumpt;

Coords coords;

Flu flu;
Rbc rbc;
Rig rig;
Wall wall;

ObjInter objinter;
BounceBack bb;
Colorer colorer;
PidVCont vcont;
Outflow *outflow;
Inflow *inflow;

DCont    *denoutflow;
DContMap *mapoutflow;

namespace a /* all */
{
Particle *pp_hst; /* particles on host */
}

