bool solids0;

bop::Ticket dumpt;

Coords coords;

Wall wall;

ObjInter objinter;
BounceBack bb;
Colorer colorer;
PidVCont vcont;
Outflow *outflow;
Inflow *inflow;

DCont    *denoutflow;
DContMap *mapoutflow;

Config *config;

namespace a /* all */
{
Particle *pp_hst; /* particles on host */
}

