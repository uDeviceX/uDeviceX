struct IoRig;
struct Coords;
struct Solid;

void io_rig_ini(IoRig**);
void io_rig_fin(IoRig*);

void io_rig_dump(const Coords *c, int ns, float t, const Solid *ss, const Solid *ssbb);
