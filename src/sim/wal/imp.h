struct Wall;

struct Config;
struct PairParams;
struct Coords;

void wall_ini(const Config*, int3 L, Wall**);
void wall_fin(Wall*);

void wall_interact(const Coords*, const PairParams*, Wall*, PFarrays*);
