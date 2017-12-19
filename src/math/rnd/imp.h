struct RNDunif;

void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **r);
void rnd_fin(RNDunif *r);

float get(const RNDunif *r);
