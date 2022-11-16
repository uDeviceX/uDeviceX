// tag::interface[]
struct RNDunif;                                             // <1>
void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **r); // <2>
void rnd_fin(RNDunif *r);                                   // <3>
float rnd_get(RNDunif *r);                                  // <4>
// end::interface[]
