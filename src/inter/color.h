struct Coords;
namespace inter {
void color_hst(const Coords *coords, Particle *pp, int n, /**/ int *cc);
void color_dev(const Coords *coords, Particle *pp, int n, /*o*/ int *cc, /*w*/ Particle *pp_hst, int *cc_hst);
}
