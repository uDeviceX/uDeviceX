namespace area_volume {
struct Ticket;
void dev(int nt, int nv, int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *av);
void hst(int nt, int nv, int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *hst);

void ini(int nmax, Particle*, int nt, int4 *tri, /**/ Ticket**);
void fin(Ticket*);
}
