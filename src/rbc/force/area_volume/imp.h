namespace area_volume {
struct Ticket;
void main(int nt, int nv, int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *av);
void main_hst(int nt, int nv, int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *hst);
}
