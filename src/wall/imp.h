namespace wall {
void build_cells(const int n, float4 *pp4, clist::Clist *cells);

void gen_quants(sdf::Tex_t texsdf, /**/ int *o_n, Particle *o_pp, int *w_n, float4 **w_pp);
void strt_quants(int *w_n, float4 **w_pp);

void gen_ticket(const int w_n, float4 *w_pp, clist::Clist *cells, clist::Map *mcells, Texo<int> *texstart, Texo<float4> *texpp);

void strt_dump_templ(const int n, const float4 *pp);
}
