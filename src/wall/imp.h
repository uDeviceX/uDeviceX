namespace wall {
struct Quants {
    float4 *pp;
    int n;
};

struct Ticket {
    rnd::KISS *rnd;
    clist::Clist cells;
    clist::Map mcells;
    Texo<int> texstart;
    Texo<float4> texpp;
};

namespace grey {
void force(const sdf::Quants qsdf, const Quants q, const Ticket t, Cloud cloud, const int n, Force *ff);
}

namespace color {
void force(const sdf::Quants qsdf, const Quants q, const Ticket t, Cloud cloud, const int n, Force *ff);
}

void alloc_quants(Quants *q);
void alloc_ticket(Ticket *t);

void free_quants(Quants *q);
void free_ticket(Ticket *t);

void gen_quants(int maxn, const sdf::Quants qsdf, /**/ int *n, Particle* pp, Quants *q);
void strt_quants(int maxn, Quants *q);

void gen_ticket(const Quants q, Ticket *t);

void strt_dump_templ(const Quants q);

}
