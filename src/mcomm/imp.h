/* Structure containing all kinds of requests a fragment can have */
struct Reqs {
    MPI_Request pp[26], counts[26];
};

void cancel_req(Reqs *r) {
    for (int i = 0; i < 26; ++i) {
        MC(MPI_Cancel(r->pp     + i));
        MC(MPI_Cancel(r->counts + i));
    }
}

void wait_req(Reqs *r) {
    MPI_Status ss[26];
    MC(l::m::Waitall(26, r->pp,     ss));
    MC(l::m::Waitall(26, r->counts, ss));
}

enum {X, Y, Z};
enum {BULK, FACE, EDGE, CORN};

#define i2del(i) {((i) + 1) % 3 - 1,            \
            ((i) / 3 + 1) % 3 - 1,              \
            ((i) / 9 + 1) % 3 - 1}

static inline int dc2vc(const int d) {return (3 + d) % 3;}

static int d2i(const int d[3]) {
    return dc2vc(d[X]) + 3 * (dc2vc(d[Y]) + 3 * dc2vc(d[Z]));
}

namespace travel {
void faces(const int i, const int d[3], /**/ std::vector<int> travellers[27]) {
    for (int c = 0; c < 3; ++c) {
        if (d[c]) {
            int df[3] = {0, 0, 0}; df[c] = d[c];
            int code = d2i(df);
            travellers[code].push_back(i);
        }
    }
}

void edges(const int i, const int d[3], /**/ std::vector<int> travellers[27]) {
    for (int c = 0; c < 3; ++c) {
        int de[3] = {d[X], d[Y], d[Z]}; de[c] = 0;
        if (de[(c + 1) % 3] && de[(c + 2) % 3]) {
            int code = d2i(de);
            travellers[code].push_back(i);
        }
    }
}

void cornr(const int i, const int d[3], /**/ std::vector<int> travellers[27]) {
    assert(d[X] && d[Y] && d[Z]);
    int code = d2i(d);
    travellers[code].push_back(i);
}
}

void pack(const float3* minext_hst, const float3 *maxext_hst, const Particle *pp, const int nv, const int nm, /**/ Particle *spp[27]) {
    std::vector<int> travellers[27];
    int d[3];

    for (int i = 0; i < nm; ++i) {
        const float3 lo = minext_hst[i];
        const float3 hi = maxext_hst[i];

        d[X] = -1 + (lo.x > -XS/2) + (hi.x > XS/2);
        d[Y] = -1 + (lo.y > -YS/2) + (hi.y > YS/2);
        d[Z] = -1 + (lo.z > -ZS/2) + (hi.z > ZS/2);

        const int type = fabs(d[X]) + fabs(d[Y]) + fabs(d[Z]);

        if (type >= BULK) travellers[0].push_back(i);
        if (type >= FACE) travel::faces(i, d, /**/ travellers);
        if (type >= EDGE) travel::edges(i, d, /**/ travellers);
        if (type >= CORN) travel::cornr(i, d, /**/ travellers);
    }

    /* copy data */
    for (int i = 0; i < 27; ++i) {
        int s = 0; /* start */

        for (int id : travellers[i])
        CC(cudaMemcpyAsync(spp[i] + nv * (s++), pp + nv * id, nv * sizeof(Particle), D2H));
    }
}
