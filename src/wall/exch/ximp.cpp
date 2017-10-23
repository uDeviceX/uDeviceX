
using namespace comm;

enum {
    LX = XS + 2 * WXM,
    LY = YS + 2 * WYM,
    LZ = ZS + 2 * WZM
};

static bool is_inside(int fid, const Particle p) {
    enum {X, Y, Z};
    const int d[3] = frag_i2d3(fid);
    float r[3] = {
        p.r[X] + d[X] * XS,
        p.r[Y] + d[Y] * YS,
        p.r[Z] + d[Z] * ZS
    };
    
    return
        r[X] >=  -0.5 * LX && r[X] < 0.5 * LX &&
        r[Y] >=  -0.5 * LY && r[Y] < 0.5 * LY &&
        r[Z] >=  -0.5 * LZ && r[Z] < 0.5 * LZ;
}

static void fill_bags(int n, const Particle *pp, hBags *b) {
    int i, j, *cc, c;
    Particle p, **dst;

    cc  = b->counts;
    dst = b->data;

    memset(cc, 0, NBAGS * sizeof(int));
    
    for (i = 0; i < n; ++i) {
        p = pp[i];
        for (j = 0; j < NFRAGS; ++j) {
            if (is_inside(i, p)) {
                c = cc[i] ++;
                dst[i][c] = p;
            }
        }
    }
}

static void communicate(const hBags *s, Stamp *c, hBags *r) {
    UC(post_send(s, c));
    post_recv(r, c);
    wait_send(c);
    UC(wait_recv(c, /**/ r));
}

static int unpack(int maxn, const hBags *b, /*io*/ int *n, Particle *pp) {
    int i, j, k, c;
    const Particle *src;
    k = *n;
    for (j = 0; j < NFRAGS; ++j) {
        c = b->counts[j];
        src = b->data[j];
        for (i = 0; i < c; ++i) {
            pp[k] = src[i];
            ++k;
        }
    }
    *n = k;
}

/* exchange pp(hst) between processors to get a wall margin */
void exch(int maxn, /*io*/ Particle *pp, int *n) {
    hBags send, recv;
    Stamp stamp;
    basetags::TagGen tg;
    int i, capacity[NBAGS];

    for (i = 0; i < NBAGS; ++i) capacity[i] = maxn;
    ini(HST_ONLY, NONE, sizeof(Particle), capacity, &send, NULL);
    ini(HST_ONLY, NONE, sizeof(Particle), capacity, &recv, NULL);
    ini(&tg);
    ini(m::cart, &tg, &stamp);

    fill_bags(*n, pp, /**/ &send);
    communicate(&send, /**/ &stamp, &recv);
    unpack(maxn, &recv, /**/ n, pp);
    
    fin(HST_ONLY, NONE, &send, NULL);
    fin(HST_ONLY, NONE, &recv, NULL);
    fin(&stamp);
}
