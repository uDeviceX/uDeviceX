namespace rbc {
namespace sub {
void setup(const char *r_templ, int *faces, int4 *tri, int *adj0, int *adj1);
void setup_from_strt(const int id, /**/ Particle *pp, int *nc, int *n, /*w*/ Particle *pp_hst);
void setup_textures(int4 *tri, Texo<int4> *textri, int *adj0, Texo<int> *texadj0,
                    int *adj1, Texo<int> *texadj1, Particle *pp, Texo<float2> *texvert);
void forces(int nc, const Texo<float2> texvert, const Texo<int4> textri, const Texo<int> texadj0, const Texo<int> texadj1, Force *ff, float* av);
void strt_dump(const int id, const int n, const Particle *pp, /*w*/ Particle *pp_hst);
}

namespace ic {
void setup_from_pos(const char *r_templ, const char *r_state, int nv, /**/ Particle *pp, int *nc, int *n, /* storage */ Particle *pp_hst);
}

}
