namespace hforces {
namespace dev {

inline void cloud_get_color(Cloud c, int i, /**/ forces::Pa *p) {
    int color = c.cc[i];
    forces::c2p(color, /**/ p);
}

} // dev
} // hforces
