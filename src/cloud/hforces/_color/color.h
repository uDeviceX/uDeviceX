namespace hforces {
namespace dev {

inline void cloud_get_color(Cloud c, int i, /**/ forces::Pa *p) {
    int c = c.cc[i];
    forces::c2p(c, /**/ p);
}

} // dev
} // hforces
