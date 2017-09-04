namespace hforces {
namespace dev {
#error test
inline __device__ void cloud_get_color(Cloud c, int i, /**/ forces::Pa *p) {
    int color = c.cc[i];
    forces::c2p(color, /**/ p);
}

} // dev
} // hforces
