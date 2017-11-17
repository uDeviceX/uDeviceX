namespace g { /* global */
enum {B = BLUE_COLOR, R = RED_COLOR, S = SOLID_COLOR, W = WALL_COLOR};
__constant__ static float gg[N_COLOR][N_COLOR] = {
    {gdpd_b ,  gdpd_br,  gdpd_bs, gdpd_bw},
    {gdpd_br,  gdpd_r ,  gdpd_rs, gdpd_rw},
    {gdpd_bs,  gdpd_rs,  -99999 , gdpd_sw},
    {gdpd_bw,  gdpd_rw,  gdpd_sw, -99999 }
};
__constant__ static float aa[N_COLOR][N_COLOR] = {
    {adpd_b ,  adpd_br,  adpd_bs, adpd_bw},
    {adpd_br,  adpd_r ,  adpd_rs, adpd_rw},
    {adpd_bs,  adpd_rs,  -99999 , adpd_sw},
    {adpd_bw,  adpd_rw,  adpd_sw, -99999 }
};
}
static __device__ void color2par(int ca, int cb, /**/ DPDparam *p) {
    using namespace g;
    if         (!multi_solvent) {
        p->gamma = gg[B][B];
        p->a     = aa[B][B];
    } else {
        p->gamma = gg[ca][cb];
        p->a     = aa[ca][cb];
    }
}
