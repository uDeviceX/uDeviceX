enum {
    TYPE_NONE,
    TYPE_CSTE,
    TYPE_DP,
    TYPE_SHEAR,
    TYPE_ROL
};

void body_force(Coords c, float mass, FParam fpar, int n, const Particle *pp, /**/ Force *ff) {
    int type;
    FParam_d p;
    type = fpar.type;
    p    = fpar.dev;

    switch (type) {
    case TYPE_NONE:
        break;
    case TYPE_CSTE:
        break;
    case TYPE_DP:
        break;
    default:
        break;
    };
}
