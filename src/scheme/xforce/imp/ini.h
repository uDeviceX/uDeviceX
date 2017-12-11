void ini_none(/**/ FParam *p) {
    p->type = TYPE_NONE;
}

void ini(FParam_cste_d par, /**/ FParam *p) {
    p->type = TYPE_CSTE;
    p->dev.cste = par;
}

void ini(FParam_dp_d par, /**/ FParam *p) {
    p->type = TYPE_DP;
    p->dev.dp = par;
}

void ini(FParam_shear_d par, /**/ FParam *p) {
    p->type = TYPE_SHEAR;
    p->dev.shear = par;
}

void ini(FParam_rol_d par, /**/ FParam *p) {
    p->type = TYPE_ROL;
    p->dev.rol = par;
}
