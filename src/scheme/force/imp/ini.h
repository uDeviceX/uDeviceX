void ini_none(/**/ BForce *p) {
    p->type = BODY_FORCE_V_NONE;
}

void ini(BForce_cste_v par, /**/ BForce *p) {
    p->type = BODY_FORCE_V_CSTE;
    p->dev.cste = par;
}

void ini(BForce_dp_v par, /**/ BForce *p) {
    p->type = BODY_FORCE_V_DP;
    p->dev.dp = par;
}

void ini(BForce_shear_v par, /**/ BForce *p) {
    p->type = BODY_FORCE_V_SHEAR;
    p->dev.shear = par;
}

void ini(BForce_rol_v par, /**/ BForce *p) {
    p->type = BODY_FORCE_V_ROL;
    p->dev.rol = par;
}

void ini(BForce_rad_v par, /**/ BForce *p) {
    p->type = BODY_FORCE_V_RAD;
    p->dev.rad = par;
}
