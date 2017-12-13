void ini_none(/**/ BForce_v *p) {
    p->type = BODY_FORCE_V_NONE;
}

void ini(BForce_cste_v par, /**/ BForce_v *p) {
    p->type = BODY_FORCE_V_CSTE;
    p->p.cste = par;
}

void ini(BForce_dp_v par, /**/ BForce_v *p) {
    p->type = BODY_FORCE_V_DP;
    p->p.dp = par;
}

void ini(BForce_shear_v par, /**/ BForce_v *p) {
    p->type = BODY_FORCE_V_SHEAR;
    p->p.shear = par;
}

void ini(BForce_rol_v par, /**/ BForce_v *p) {
    p->type = BODY_FORCE_V_ROL;
    p->p.rol = par;
}

void ini(BForce_rad_v par, /**/ BForce_v *p) {
    p->type = BODY_FORCE_V_RAD;
    p->p.rad = par;
}
