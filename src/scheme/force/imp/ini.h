void ini_none(/**/ BForce *p) {
    p->type = BODY_FORCE_NONE;
}

void ini(BForce_cste par, /**/ BForce *p) {
    p->type = BODY_FORCE_CSTE;
    p->p.cste = par;
}

void ini(BForce_dp par, /**/ BForce *p) {
    p->type = BODY_FORCE_DP;
    p->p.dp = par;
}

void ini(BForce_shear par, /**/ BForce *p) {
    p->type = BODY_FORCE_SHEAR;
    p->p.shear = par;
}

void ini(BForce_rol par, /**/ BForce *p) {
    p->type = BODY_FORCE_ROL;
    p->p.rol = par;
}

void ini(BForce_rad par, /**/ BForce *p) {
    p->type = BODY_FORCE_RAD;
    p->p.rad = par;
}
