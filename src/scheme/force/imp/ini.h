void bforce_ini(BForce **p) { UC(emalloc(sizeof(BForce), (void**) p));}
void bforce_fin(BForce *p)  {UC(efree(p));}


void bforce_ini_none(/**/ BForce *p) {
    p->type = BODY_FORCE_NONE;
}

void bforce_ini(BForce_cste par, /**/ BForce *p) {
    p->type = BODY_FORCE_CSTE;
    p->p.cste = par;
}

void bforce_ini(BForce_dp par, /**/ BForce *p) {
    p->type = BODY_FORCE_DP;
    p->p.dp = par;
}

void bforce_ini(BForce_shear par, /**/ BForce *p) {
    p->type = BODY_FORCE_SHEAR;
    p->p.shear = par;
}

void bforce_ini(BForce_rol par, /**/ BForce *p) {
    p->type = BODY_FORCE_ROL;
    p->p.rol = par;
}

void bforce_ini(BForce_rad par, /**/ BForce *p) {
    p->type = BODY_FORCE_RAD;
    p->p.rad = par;
}
