void bforce_ini(BForce **p) { UC(emalloc(sizeof(BForce), (void**) p));}
void bforce_fin(BForce *p)  {UC(efree(p));}

void bforce_set_none(/**/ BForce *p) {
    p->type = BODY_FORCE_NONE;
}

void bforce_set_cste(float3 f, /**/ BForce *p) {
    p->type = BODY_FORCE_CSTE;
    p->p.cste.a = f;
}

void bforce_set_dp(float a, /**/ BForce *p) {
    p->type = BODY_FORCE_DP;
    p->p.dp.a = a;
}

void bforce_set_shear(float a, /**/ BForce *p) {
    p->type = BODY_FORCE_SHEAR;
    p->p.shear.a = a;
}

void bforce_set_rol(float a, /**/ BForce *p) {
    p->type = BODY_FORCE_ROL;
    p->p.rol.a = a;
}

void bforce_set_rad(float a, /**/ BForce *p) {
    p->type = BODY_FORCE_RAD;
    p->p.rad.a = a;
}
