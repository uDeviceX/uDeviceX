void run_eq(long te, Sim *s) { /* equilibrate */
    BForce bforce;
    s->equilibrating = true;
    
    ini_none(/**/ &bforce);    
    bool wall0 = false;
    for (long it = 0; it < te; ++it) step(&bforce, wall0, it, s);
    UC(distribute_flu(/**/ s));
}

// TODO
static void ini_bforce(BForce *bforce) {
#if   defined(FORCE_NONE)
    ini_none(/**/ bforce);
#elif defined(FORCE_CONSTANT)
    BForce_cste par;
    float ex, ey, ez;
    os::env2float_d("FORCE_PAR_EX", 1, &ex);
    os::env2float_d("FORCE_PAR_EY", 0, &ey);
    os::env2float_d("FORCE_PAR_EZ", 0, &ez);
    par.a = make_float3(FORCE_PAR_A*ex, FORCE_PAR_A*ey, FORCE_PAR_A*ez);
    UC(ini(par, /**/ bforce));
#elif defined(FORCE_DOUBLE_POISEUILLE)
    BForce_dp par;
    par.a = FORCE_PAR_A;
    UC(ini(par, /**/ bforce));
#elif defined(FORCE_SHEAR)
    BForce_shear par;
    par.a = FORCE_PAR_A;
    UC(ini(par, /**/ bforce));
#elif defined(FORCE_4ROLLER)
    BForce_rol par;
    par.a = FORCE_PAR_A;
    UC(ini(par, /**/ bforce));
#elif defined(FORCE_RADIAL)
    BForce_rad par;
    par.a = FORCE_PAR_A;
    UC(ini(par, /**/ bforce));
#else
#error FORCE_* is undefined
#endif
}

void run(long ts, long te, Sim *s) {
    long it; /* current timestep */
    Wall *wall = &s->wall;

    s->equilibrating = false;

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */

    BForce bforce;
    ini_bforce(&bforce);
    
    /* ts, te: time start and end */
    for (it = ts; it < te; ++it) {
        step2params(it - ts, &wall->vel, /**/ &wall->vview);
        step(&bforce, walls, it, s);
    }
    UC(distribute_flu(/**/ s));
}
