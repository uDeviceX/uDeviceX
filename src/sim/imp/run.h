void run_eq(long te) { /* equilibrate */
    BForce bforce;
    ini_none(/**/ &bforce);    
    bool wall0 = false;
    for (long it = 0; it < te; ++it) step(&bforce, wall0, it);
    UC(distribute_flu(/**/ &flu));
}

void run(long ts, long te) {
    long it; /* current timestep */
    dump_strt_templ(coords, &wall); /* :TODO: is it the right place? */

    BForce bforce;
    // TODO
#if   defined(FORCE_NONE)
    ini_none(/**/ &bforce);
#elif defined(FORCE_CONSTANT)
    BForce_cste par;
    float ex, ey, ez;
    os::env2float_d("FORCE_PAR_EX", 1, &ex);
    os::env2float_d("FORCE_PAR_EY", 0, &ey);
    os::env2float_d("FORCE_PAR_EZ", 0, &ez);
    par.a = make_float3(FORCE_PAR_A*ex, FORCE_PAR_A*ey, FORCE_PAR_A*ez);
    UC(ini(par, /**/ &bforce));
#elif defined(FORCE_DOUBLE_POISEUILLE)
    BForce_dp par;
    par.a = FORCE_PAR_A;
    UC(ini(par, /**/ &bforce));
#elif defined(FORCE_SHEAR)
    BForce_shear par;
    par.a = FORCE_PAR_A;
    UC(ini(par, /**/ &bforce));
#elif defined(FORCE_4ROLLER)
    BForce_rol par;
    par.a = FORCE_PAR_A;
    UC(ini(par, /**/ &bforce));
#elif defined(FORCE_RADIAL)
    BForce_rad par;
    par.a = FORCE_PAR_A;
    UC(ini(par, /**/ &bforce));
#else
#error FORCE_* is undefined
#endif
    
    /* ts, te: time start and end */
    for (it = ts; it < te; ++it) {
        step2params(it - ts, &wall.vel, /**/ &wall.vview);
        step(&bforce, walls, it);
    }
    UC(distribute_flu(/**/ &flu));
}
