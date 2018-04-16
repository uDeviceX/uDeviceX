#ifdef FORCE_KANTOR1_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace force_kantor1_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_kantor1_dev {
  #define END }
#endif

BEGIN

#ifdef FORCE_KANTOR1_HOST
_S_ double rsqrt0(double x) { return pow(x, -0.5); }
#define PRINT(fmt, ...) msg_print((fmt), ##__VA_ARGS__)
#define EXIT() ERR("assert")
#else
_S_ double rsqrt0(double x) { return rsqrt(x); }
#define PRINT(fmt, ...) printf((fmt), ##__VA_ARGS__)
#define EXIT() assert(0)
#endif

_S_ void dih_a0(int Flag_a, double ax, double ay, double az,
                double bx, double by, double bz,
                double cx, double cy, double cz,
                double dx, double dy, double dz,
                double *pfx, double *pfy, double *pfz) {
    double E_ax,E_ay,E_az,E_bx,E_by,E_bz,E_kk,E_kn,E_kx,
        E_ky,E_kz,E_nn,E_nx,E_ny,E_nz,E_rsq,kk,
        kk_kx,kk_ky,kk_kz,kn_kx,kn_ky,kn_kz,kn_nx,kn_ny,kn_nz,kx,kx_ay,kx_az,kx_by,
        kx_bz,ky,ky_ax,ky_az,ky_bx,ky_bz,kz,kz_ax,kz_ay,
        kz_bx,kz_by,nn,nn_nx,nn_ny,nn_nz,nx,nx_by,nx_bz,
        ny,ny_bx,ny_bz,nz,nz_bx,nz_by,rsq_kk,rsq_nn,rsq,kn;
    kz = (bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
    ky = (bz-az)*(cx-ax)-(bx-ax)*(cz-az);
    kx = (by-ay)*(cz-az)-(bz-az)*(cy-ay);
    nz = (cy-by)*(dx-bx)-(cx-bx)*(dy-by);
    ny = (cx-bx)*(dz-bz)-(cz-bz)*(dx-bx);
    nx = (cz-bz)*(dy-by)-(cy-by)*(dz-bz);
    nn = pow(nz,2)+pow(ny,2)+pow(nx,2);
    kk = pow(kz,2)+pow(ky,2)+pow(kx,2);
    PRINT("kk,nn: %g %g\n", sqrt(kk), sqrt(nn));
    rsq = 1/sqrt(kk*nn);
    kn = kz*nz+ky*ny+kx*nx;
    E_kn = rsq;
    E_rsq = kn;
    kn_kx = nx;
    kn_ky = ny;
    kn_kz = nz;
    kn_nx = kx;
    kn_ny = ky;
    kn_nz = kz;
    E_kx = E_kn*kn_kx;
    E_ky = E_kn*kn_ky;
    E_kz = E_kn*kn_kz;
    E_nx = E_kn*kn_nx;
    E_ny = E_kn*kn_ny;
    E_nz = E_kn*kn_nz;
    rsq_kk = -(nn*pow(kk*nn,(-3.0E+0)/2.0E+0))/2.0E+0;
    rsq_nn = -(kk*pow(kk*nn,(-3.0E+0)/2.0E+0))/2.0E+0;
    E_kk = E_rsq*rsq_kk;
    E_nn = E_rsq*rsq_nn;
    kk_kx = 2*kx;
    kk_ky = 2*ky;
    kk_kz = 2*kz;
    E_kx = E_kk*kk_kx;
    E_ky = E_kk*kk_ky;
    E_kz = E_kk*kk_kz;
    nn_nx = 2*nx;
    nn_ny = 2*ny;
    nn_nz = 2*nz;
    E_nx = E_nn*nn_nx;
    E_ny = E_nn*nn_ny;
    E_nz = E_nn*nn_nz;
    nx_by = dz-cz;
    nx_bz = cy-dy;
    E_by = E_nx*nx_by;
    E_bz = E_nx*nx_bz;
    ny_bx = cz-dz;
    ny_bz = dx-cx;
    E_bx = E_ny*ny_bx;
    E_bz = E_ny*ny_bz;
    nz_bx = dy-cy;
    nz_by = cx-dx;
    E_bx = E_nz*nz_bx;
    E_by = E_nz*nz_by;
    kx_ay = bz-cz;
    kx_az = cy-by;
    kx_by = cz-az;
    kx_bz = ay-cy;
    E_ay = E_kx*kx_ay;
    E_az = E_kx*kx_az;
    E_by = E_kx*kx_by;
    E_bz = E_kx*kx_bz;
    ky_ax = cz-bz;
    ky_az = bx-cx;
    ky_bx = az-cz;
    ky_bz = cx-ax;
    E_ax = E_ky*ky_ax;
    E_az = E_ky*ky_az;
    E_bx = E_ky*ky_bx;
    E_bz = E_ky*ky_bz;
    kz_ax = by-cy;
    kz_ay = cx-bx;
    kz_bx = cy-ay;
    kz_by = ax-cx;
    E_ax = E_kz*kz_ax;
    E_ay = E_kz*kz_ay;
    E_bx = E_kz*kz_bx;
    E_by = E_kz*kz_by;

    if (Flag_a) {
        *pfx = E_ax; *pfy = E_ay; *pfz = E_az;
    } else {
        *pfx = E_bx; *pfy = E_by; *pfz = E_bz;
    }
}

_I_ double3 dih_a(double phi, double kb,
                  double3 a, double3 b, double3 c, double3 d) {
    double3 f;
    double fx, fy, fz;
    int Flag_a;
    Flag_a = 1;
    dih_a0(Flag_a,
           a.x, a.y, a.z, b.x, b.y, b.z,
           c.x, c.y, c.z, d.x, d.y, d.z,
           &fx, &fy, &fz);
    f.x = kb*fx; f.y = kb*fy; f.z = kb*fz;
    return f;
}

_I_ double3 dih_b(double phi, double kb,
                  double3 a, double3 b, double3 c, double3 d) {
    double3 f;
    double fx, fy, fz;
    int Flag_a;
    Flag_a = 0;
    dih_a0(Flag_a,
           a.x, a.y, a.z, b.x, b.y, b.z,
           c.x, c.y, c.z, d.x, d.y, d.z,
           &fx, &fy, &fz);
    f.x = kb*fx; f.y = kb*fy; f.z = kb*fz;
    return f;
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
