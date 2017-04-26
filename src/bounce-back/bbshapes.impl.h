namespace bbshapes {

    enum {X, Y, Z};

    /*
      return true if a root h lies in [0, dt] (output h), false otherwise
      if 2 roots in [0, dt], smallest root h0 is returned (first collision)
    */
    _DH_ bool robust_quadratic_roots(const float a, const float b, const float c, /**/ float *h)
    {
        float D, h0, h1;
        int sgnb;

        sgnb = b > 0 ? 1 : -1;
        D = b*b - 4*a*c;

        if (D < 0) return false;
        
        h0 = (-b - sgnb * sqrt(D)) / (2 * a);
        h1 = c / (a * h0);
        
        if (h0 > h1)
        {
            float htmp = h1;
            h1 = h0; h0 = htmp;
        }

        if (h0 >= 0 && h0 <= dt) {*h = h0; return true;}
        if (h1 >= 0 && h1 <= dt) {*h = h1; return true;}
        
        return false;
    }

    
#if defined(rsph)

#define shape sphere

    namespace sphere
    {
#define rsph_bb rsph

        _DH_ bool inside(const float *r)
        {
            return r[X] * r[X] + r[Y] * r[Y] + r[Z] * r[Z] < rsph_bb * rsph_bb;
        }

        _DH_ bool intersect(const float *r0, const float *v0, const float *vcm, const float *om0, /**/ float *h)
        {
            float r0x = r0[X] + dt * vcm[X], r0y = r0[Y] + dt * vcm[Y], r0z = r0[Z] + dt * vcm[Z];
            float v0x = v0[X] - vcm[X],      v0y = v0[Y] - vcm[Y],      v0z = v0[Z] - vcm[Z];
                        
            const float a = v0x*v0x + v0y*v0y + v0z*v0z;
            
            const float b = 2 * (r0x * v0x + r0y * v0y + r0z * v0z);
            const float c = r0x * r0x + r0y * r0y + r0z * r0z - rsph_bb * rsph_bb;
        
            return robust_quadratic_roots(a, b, c, h);
        }

        _DH_ void rescue(float *r)
        {
            float scale = (rsph_bb + 1e-6) / sqrt(r[X] * r[X] + r[Y] * r[Y] + r[Z] * r[Z]);

            r[X] *= scale;
            r[Y] *= scale;
            r[Z] *= scale;
        }
    }

#elif defined(thrbc)

#define shape rbcshape
#define pin_axis (true)
    
    namespace rbcshape
    {
        const float a0 = 0.0518, a1 = 2.0026, a2 = -4.491;
        const float D0 = 7.82;
                
        _DH_ bool inside(const float *r) {
            
            const float x = r[X] * thrbc, y = r[Y] * thrbc, z = r[Z] * thrbc;
               
            const float rho = (x*x+y*y)/(D0*D0);
            const float s = 1 - 4*rho;
            
            if (s < 0)
            return false;
            
            const float zrbc = D0 * sqrt(s) * (a0 + a1*rho + a2*rho*rho);
            
            return z > -zrbc && z < zrbc;
        }

        _DH_ inline float min2(float a, float b) {return a < b ? a : b;}
        _DH_ inline float max2(float a, float b) {return a < b ? b : a;}
        
        _DH_ bool intersect(const float *r0, const float *v0, const float *vcm, const float *om0, /**/ float *h)
        {
            const float r0x  = thrbc *  r0[X],           r0y  = thrbc * r0[Y],            r0z  = thrbc * r0[Z];
            const float v0x  = thrbc * (v0[X] - vcm[X]), v0y  = thrbc * (v0[Y] - vcm[Y]), v0z  = thrbc * (v0[Z] - vcm[Z]);
            const float om0x = om0[X],                   om0y = om0[Y],                   om0z = om0[Z];

            const float omcrx = om0y * r0z - om0z * r0y;
            const float omcry = om0z * r0x - om0x * r0z;
            const float omcrz = om0x * r0y - om0y * r0x;

            const float omcvx = om0y * v0z - om0z * v0y;
            const float omcvy = om0z * v0x - om0x * v0z;
            const float omcvz = om0x * v0y - om0y * v0x;
            
            // first guess
            float hn = dt*0.5, f = 1.f;
            
            // newton steps
            for (int step = 0; step < 10; ++step)
            {
                const float x = r0x + hn * (v0x - omcrx - hn * omcvx);
                const float y = r0y + hn * (v0y - omcry - hn * omcvy);
                const float z = r0z + hn * (v0z - omcrz - hn * omcvz);

                const float dxdt = v0x - omcrx - 2 * hn * omcvx;
                const float dydt = v0y - omcry - 2 * hn * omcvy;
                const float dzdt = v0z - omcrz - 2 * hn * omcvz;
                
                const float rho = (x*x + y*y) / (D0*D0);
                const float s = 1 - 4 * rho;
                const float subg = (a0 + a1*rho + a2*rho*rho);

                f = D0*D0 * s * subg * subg - z*z;

                const float dgdrho = 2 * (1 - 4*rho) * (a1 + 2*a2*rho) * subg - 4 * subg * subg;

                const float df = 2 * ((dxdt * x + dydt * y) * dgdrho - dzdt * z);

                //printf("%f %f %f\n", hn, f, df);
                
                hn = max2(0, min2(dt-1e-8, hn - f / df));
            }

            *h = hn;

            //printf("h = %f, f = %f\n", *h, f);
            
            return fabs(f) < 1e-4;
        }

        _DH_ void rescue(float *r)
        {
            const float x = r[X] * thrbc, y = r[Y] * thrbc;
            const float rho = (x*x + y*y) / (D0*D0);
            const float s = 1 - 4*rho;
            assert(s >= 0);
            
            const float zrbc = D0 * sqrt(s) * (a0 + a1*rho + a2*rho*rho);
            
            r[Z] = r[Z] < 0 ? -(zrbc + 1e-6) : zrbc + 1e-6;
            r[Z] /= thrbc;
        } 
    }

    
#elif defined(rcyl)

#define shape cylinder
    
    namespace cylinder
    {   
#define rcyl_bb rcyl

        _DH_ bool inside(const float *r)
        {
            return r[X] * r[X] + r[Y] * r[Y] < rcyl_bb * rcyl_bb;
        }

        /* output h between 0 and dt */
        _DH_ bool intersect(const float *r0, const float *v0, const float *vcm, const float *om0, /**/ float *h)
        {
            float r0x = r0[X],          r0y = r0[Y];
            float v0x = v0[X] - vcm[X], v0y = v0[Y] - vcm[Y];

            const float a = v0x * v0x + v0y * v0y;
            
            const float b = 2 * (r0x * v0x + r0y * v0y);
                        
            const float c = r0x * r0x + r0y * r0y - rcyl_bb * rcyl_bb;

            return robust_quadratic_roots(a, b, c, h);
        }

        _DH_ void rescue(float *r)
        {
            float scale = (rcyl_bb + 1e-6) / sqrt(r[X] * r[X] + r[Y] * r[Y]);

            r[X] *= scale;
            r[Y] *= scale;
        }
    }

#elif defined(a2_ellipse)

#define shape ellipse // "extruded" ellipse x^2/2^ + y^2/b^2 = 1
    
    namespace ellipse
    {
#define a2_bb a2_ellipse 
#define b2_bb b2_ellipse

        _DH_ bool inside(const float *r)
        {
            const float x = r[X];
            const float y = r[Y];
            
            return x*x / a2_bb + y*y / b2_bb < 1;
        }
        
        /* output h between 0 and dt */
        _DH_ bool intersect(const float *r0, const float *v0, const float *vcm, const float *om0, /**/ float *h)
        {
            const float r0x = r0[X],          r0y = r0[Y];
            const float v0x = v0[X] - vcm[X], v0y = v0[Y] - vcm[Y];

            const float om0z = -om0[Z];
            
            const float v0x_ = v0x - om0z * (r0y + dt * v0y);
            const float v0y_ = v0y + om0z * (r0x + dt * v0x);

            const float r0x_ = r0x + dt * om0z * (r0y + dt * v0y);
            const float r0y_ = r0y - dt * om0z * (r0x + dt * v0x);
            
            const float a = v0x_*v0x_ / a2_bb + v0y_*v0y_ / b2_bb;
            
            const float b = 2 * (r0x_ * v0x_ / a2_bb + r0y_ * v0y_ / b2_bb);
                        
            const float c = r0x_ * r0x_ / a2_bb + r0y_ * r0y_ / b2_bb - 1;

            return robust_quadratic_roots(a, b, c, h);
        }

        _DH_ void rescue(float *r) // cheap rescue
        {
            float scale = (1 + 1e-6) / sqrt(r[X] * r[X] / a2_bb + r[Y] * r[Y] / b2_bb);
            
            r[X] *= scale;
            r[Y] *= scale;
        }
    }

#elif defined(a2_ellipsoid)

#define shape ellipsoid
    
    namespace ellipsoid
    {
#define a2_bb a2_ellipsoid
#define b2_bb b2_ellipsoid
#define c2_bb c2_ellipsoid

        __DH__ bool inside(const float *r)
        {
            const float x = r[X];
            const float y = r[Y];
            const float z = r[Z];
            
            return x*x / a2_bb + y*y / b2_bb + z*z / c2_bb < 1;
        }

        _DH_ bool intersect(const float *r0, const float *v0, const float *vcm, const float *om0, /**/ float *h)
        {
            const float r0x  = r0[X],          r0   = r0[Y],          r0z  = r0[Z];
            const float v0x  = v0[X] - vcm[X], v0   = v0[Y] - vcm[Y], v0z  = v0[Z] - vcm[Z];
            const float om0x = om0[X],         om0y = om0[Y],         om0z = om0[Z];

            const float r1x = r0x + dt * v0x;
            const float r1y = r0y + dt * v0y;
            const float r1z = r0z + dt * v0z;
            
            const float v0x_ = v0x + om0y * r1z - om0z * r1y;
            const float v0y_ = v0y + om0z * r1x - om0x * r1z;
            const float v0z_ = v0z + om0x * r1y - om0y * r1x;

            const float r0x_ = r0x - dt * (om0z * r1z - om0z * r1y);
            const float r0y_ = r0y - dt * (om0z * r1x - om0x * r1z);
            const float r0z_ = r0z - dt * (om0x * r1y - om0y * r1x);
                
            
            const float a = v0x_*v0x_ / a2_bb + v0y_*v0y_ / b2_bb + v0z_*v0z_ / c2_bb;
            
            const float b = 2 * (r0x_*v0x_ / a2_bb + r0y_*v0y_ / b2_bb + r0z_*v0z_ / c2_bb);
                        
            const float c = r0x_*r0x_ / a2_bb + r0y_*r0y_ / b2_bb + r0z_*r0z_ / c2_bb - 1;

            return robust_quadratic_roots(a, b, c, h);
        }

        _DH_ void rescue(float *r) // cheap rescue
        {
            float scale = (1 + 1e-6) / sqrt(r[X] * r[X] / a2_bb + r[Y] * r[Y] / b2_bb + r[Z] * r[Z] / c2_bb);

            r[X] *= scale;
            r[Y] *= scale;
            r[Z] *= scale;
        }
    }
    
#else

#define shape none
    namespace none
    {
        _DH_ bool inside(const float *r)
        {
            printf("solidbounce: not implemented\n");
            exit(1);

            return false;
        }

        _DH_ bool intersect(const float *r0, const float *v0, const float *vcm, const float *om0, /**/ float *h)
        {
            printf("solidbounce: not implemented\n");
            exit(1);

            return false;
        }

        _DH_ void rescue(float *r)
        {
            printf("solidbounce: not implemented\n");
            exit(1);
        }
    }
    
#endif
}
