//
//  main.cpp
//  cpudpd
//
//  Created by Dmitry Alexeev on 26/10/15.
//  Copyright © 2015 Dmitry Alexeev. All rights reserved.
//

#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <fenv.h>       /* fesetround, FE_* */
#include <cstdio>

#include <immintrin.h>

#include "logistic.h"
#include "timer.h"
#include "dpd8x8.h"
#include "iacaMarks.h"

#ifdef __INTEL_COMPILER
#include "avxoperators.h"
#endif

using namespace std;

struct Particle
{
	float *x[3], *u[3], *id;
};

struct Acceleration
{
	float *a[3];
};

const float dt = 0.001;
const float kBT = 0.0945;
const float gammadpd = 45;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = sigma / sqrt(dt);
const float aij = 25;

template<int s>
inline float viscosity_function(float x)
{
	return sqrtf(viscosity_function<s - 1>(x));
}

template<> inline float viscosity_function<1>(float x) { return sqrtf(x); }
template<> inline float viscosity_function<0>(float x) { return x; }


void streamCalc(int n,
				const float * const __restrict x1,  const float * const __restrict y1,  const float * const __restrict z1,
				const float * const __restrict x2,  const float * const __restrict y2,  const float * const __restrict z2,
				const float * const __restrict u1,  const float * const __restrict v1,  const float * const __restrict w1,
				const float * const __restrict u2,  const float * const __restrict v2,  const float * const __restrict w2,
				float * __restrict ax1, float * __restrict ay1, float * __restrict az1,
				float * __restrict ax2, float * __restrict ay2, float * __restrict az2)
{
	const __m256 vone    = _mm256_set1_ps(1.0f);
	const __m256 vaij    = _mm256_set1_ps(aij);
	const __m256 vgamma  = _mm256_set1_ps(gammadpd);
	const __m256 vsigmaf = _mm256_set1_ps(sigmaf);
	const __m256 veps    = _mm256_set1_ps(1e-14);
	
#pragma omp parallel for
	for (int i=0; i<n; i+=8)
	{
		const __m256 _xr = _mm256_load_ps(x1+i) - _mm256_load_ps(x2+i);
		const __m256 _yr = _mm256_load_ps(y1+i) - _mm256_load_ps(y2+i);
		const __m256 _zr = _mm256_load_ps(z1+i) - _mm256_load_ps(z2+i);
		
		const __m256 rij2 = _xr * _xr + _yr * _yr + _zr * _zr + veps;
		
		const __m256 invrij = _mm256_rsqrt_ps(rij2);
		const __m256 rij = rij2 * invrij;
		
		const __m256 argwr = _mm256_max_ps(vone - rij, _mm256_setzero_ps());
		const __m256 wr = argwr; // viscosity
		
		const __m256 xr = _xr * invrij;
		const __m256 yr = _yr * invrij;
		const __m256 zr = _zr * invrij;
		
		const __m256 _ur = _mm256_load_ps(u1+i) - _mm256_load_ps(u2+i);
		const __m256 _vr = _mm256_load_ps(v1+i) - _mm256_load_ps(v2+i);
		const __m256 _wr = _mm256_load_ps(w1+i) - _mm256_load_ps(w2+i);
		
		const __m256 rdotv = xr * _ur + yr * _vr + zr * _wr;
		
		const __m256 myrandnr = Logistic::mean0var1_unrolled(vone,
													_mm256_set_ps(i+8,   i+7,   i+6,   i+5,   i+4,   i+3,   i+2,   i+1),
													_mm256_set_ps(i+n+8, i+n+7, i+n+6, i+n+5, i+n+4, i+n+3, i+n+2, i+n+1));
		
		const __m256 strength = vaij * argwr - ( vgamma * wr * rdotv + vsigmaf * myrandnr ) * wr;
		
		const __m256 xinteraction = strength * xr;
		const __m256 yinteraction = strength * yr;
		const __m256 zinteraction = strength * zr;
		
		
		_mm256_store_ps(ax1+i, xinteraction);
		_mm256_store_ps(ay1+i, yinteraction);
		_mm256_store_ps(az1+i, zinteraction);
		
		_mm256_store_ps(ax2+i, -xinteraction);
		_mm256_store_ps(ay2+i, -yinteraction);
		_mm256_store_ps(az2+i, -zinteraction);
	}
}

inline float horizontal_add (__m256 a)
{
	// http://stackoverflow.com/questions/13879609/horizontal-sum-of-8-packed-32bit-floats/18616679#18616679
	__m256 t1 = _mm256_hadd_ps(a,a);
	__m256 t2 = _mm256_hadd_ps(t1,t1);
	__m128 t3 = _mm256_extractf128_ps(t2,1);
	__m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
	return _mm_cvtss_f32(t4);
}

inline __m256 ROL(__m256 x)
{
	const __m256 t0 = _mm256_permute_ps(x, 0b00111001);
	const __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0b0101);
	return _mm256_blend_ps(t0, t1, 0b10001000);
}

inline __m256 ROR(__m256 x)
{
	const __m256 t0 = _mm256_permute_ps(x, 0b10010011);
	const __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0b0101);
	return _mm256_blend_ps(t0, t1, 0b00010001);
}

void squareCalc(int n,
				const float * const __restrict x1,  const float * const __restrict y1,  const float * const __restrict z1,
				const float * const __restrict u1,  const float * const __restrict v1,  const float * const __restrict w1,
				float * const __restrict ax1, float * const __restrict ay1, float * const __restrict az1,
				const float * const __restrict id1)
{
	const __m256 vone    = _mm256_set1_ps(1.0f);
	const __m256 vaij    = _mm256_set1_ps(aij);
	const __m256 vgamma  = _mm256_set1_ps(gammadpd);
	const __m256 vsigmaf = _mm256_set1_ps(sigmaf);
	
#pragma omp parallel for
	for (int i=0; i<n; i++)
	{
		__m256 x = _mm256_set1_ps(x1[i]);
		__m256 y = _mm256_set1_ps(y1[i]);
		__m256 z = _mm256_set1_ps(z1[i]);
		
		__m256 u = _mm256_set1_ps(u1[i]);
		__m256 v = _mm256_set1_ps(v1[i]);
		__m256 w = _mm256_set1_ps(w1[i]);
		
		__m256 ax = _mm256_set1_ps(0);
		__m256 ay = _mm256_set1_ps(0);
		__m256 az = _mm256_set1_ps(0);
		
		__m256 id = _mm256_set1_ps(id1[i]);
		
		for (int k=0; k<n; k+=32)
		{
			IACA_START
			
			__m256 r0, r1, r2, r3;
			Logistic::mean0var1_unrolledx4(r0, r1, r2, r3, vone, id,
										   _mm256_load_ps(id1 + k + 0),  _mm256_load_ps(id1 + k + 8),
										   _mm256_load_ps(id1 + k + 16), _mm256_load_ps(id1 + k + 24));
			
			inter8x4(x1+k, y1+k, z1+k, u1+k, v1+k, w1+k,
					 x,  y,  z,  u,  v,  w,
					 ax, ay, az,
					 r0, r1, r2, r3);
		}
		IACA_END
		
		ax1[i] = horizontal_add(ax);
		ay1[i] = horizontal_add(ay);
		az1[i] = horizontal_add(az);
	}
}

//=====================================================================================================
//=====================================================================================================
//  CHECHERS
//=====================================================================================================
//=====================================================================================================

void streamCheck(int n,
				 const float * const __restrict x1,  const float * const __restrict y1,  const float * const __restrict z1,
				 const float * const __restrict x2,  const float * const __restrict y2,  const float * const __restrict z2,
				 const float * const __restrict u1,  const float * const __restrict v1,  const float * const __restrict w1,
				 const float * const __restrict u2,  const float * const __restrict v2,  const float * const __restrict w2,
				 float * __restrict ax1, float * __restrict ay1, float * __restrict az1,
				 float * __restrict ax2, float * __restrict ay2, float * __restrict az2)

{
#pragma omp parallel for
	for (int i=0; i<n; i++)
	{
		const float _xr = x1[i] - x2[i];
		const float _yr = y1[i] - y2[i];
		const float _zr = z1[i] - z2[i];
		
		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr + 1e-14;
		
		const float invrij = 1.0f / sqrtf( rij2 );
		const float rij = rij2 * invrij;
		const float wc = max(0.0f, 1.0f - rij);
		
		const float wr = viscosity_function < 0 > ( wc );
		
		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;
		
		const float rdotv =
		xr * ( u1[i] - u2[i] ) +
		yr * ( v1[i] - v2[i] ) +
		zr * ( w1[i] - w2[i] );
		
		const float myrandnr = LogisticNonvec::mean0var1( 1.0f, i+1, i+n+1 );
		
		const float strength = aij * wc - ( gammadpd * wr * rdotv + sigmaf * myrandnr ) * wr;
		
		ax1[i] =  strength * xr;
		ay1[i] =  strength * yr;
		az1[i] =  strength * zr;
		
		ax2[i] =  -strength * xr;
		ay2[i] =  -strength * yr;
		az2[i] =  -strength * zr;
	}
}

void squareCheck(int n,
				 const float * const __restrict x1,  const float * const __restrict y1,  const float * const __restrict z1,
				 const float * const __restrict u1,  const float * const __restrict v1,  const float * const __restrict w1,
				 float * const __restrict ax1, float * const __restrict ay1, float * const __restrict az1,
				 const float * const __restrict id1)
{
#pragma omp parallel for
	for (int i=0; i<n; i++)
	{
		float ax = 0, ay = 0, az = 0;
		
		for (int j=0; j<n; j++)
			if (i != j)
			{
				const float _xr = x1[i] - x1[j];
				const float _yr = y1[i] - y1[j];
				const float _zr = z1[i] - z1[j];
				
				const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
				
				const float invrij = 1.0f / sqrtf( rij2 );
				const float rij = rij2 * invrij;
				const float wc = max(0.0f, 1.0f - rij);
				
				const float wr = viscosity_function < 0 > ( wc );
				
				const float xr = _xr * invrij;
				const float yr = _yr * invrij;
				const float zr = _zr * invrij;
				
				const float rdotv =
				xr * ( u1[i] - u1[j] ) +
				yr * ( v1[i] - v1[j] ) +
				zr * ( w1[i] - w1[j] );
				
				const float myrandnr = LogisticNonvec::mean0var1( 1.0f, i+1, j+1 );
				
				const float strength = aij * wc - ( gammadpd * wr * rdotv + sigmaf * myrandnr ) * wr;
				
				ax +=  strength * xr;
				ay +=  strength * yr;
				az +=  strength * zr;
			}
		
		ax1[i] = ax;
		ay1[i] = ay;
		az1[i] = az;
	}
}

void sum(int n,
		 const float * const __restrict ax1, const float * const __restrict ay1, const float * const __restrict az1,
		 double &dx, double &dy, double &dz)
{
	double ddx=0, ddy=0, ddz=0;
	
#pragma omp parallel for reduction(+:ddx,ddy,ddz)
	for (int i = 0; i<n; i++)
	{
		ddx += ax1[i];
		ddy += ay1[i];
		ddz += az1[i];
	}
	
	dx += ddx;
	dy += ddy;
	dz += ddz;
}


int main(int argc, const char * argv[])
{
	static const int n = 8*5000000;
	
	Particle p1, p2;
	p1.x[0] = (float *)_mm_malloc(n*sizeof(float), 32);
	p1.x[1] = (float *)_mm_malloc(n*sizeof(float), 32);
	p1.x[2] = (float *)_mm_malloc(n*sizeof(float), 32);
	p1.u[0] = (float *)_mm_malloc(n*sizeof(float), 32);
	p1.u[1] = (float *)_mm_malloc(n*sizeof(float), 32);
	p1.u[2] = (float *)_mm_malloc(n*sizeof(float), 32);
	p1.id   = (float *)_mm_malloc(n*sizeof(float), 32);
	
	p2.x[0] = (float *)_mm_malloc(n*sizeof(float), 32);
	p2.x[1] = (float *)_mm_malloc(n*sizeof(float), 32);
	p2.x[2] = (float *)_mm_malloc(n*sizeof(float), 32);
	p2.u[0] = (float *)_mm_malloc(n*sizeof(float), 32);
	p2.u[1] = (float *)_mm_malloc(n*sizeof(float), 32);
	p2.u[2] = (float *)_mm_malloc(n*sizeof(float), 32);
	p2.id   = (float *)_mm_malloc(n*sizeof(float), 32);
	
	Acceleration a1, a2;
	a1.a[0] = (float *)_mm_malloc(n*sizeof(float), 32);
	a1.a[1] = (float *)_mm_malloc(n*sizeof(float), 32);
	a1.a[2] = (float *)_mm_malloc(n*sizeof(float), 32);
	a2.a[0] = (float *)_mm_malloc(n*sizeof(float), 32);
	a2.a[1] = (float *)_mm_malloc(n*sizeof(float), 32);
	a2.a[2] = (float *)_mm_malloc(n*sizeof(float), 32);
	
	printf("Initializing...\n");
	
	srand48(time(NULL));
	
#pragma omp parallel for
	for (int i=0; i<n; i++)
	{
		p1.x[0][i] = drand48();
		p1.x[1][i] = drand48();
		p1.x[2][i] = drand48();
		p2.x[0][i] = drand48();
		p2.x[1][i] = drand48();
		p2.x[2][i] = drand48();
		
		p1.u[0][i] = drand48() - 0.5f;
		p1.u[1][i] = drand48() - 0.5f;
		p1.u[2][i] = drand48() - 0.5f;
		p2.u[0][i] = drand48() - 0.5f;
		p2.u[1][i] = drand48() - 0.5f;
		p2.u[2][i] = drand48() - 0.5f;
		
		p1.id[i] = (float) i+1;
		p2.id[i] = (float) i+1;
	}
	
	int nth;
#pragma omp parallel
	{
		fesetround(FE_TOWARDZERO);
		
		nth = omp_get_num_threads();
	}
	printf("Benchmark will begin now with %d threads\n", nth);
	
	//=====================================================================================================
	//=====================================================================================================
	
	const int trials = 20;
	
	double avg = 0;
	vector<double> records;
	double dx = 0, dy = 0, dz = 0;
	
	{
		const int nch = 8*10;
		float *a1xch = new float[nch];
		float *a1ych = new float[nch];
		float *a1zch = new float[nch];
		
		float *a2xch = new float[nch];
		float *a2ych = new float[nch];
		float *a2zch = new float[nch];
		
		streamCheck(nch,
					p1.x[0], p1.x[1], p1.x[2], p2.x[0], p2.x[1], p2.x[2],
					p1.u[0], p1.u[1], p1.u[2], p2.u[0], p2.u[1], p2.u[2],
					a1xch,   a1ych,   a1zch,   a2xch,   a2ych,   a2zch);
		
		streamCalc(nch,
				   p1.x[0], p1.x[1], p1.x[2], p2.x[0], p2.x[1], p2.x[2],
				   p1.u[0], p1.u[1], p1.u[2], p2.u[0], p2.u[1], p2.u[2],
				   a1.a[0], a1.a[1], a1.a[2], a2.a[0], a2.a[1], a2.a[2]);
		
		double linf = 0, l2 = 0;
		for (int i = 0; i<nch; i++)
		{
			double diff = sqrt((a1.a[0][i] - a1xch[i]) * (a1.a[0][i] - a1xch[i]) +
							   (a1.a[1][i] - a1ych[i]) * (a1.a[1][i] - a1ych[i]) +
							   (a1.a[2][i] - a1zch[i]) * (a1.a[2][i] - a1zch[i]));
			
			//printf("%f  %f\n", a1.a[0][i] , a1xch[i]);

			l2 += diff*diff;
			linf = max(diff, linf);
		}
		
		l2 = sqrt(l2/n);
		printf("\nStream of interactions:\n");
		printf("  Accuracy check  Linf:  %f,  L2:  %f\n", linf, l2);
	}
	
	
	for (int i = 0; i<trials; i++)
	{
		Timer tm;
		tm.start();
		
		streamCalc(n,
				   p1.x[0], p1.x[1], p1.x[2], p2.x[0], p2.x[1], p2.x[2],
				   p1.u[0], p1.u[1], p1.u[2], p2.u[0], p2.u[1], p2.u[2],
				   a1.a[0], a1.a[1], a1.a[2], a2.a[0], a2.a[1], a2.a[2]);
		
		double inters = (double)n / ((double)tm.elapsed() * 1.0e-9) * 1e-6;
		sum(n, a1.a[0], a1.a[1], a1.a[2], dx, dy, dz);
		sum(n, a2.a[0], a2.a[1], a2.a[2], dx, dy, dz);
		records.push_back(inters);
		avg += inters;
	}
	avg /= trials;
	
	sort(records.begin(), records.end());
	
	printf("  Sum: [%8f %8f %8f]\n", dx, dy, dz);
	printf("  10%% / mean / 90%%  :  %.3f / %.3f / %.3f  MI/s (max %.3f GB/s)\n",
		   records[trials / 10], avg, records[(9*trials) / 10 - 1], 72.0 * records.back() / 1e3);
	
	//=====================================================================================================
	//=====================================================================================================
	
	dx = 0, dy = 0, dz = 0;
	avg = 0;
	const int ns = 32*2000;
	records.clear();
	
	{
		const int nch = 32*10;
		float *axch = new float[nch];
		float *aych = new float[nch];
		float *azch = new float[nch];
		
		squareCalc(nch,
				   p1.x[0], p1.x[1], p1.x[2],
				   p1.u[0], p1.u[1], p1.u[2],
				   a1.a[0], a1.a[1], a1.a[2], p1.id);
		
		squareCheck(nch,
					p1.x[0], p1.x[1], p1.x[2],
					p1.u[0], p1.u[1], p1.u[2],
					axch, aych, azch, p1.id);
		
		double linf = 0, l2 = 0;
		for (int i = 0; i<nch; i++)
		{
			double diff = sqrt((a1.a[0][i] - axch[i]) * (a1.a[0][i] - axch[i]) +
							   (a1.a[1][i] - aych[i]) * (a1.a[1][i] - aych[i]) +
							   (a1.a[2][i] - azch[i]) * (a1.a[2][i] - azch[i]));
			
			//printf("%f  %f\n", a1.a[0][i] , axch[i]);
			
			l2 += diff*diff;
			linf = max(diff, linf);
		}
		
		l2 = sqrt(l2/n);
		printf("\nAll to all interactions:\n");
		printf("  Accuracy check  Linf:  %f,  L2:  %f\n", linf, l2);
	}
	
	for (int i = 0; i<trials; i++)
	{
		Timer tm;
		tm.start();
		
		squareCalc(ns,
				   p1.x[0], p1.x[1], p1.x[2],
				   p1.u[0], p1.u[1], p1.u[2],
				   a1.a[0], a1.a[1], a1.a[2], p1.id);
		
		double inters = (double)ns*ns / ((double)tm.elapsed() * 1.0e-9) * 1e-6;
		sum(ns, a1.a[0], a1.a[1], a1.a[2], dx, dy, dz);
		records.push_back(inters);
		avg += inters;
	}
	avg /= trials;
	
	sort(records.begin(), records.end());
	
	printf("  Sum: [%8f %8f %8f]\n", dx, dy, dz);
	printf("  10%% / mean / 90%%  :  %.3f / %.3f / %.3f  MI/s (max %.3f Gflops/s)\n",
		   records[trials / 10], avg, records[(9*trials) / 10 - 1], 103 * records.back() / 1e3);
	
	const double t = (double)ns*ns / records.back() * 1e-6;
	printf("  Cycles per 8x4:  %.1f\n", t / ((double)ns*ns/32.0 / (nth/2.0)) * 2.6e9);
	
	
	return 0;
}
