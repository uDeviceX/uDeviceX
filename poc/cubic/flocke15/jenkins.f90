!! ----------------------------------------------------------------------------------
!! MODULE JenkinsTraubSolver
!!
!!    This module contains the Jenkins-Traub solver for real polynomials.
!!    Adapted for variable working precision.
!!
!! ----------------------------------------------------------------------------------

module JenkinsTraubSolver

use SetWorkingPrecision, ONLY : wp

real (kind = wp) p(101), qp(101), k(101),                     &
                 qk(101), svk(101), sr, si, u, v, a, b, c, d, &
                 a1, a2, a3, a6, a7, e, f, g, h, szr, szi,    &
                 lzr, lzi

real (kind = wp) eta, are, mre

integer n, nn

contains


      subroutine rpoly(op, degree, zeror, zeroi, fail)
!
! finds the zeros of a real polynomial
! op  - double precision vector of coefficients in
!       order of decreasing powers.
! degree   - integer degree of polynomial.
! zeror, zeroi - output double precision vectors of
!                real and imaginary parts of the
!                zeros.
! fail  - output logical parameter, true only if
!         leading coefficient is zero or if rpoly
!         has found fewer than degree zeros.
!         in the latter case degree is reset to
!         the number of zeros found.
! to change the size of polynomials which can be
! solved, reset the dimensions of the arrays in the
! common area and in the following declarations.
! the subroutine uses single precision calculations
! for scaling, bounds and error calculations. all
! calculations for the iterations are done in double
! precision.
!
      real (kind = wp) op(101), temp(101), zeror(100), zeroi(100), &
                       t, aa, bb, cc, factor

      real (kind = wp) pt(101), lo, max, min, xx, yy, cosr,          &
                       sinr, xxx, x, sc, bnd, xm, ff, df, dx, infin, &
                       smalno, base

      integer degree, cnt, nz, i, l, j, jj, nm1

      logical fail, zerok
!
! the following statements set machine constants used
! in various parts of the program. the meaning of the
! four constants are...
! eta     the maximum relative representation error
!         which can be described as the smallest
!         positive floating point number such that
!         1.d0+eta is greater than 1.
! infiny  the largest floating-point number.
! smalno  the smallest positive floating-point number
!         if the exponent range differs in single and
!         double precision then smalno and infin
!         should indicate the smaller range.
! base    the base of the floating-point number
!         system used.
! the values below correspond to the burroughs b6700
!
      base   = radix   (1.0_wp)
      eta    = epsilon (1.0_wp)
      infin  = huge    (1.0_wp)
      smalno = tiny    (1.0_wp)
!
! are and mre refer to the unit error in + and *
! respectively. they are assumed to be the same as
! eta.
!
      are = eta
      mre = eta
      lo = smalno/eta
!
! initialization of constants for shift rotation
!
      xx = .70710678_wp
      yy = -xx
      cosr = -.069756474_wp
      sinr = .99756405_wp
      fail = .false.
      n = degree
      nn = n + 1
!
! algorithm fails if the leading coefficient is zero.
!
      if (op(1).ne.0.0_wp) go to 10
      fail = .true.
      degree = 0
      return
!
! remove the zeros at the origin if any
!
   10 if (op(nn).ne.0.0_wp) go to 20
      j = degree - n + 1
      zeror(j) = 0.0_wp
      zeroi(j) = 0.0_wp
      nn = nn - 1
      n = n - 1
      go to 10
!
! make a copy of the coefficients
!
   20 do 30 i=1,nn
        p(i) = op(i)
   30 continue
!
! start the algorithm for one zero
!
   40 if (n.gt.2) go to 60
      if (n.lt.1) return
!
! calculate the final zero or pair of zeros
!
      if (n.eq.2) go to 50
      zeror(degree) = -p(2)/p(1)
      zeroi(degree) = 0.0_wp
      return
   50 call quad(p(1), p(2), p(3), zeror(degree-1),           &
                zeroi(degree-1), zeror(degree), zeroi(degree))
      return
!
! find largest and smallest moduli of coefficients.
!
   60 max = 0.0_wp
      min = infin
      do 70 i=1,nn
        x = abs(p(i))
        if (x.gt.max) max = x
        if (x.ne.0. .and. x.lt.min) min = x
   70 continue
!
! scale if there are large or very small coefficients
! computes a scale factor to multiply the
! coefficients of the polynomial. the scaling is done
! to avoid overflow and to avoid undetected underflow
! interfering with the convergence criterion.
! the factor is a power of the base
!
      sc = lo/min
      if (sc.gt.1.0_wp) go to 80
      if (max.lt.10.0_wp) go to 110
      if (sc.eq.0.0_wp) sc = smalno
      go to 90
   80 if (infin/sc.lt.max) go to 110
   90 l = int(log(sc)/log(base) + 0.5_wp)
      factor = (base*1.0_wp)**l
      if (factor.eq.1.0_wp) go to 110
      do 100 i=1,nn
        p(i) = factor*p(i)
  100 continue
!
! compute lower bound on moduli of zeros.
!
  110 do 120 i=1,nn
        pt(i) = abs(p(i))
  120 continue
      pt(nn) = -pt(nn)
!
! compute upper estimate of bound
!
      x = exp((log(-pt(nn))-log(pt(1))) / real (n,wp))
      if (pt(n).eq.0.0_wp) go to 130
!
! if newton step at the origin is better, use it.
!
      xm = -pt(nn)/pt(n)
      if (xm.lt.x) x = xm
!
! chop the interval (0,x) until ff .le. 0
!
  130 xm = x * 0.1_wp
      ff = pt(1)
      do 140 i=2,nn
        ff = ff*xm + pt(i)
  140 continue
      if (ff.le.0.0_wp) go to 150
      x = xm
      go to 130
  150 dx = x
!
! do newton iteration until x converges to two
! decimal places
!
  160 if (abs(dx/x).le.0.005_wp) go to 180
      ff = pt(1)
      df = ff
      do 170 i=2,n
        ff = ff*x + pt(i)
        df = df*x + ff
  170 continue
      ff = ff*x + pt(nn)
      dx = ff/df
      x = x - dx
      go to 160
  180 bnd = x
!
! compute the derivative as the intial k polynomial
! and do 5 steps with no shift
!
      nm1 = n - 1
      do 190 i=2,n
        k(i) = real(nn-i,wp)*p(i)/real(n,wp)
  190 continue
      k(1) = p(1)
      aa = p(nn)
      bb = p(n)
      zerok = k(n).eq.0.0_wp
      do 230 jj=1,5
        cc = k(n)
        if (zerok) go to 210
!
! use scaled form of recurrence if value of k at 0 is
! nonzero
!
        t = -aa/cc
        do 200 i=1,nm1
          j = nn - i
          k(j) = t*k(j-1) + p(j)
  200   continue
        k(1) = p(1)
        zerok = abs(k(n)).le.abs(bb)*eta*10.0_wp
        go to 230
!
! use unscaled form of recurrence
!
  210   do 220 i=1,nm1
          j = nn - i
          k(j) = k(j-1)
  220   continue
        k(1) = 0.d0
        zerok = k(n).eq.0.0_wp
  230 continue
!
! save k for restarts with new shifts
!
      do 240 i=1,n
        temp(i) = k(i)
  240 continue
!
! loop to select the quadratic  corresponding to each
! new shift
!
      do 280 cnt=1,20
!
! quadratic corresponds to a double shift to a
! non-real point and its complex conjugate. the point
! has modulus bnd and amplitude rotated by 94 degrees
! from the previous shift
!
        xxx = cosr*xx - sinr*yy
        yy = sinr*xx + cosr*yy
        xx = xxx
        sr = bnd*xx
        si = bnd*yy
        u = - 2.0_wp * sr
        v = bnd
!
! second stage calculation, fixed quadratic
!
        call fxshfr(20*cnt, nz)
        if (nz.eq.0) go to 260
!
! the second stage jumps directly to one of the third
! stage iterations and returns here if successful.
! deflate the polynomial, store the zero or zeros and
! return to the main algorithm.
!
        j = degree - n + 1
        zeror(j) = szr
        zeroi(j) = szi
        nn = nn - nz
        n = nn - 1
        do 250 i=1,nn
          p(i) = qp(i)
  250   continue
        if (nz.eq.1) go to 40
        zeror(j+1) = lzr
        zeroi(j+1) = lzi
        go to 40
!
! if the iteration is unsuccessful another quadratic
! is chosen after restoring k
!
  260   do 270 i=1,n
          k(i) = temp(i)
  270   continue
  280 continue
!
! return with failure if no convergence with 20
! shifts
!
      fail = .true.
      degree = degree - n
      return
      end subroutine rpoly


      subroutine fxshfr(l2, nz)
!
! computes up to  l2  fixed shift k-polynomials,
! testing for convergence in the linear or quadratic
! case. initiates one of the variable shift
! iterations and returns with the number of zeros
! found.
! l2 - limit of fixed shift steps
! nz - number of zeros found
!
      real (kind = wp) svu, svv, ui, vi, s

      real (kind = wp) betas, betav, oss, ovv, ss, vv, ts, tv, ots, otv, tvv, tss

      integer l2, nz, type, i, j, iflag

      logical vpass, spass, vtry, stry

      nz = 0
      betav = .25_wp
      betas = .25_wp
      oss = sr
      ovv = v
!
! evaluate polynomial by synthetic division
!
      call quadsd(nn, u, v, p, qp, a, b)
      call calcsc(type)

      do 80 j=1,l2
!
! calculate next k polynomial and estimate v
!
        call nextk(type)
        call calcsc(type)
        call newest(type, ui, vi)

        vv = vi
!
! estimate s
!
        ss = 0.0_wp
        if (k(n).ne.0.0_wp) ss = -p(nn)/k(n)
        tv = 1.0_wp
        ts = 1.0_wp
        if (j.eq.1 .or. type.eq.3) go to 70
!
! compute relative measures of convergence of s and v
! sequences
!
        if (vv.ne.0.) tv = abs((vv-ovv)/vv)
        if (ss.ne.0.) ts = abs((ss-oss)/ss)
!
! if decreasing, multiply two most recent
! convergence measures
!
        tvv = 1.0_wp
        if (tv.lt.otv) tvv = tv*otv
        tss = 1.0_wp
        if (ts.lt.ots) tss = ts*ots
!
! compare with convergence criteria
!
        vpass = tvv.lt.betav
        spass = tss.lt.betas
        if (.not.(spass .or. vpass)) go to 70
!
! at least one sequence has passed the convergence
! test. store variables before iterating
!
        svu = u
        svv = v
        do 10 i=1,n
          svk(i) = k(i)
   10   continue
        s = ss
!
! choose iteration according to the fastest
! converging sequence
!
        vtry = .false.
        stry = .false.
        if (spass .and. ((.not.vpass) .or. tss.lt.tvv)) go to 40
   20   call quadit(ui, vi, nz)
        if (nz.gt.0) return
!
! quadratic iteration has failed. flag that it has
! been tried and decrease the convergence criterion.
!
        vtry = .true.
        betav = betav * 0.25_wp
!
! try linear iteration if it has not been tried and
! the s sequence is converging
!
        if (stry .or. (.not.spass)) go to 50
        do 30 i=1,n
          k(i) = svk(i)
   30   continue
   40   call realit(s, nz, iflag)
        if (nz.gt.0) return
!
! linear iteration has failed. flag that it has been
! tried and decrease the convergence criterion
!
        stry = .true.
        betas = betas * 0.25_wp
        if (iflag.eq.0) go to 50
!
! if linear iteration signals an almost double real
! zero attempt quadratic interation
!
        ui = -(s+s)
        vi = s*s
        go to 20
!
! restore variables
!
   50   u = svu
        v = svv
        do 60 i=1,n
          k(i) = svk(i)
   60   continue
!
! try quadratic iteration if it has not been tried
! and the v sequence is converging
!
        if (vpass .and. (.not.vtry)) go to 20
!
! recompute qp and scalar values to continue the
! second stage
!
        call quadsd(nn, u, v, p, qp, a, b)
        call calcsc(type)

   70   ovv = vv
        oss = ss
        otv = tv
        ots = ts
   80 continue
      return
      end subroutine fxshfr


      subroutine quadit(uu, vv, nz)
!
! variable-shift k-polynomial iteration for a
! quadratic factor converges only if the zeros are
! equimodular or nearly so.
! uu,vv - coefficients of starting quadratic
! nz - number of zero found
!
      real (kind = wp) ui, vi, uu, vv

      real (kind = wp) mp, omp, ee, relstp, t, zm

      integer nz, type, i, j

      logical tried

      nz = 0
      tried = .false.
      u = uu
      v = vv
      j = 0
!
! main loop
!
   10 call quad(1.0_wp, u, v, szr, szi, lzr, lzi)
!
! return if roots of the quadratic are real and not
! close to multiple or nearly equal and  of opposite
! sign
!
      if (abs(abs(szr)-abs(lzr)).gt.0.01_wp*abs(lzr)) return
!
! evaluate polynomial by quadratic synthetic division
!
      call quadsd(nn, u, v, p, qp, a, b)
      mp = abs(a-szr*b) + abs(szi*b)
!
! compute a rigorous  bound on the rounding error in
! evaluting p
!
      zm = sqrt(abs(v))
      ee = 2.0_wp * abs(qp(1))
      t = -szr*b
      do 20 i=2,n
        ee = ee*zm + abs(qp(i))
   20 continue
      ee = ee * zm + abs(a+t)
      ee =  (5.0_wp * mre + 4.0_wp * are) * ee                    &
          - (5.0_wp * mre + 2.0_wp * are) * (abs(a+t)+abs(b)*zm)  &
          + 2.0_wp * are * abs(t)
!
! iteration has converged sufficiently if the
! polynomial value is less than 20 times this bound
!
      if (mp.gt. 20.0_wp * ee) go to 30
      nz = 2
      return
   30 j = j + 1
!
! stop iteration after 20 steps
!
      if (j.gt.20) return
      if (j.lt.2) go to 50
      if (relstp.gt.0.01_wp .or. mp.lt.omp .or. tried) go to 50
!
! a cluster appears to be stalling the convergence.
! five fixed shift steps are taken with a u,v close
! to the cluster
!
      if (relstp.lt.eta) relstp = eta
      relstp = sqrt(relstp)
      u = u - u * relstp
      v = v + v * relstp
      call quadsd(nn, u, v, p, qp, a, b)
      do 40 i=1,5
        call calcsc(type)
        call nextk(type)
   40 continue
      tried = .true.
      j = 0
   50 omp = mp
!
! calculate next k polynomial and new u and v
!
      call calcsc(type)
      call nextk(type)
      call calcsc(type)
      call newest(type, ui, vi)
!
! if vi is zero the iteration is not converging
!
      if (vi.eq.0.0_wp) return
      relstp = abs((vi-v)/vi)
      u = ui
      v = vi
      go to 10
      end subroutine quadit


      subroutine realit(sss, nz, iflag)
!
! variable-shift h polynomial iteration for a real
! zero.
! sss   - starting iterate
! nz    - number of zero found
! iflag - flag to indicate a pair of zeros near real
!         axis.
!
      real (kind = wp) pv, kv, t, s, sss

      real (kind = wp) ms, mp, omp, ee

      integer nz, iflag, i, j

      nz = 0
      s = sss
      iflag = 0
      j = 0
!
! main loop
!
   10 pv = p(1)
!
! evaluate p at s
!
      qp(1) = pv
      do 20 i=2,nn
        pv = pv*s + p(i)
        qp(i) = pv
   20 continue
      mp = abs(pv)
!
! compute a rigorous bound on the error in evaluating p
!
      ms = abs(s)
      ee = (mre/(are+mre))*abs(qp(1))
      do 30 i=2,nn
        ee = ee*ms + abs(qp(i))
   30 continue
!
! iteration has converged sufficiently if the
! polynomial value is less than 20 times this bound
!
      if (mp.gt.20.0_wp*((are+mre)*ee-mre*mp)) go to 40
      nz = 1
      szr = s
      szi = 0.0_wp
      return
   40 j = j + 1
!
! stop iteration after 10 steps
!
      if (j.gt.10) return
      if (j.lt.2) go to 50
      if (abs(t).gt.0.001_wp*abs(s-t) .or. mp.le.omp) go to 50
!
! a cluster of zeros near the real axis has been
! encountered return with iflag set to initiate a
! quadratic iteration
!
      iflag = 1
      sss = s
      return
!
! return if the polynomial value has increased
! significantly
!
   50 omp = mp
!
! compute t, the next polynomial, and the new iterate
!
      kv = k(1)
      qk(1) = kv
      do 60 i=2,n
        kv = kv*s + k(i)
        qk(i) = kv
   60 continue
      if (abs(kv).le.abs(k(n))*10.0_wp*eta) go to 80
!
! use the scaled form of the recurrence if the value
! of k at s is nonzero
!
      t = -pv/kv
      k(1) = qp(1)
      do 70 i=2,n
        k(i) = t*qk(i-1) + qp(i)
   70 continue
      go to 100
!
! use unscaled form
!
   80 k(1) = 0.0_wp
      do 90 i=2,n
        k(i) = qk(i-1)
   90 continue
  100 kv = k(1)
      do 110 i=2,n
        kv = kv*s + k(i)
  110 continue
      t = 0.0_wp
      if (abs(kv).gt.abs(k(n))*10.0_wp*eta) t = -pv/kv
      s = s + t
      go to 10
      end subroutine realit


      subroutine calcsc(type)
!
! this routine calculates scalar quantities used to
! compute the next k polynomial and new estimates of
! the quadratic coefficients.
! type - integer variable set here indicating how the
! calculations are normalized to avoid overflow
!
      integer type
!
! synthetic division of k by the quadratic 1,u,v
!
      call quadsd(n, u, v, k, qk, c, d)
      if (abs(c).gt.abs(k(n))*100.0_wp*eta) go to 10
      if (abs(d).gt.abs(k(n-1))*100.0_wp*eta) go to 10
      type = 3
!
! type=3 indicates the quadratic is almost a factor
! of k
!
      return
   10 if (abs(d).lt.abs(c)) go to 20
      type = 2
!
! type=2 indicates that all formulas are divided by d
!
      e = a/d
      f = c/d
      g = u*b
      h = v*b
      a3 = (a+g)*e + h*(b/d)
      a1 = b*f - a
      a7 = (f+u)*a + h
      return
   20 type = 1
!
! type=1 indicates that all formulas are divided by c
!
      e = a/c
      f = d/c
      g = u*e
      h = v*b
      a3 = a*e + (h/c+g)*b
      a1 = b - a*(d/c)
      a7 = a + g*d + h*f
      return
      end subroutine calcsc


      subroutine nextk(type)
!
! computes the next k polynomials using scalars
! computed in calcsc
!
      real (kind = wp) temp

      integer i, type

      if (type.eq.3) go to 40
      temp = a
      if (type.eq.1) temp = b
      if (abs(a1).gt.abs(temp)*eta*10.0_wp) go to 20
!
! if a1 is nearly zero then use a special form of the
! recurrence
!
      k(1) = 0.0_wp
      k(2) = -a7*qp(1)
      do 10 i=3,n
        k(i) = a3*qk(i-2) - a7*qp(i-1)
   10 continue
      return
!
! use scaled form of the recurrence
!
   20 a7 = a7/a1
      a3 = a3/a1
      k(1) = qp(1)
      k(2) = qp(2) - a7*qp(1)
      do 30 i=3,n
        k(i) = a3*qk(i-2) - a7*qp(i-1) + qp(i)
   30 continue
      return
!
! use unscaled form of the recurrence if type is 3
!
   40 k(1) = 0.0_wp
      k(2) = 0.0_wp
      do 50 i=3,n
        k(i) = qk(i-2)
   50 continue
      return
      end subroutine nextk


      subroutine newest(type, uu, vv)
!
! compute new estimates of the quadratic coefficients
! using the scalars computed in calcsc.
!
      real (kind = wp) a4, a5, b1, b2, c1, c2, c3, c4, temp, uu, vv

      integer type
!
! use formulas appropriate to setting of type.
!
      if (type.eq.3) go to 30
      if (type.eq.2) go to 10
      a4 = a + u*b + h*f
      a5 = c + (u+v*f)*d
      go to 20
   10 a4 = (a+g)*f + h
      a5 = (f+u)*c + v*d
!
! evaluate new quadratic coefficients.
!
   20 b1 = -k(n)/p(nn)
      b2 = -(k(n-1)+b1*p(n))/p(nn)
      c1 = v*b2*a1
      c2 = b1*a7
      c3 = b1*b1*a3
      c4 = c1 - c2 - c3
      temp = a5 + b1*a4 - c4
      if (temp.eq.0.0_wp) go to 30
      uu = u - (u*(c3+c2)+v*(b1*a1+b2*a7))/temp
      vv = v*(1.0_wp+c4/temp)
      return
!
! if type=3 the quadratic is zeroed
!
   30 uu = 0.0_wp
      vv = 0.0_wp
      return
      end subroutine newest


      subroutine quadsd(nn, u, v, p, q, a, b)
!
! divides p by the quadratic  1,u,v  placing the
! quotient in q and the remainder in a,b
!
      integer i, nn

      real (kind = wp) p(nn), q(nn), u, v, a, b, c

      b = p(1)
      q(1) = b
      a = p(2) - u*b
      q(2) = a
      do 10 i=3,nn
        c = p(i) - u*a - v*b
        q(i) = c
        b = a
        a = c
   10 continue
      return
      end subroutine quadsd


      subroutine quad(a, b1, c, sr, si, lr, li)
!
! calculate the zeros of the quadratic a*z**2+b1*z+c.
! the quadratic formula, modified to avoid
! overflow, is used to find the larger zero if the
! zeros are real and both zeros are complex.
! the smaller real zero is found directly from the
! product of the zeros c/a.
!
      real (kind = wp) a, b1, c, sr, si, lr, li, b, d, e

      if (a.ne.0.0_wp) go to 20
      sr = 0.0_wp
      if (b1.ne.0.0_wp) sr = -c/b1
      lr = 0.0_wp
   10 si = 0.0_wp
      li = 0.0_wp
      return
   20 if (c.ne.0.0_wp) go to 30
      sr = 0.0_wp
      lr = -b1/a
      go to 10
!
! compute discriminant avoiding overflow
!
   30 b = b1/2.0_wp
      if (abs(b).lt.abs(c)) go to 40
      e = 1.0_wp - (a/b)*(c/b)
      d = sqrt(abs(e))*abs(b)
      go to 50
   40 e = a
      if (c.lt.0.0_wp) e = -a
      e = b*(b/abs(c)) - e
      d = sqrt(abs(e))*sqrt(abs(c))
   50 if (e.lt.0.0_wp) go to 60
!
! real zeros
!
      if (b.ge.0.0_wp) d = -d
      lr = (-b+d)/a
      sr = 0.0_wp
      if (lr.ne.0.0_wp) sr = (c/lr)/a
      go to 10
!
! complex conjugate zeros
!
   60 sr = -b/a
      lr = sr
      si = abs(d/a)
      li = -si
      return
      end subroutine quad


end module JenkinsTraubSolver
