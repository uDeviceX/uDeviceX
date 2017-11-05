!! ----------------------------------------------------------------------------------
!! MODULE Polynomial234RootSolvers
!!
!!    This module contains the three root solvers for quadratic,
!!    cubic and quartic polynomials.
!!
!! ----------------------------------------------------------------------------------

module Polynomial234RootSolvers

use SetWorkingPrecision, ONLY : wp, wpformat

contains

!!-----------------------------------------------------------------------------------
!!
!! CUBIC POLYNOMIAL ROOT SOLVER
!!
!! SYNOPSIS
!!
!!  call cubicRoots (real,              intent (in)  :: c2,
!!                   real,              intent (in)  :: c1,
!!                   real,              intent (in)  :: c0,
!!                   integer,           intent (out) :: nReal,
!!                   real,              intent (out) :: root (1:3,1:2),
!!                   logical, optional, intent (in)  :: printInfo)
!!
!! DESCRIPTION
!!
!!  Calculates all real + complex roots of the cubic polynomial:
!!
!!                 x^3 + c2 * x^2 + c1 * x + c0
!!
!!  The first real root (which always exists) is obtained using an optimized
!!  Newton-Raphson scheme. The other remaining roots are obtained through
!!  composite deflation into a quadratic. An option for printing detailed info
!!  about the intermediate stages in solving the cubic is available.
!!
!!  The cubic root solver can handle any size of cubic coefficients and there is
!!  no danger of overflow due to proper rescaling of the cubic polynomial.
!!
!!  The order of the roots is as follows:
!!
!!        1) For real roots, the order is according to their algebraic value
!!           on the number scale (largest positive first, largest negative last).
!!
!!        2) Since there can be only one complex conjugate pair root, no order
!!           is necessary.
!!
!!        3) All real roots preceede the complex ones.
!!
!! ARGUMENTS
!!
!!  c2         : coefficient of x^2 term
!!  c1         : coefficient of x term
!!  c0         : independent coefficient
!!  nReal      : number of real roots found
!!  root (n,1) : real part of n-th root
!!  root (n,2) : imaginary part of n-th root
!!  printInfo  : if given and true, detailed info will be printed about intermediate stages
!!
!! NOTES
!!
!!***

subroutine cubicRoots (c2, c1, c0, nReal, root, printInfo)

  implicit none

  logical, optional  , intent (in)  :: printInfo
  integer            , intent (out) :: nReal
  real    (kind = wp), intent (in)  :: c2, c1, c0
  real    (kind = wp), intent (out) :: root (1:3,1:2)

  logical :: bisection
  logical :: converged
  logical :: doPrint

  integer :: cubicType
  integer :: deflateCase
  integer :: oscillate

  integer, parameter :: Re = 1
  integer, parameter :: Im = 2

  integer, parameter :: allzero   = 0
  integer, parameter :: linear    = 1
  integer, parameter :: quadratic = 2
  integer, parameter :: general   = 3

  real (kind = wp) :: a0, a1, a2
  real (kind = wp) :: a, b, c, k, s, t, u, x, y, z
  real (kind = wp) :: xShift

  real (kind = wp), parameter :: macheps = epsilon (1.0_wp)
  real (kind = wp), parameter :: one27th = 1.0_wp / 27.0_wp
  real (kind = wp), parameter :: two27th = 2.0_wp / 27.0_wp
  real (kind = wp), parameter :: third   = 1.0_wp /  3.0_wp

  real (kind = wp), parameter :: p1 = 1.09574_wp         !
  real (kind = wp), parameter :: q1 = 3.23900e-1_wp      ! Newton-Raphson coeffs for class 1 and 2
  real (kind = wp), parameter :: r1 = 3.23900e-1_wp      !
  real (kind = wp), parameter :: s1 = 9.57439e-2_wp      !

  real (kind = wp), parameter :: p3 = 1.14413_wp         !
  real (kind = wp), parameter :: q3 = 2.75509e-1_wp      ! Newton-Raphson coeffs for class 3
  real (kind = wp), parameter :: r3 = 4.45578e-1_wp      !
  real (kind = wp), parameter :: s3 = 2.59342e-2_wp      !

  real (kind = wp), parameter :: q4 = 7.71845e-1_wp      ! Newton-Raphson coeffs for class 4
  real (kind = wp), parameter :: s4 = 2.28155e-1_wp      !

  real (kind = wp), parameter :: p51 = 8.78558e-1_wp     !
  real (kind = wp), parameter :: p52 = 1.92823e-1_wp     !
  real (kind = wp), parameter :: p53 = 1.19748_wp        !
  real (kind = wp), parameter :: p54 = 3.45219e-1_wp     !
  real (kind = wp), parameter :: q51 = 5.71888e-1_wp     !
  real (kind = wp), parameter :: q52 = 5.66324e-1_wp     !
  real (kind = wp), parameter :: q53 = 2.83772e-1_wp     ! Newton-Raphson coeffs for class 5 and 6
  real (kind = wp), parameter :: q54 = 4.01231e-1_wp     !
  real (kind = wp), parameter :: r51 = 7.11154e-1_wp     !
  real (kind = wp), parameter :: r52 = 5.05734e-1_wp     !
  real (kind = wp), parameter :: r53 = 8.37476e-1_wp     !
  real (kind = wp), parameter :: r54 = 2.07216e-1_wp     !
  real (kind = wp), parameter :: s51 = 3.22313e-1_wp     !
  real (kind = wp), parameter :: s52 = 2.64881e-1_wp     !
  real (kind = wp), parameter :: s53 = 3.56228e-1_wp     !
  real (kind = wp), parameter :: s54 = 4.45532e-3_wp     !
!
!
!     ...Start.
!
!
  if (present (printInfo)) then
      doPrint = printInfo
  else
      doPrint = .false.
  end if

  if (doPrint) then
      write (*,wpformat) ' initial cubic c2      = ',c2
      write (*,wpformat) ' initial cubic c1      = ',c1
      write (*,wpformat) ' initial cubic c0      = ',c0
      write (*,wpformat) ' ------------------------------------------------'
  end if
!
!
!     ...Handle special cases.
!
!            1) all terms zero
!            2) only quadratic term is nonzero -> linear equation.
!            3) only independent term is zero -> quadratic equation.
!
!
  if (c0 == 0.0_wp .and. c1 == 0.0_wp .and. c2 == 0.0_wp) then

      cubicType = allzero

  else if (c0 == 0.0_wp .and. c1 == 0.0_wp) then

      k  = 1.0_wp
      a2 = c2

      cubicType = linear

  else if (c0 == 0.0_wp) then

      k  = 1.0_wp
      a2 = c2
      a1 = c1

      cubicType = quadratic

  else
!
!
!     ...The general case. Rescale cubic polynomial, such that largest absolute coefficient
!        is (exactly!) equal to 1. Honor the presence of a special cubic case that might have
!        been obtained during the rescaling process (due to underflow in the coefficients).
!
!
      x = abs (c2)
      y = sqrt (abs (c1))
      z = abs (c0) ** third
      u = max (x,y,z)

      if (u == x) then

          k  = 1.0_wp / x
          a2 = sign (1.0_wp , c2)
          a1 = (c1 * k) * k
          a0 = ((c0 * k) * k) * k

      else if (u == y) then

          k  = 1.0_wp / y
          a2 = c2 * k
          a1 = sign (1.0_wp , c1)
          a0 = ((c0 * k) * k) * k

      else

          k  = 1.0_wp / z
          a2 = c2 * k
          a1 = (c1 * k) * k
          a0 = sign (1.0_wp , c0)

      end if

      if (doPrint) then
          write (*,wpformat) ' rescaling factor      = ',k
          write (*,wpformat) ' ------------------------------------------------'
          write (*,wpformat) ' rescaled cubic c2     = ',a2
          write (*,wpformat) ' rescaled cubic c1     = ',a1
          write (*,wpformat) ' rescaled cubic c0     = ',a0
          write (*,wpformat) ' ------------------------------------------------'
      end if

      k = 1.0_wp / k

      if (a0 == 0.0_wp .and. a1 == 0.0_wp .and. a2 == 0.0_wp) then
          cubicType = allzero
      else if (a0 == 0.0_wp .and. a1 == 0.0_wp) then
          cubicType = linear
      else if (a0 == 0.0_wp) then
          cubicType = quadratic
      else
          cubicType = general
      end if

  end if
!
!
!     ...Select the case.
!
!        1) Only zero roots.
!
!
  select case (cubicType)

    case (allzero)

      nReal = 3

      root (:,Re) = 0.0_wp
      root (:,Im) = 0.0_wp
!
!
!     ...2) The linear equation case -> additional 2 zeros.
!
!
    case (linear)

      x = - a2 * k

      nReal = 3

      root (1,Re) = max (0.0_wp, x)
      root (2,Re) = 0.0_wp
      root (3,Re) = min (0.0_wp, x)
      root (:,Im) = 0.0_wp
!
!
!     ...3) The quadratic equation case -> additional 1 zero.
!
!
    case (quadratic)

      call quadraticRoots (a2, a1, nReal, root (1:2,1:2))

      if (nReal == 2) then

          x = root (1,1) * k         ! real roots of quadratic are ordered x >= y
          y = root (2,1) * k

          nReal = 3

          root (1,Re) = max (x, 0.0_wp)
          root (2,Re) = max (y, min (x, 0.0_wp))
          root (3,Re) = min (y, 0.0_wp)
          root (:,Im) = 0.0_wp

      else

          nReal = 1

          root (3,Re) = root (2,Re) * k
          root (2,Re) = root (1,Re) * k
          root (1,Re) = 0.0_wp
          root (3,Im) = root (2,Im) * k
          root (2,Im) = root (1,Im) * k
          root (1,Im) = 0.0_wp

      end if
!
!
!     ...3) The general cubic case. Set the best Newton-Raphson root estimates for the cubic.
!           The easiest and most robust conditions are checked first. The most complicated
!           ones are last and only done when absolutely necessary.
!
!
    case (general)

      if (a0 == 1.0_wp) then

          x = - p1 + q1 * a1 - a2 * (r1 - s1 * a1)

          a = a2
          b = a1
          c = a0
          xShift = 0.0_wp

      else if (a0 == - 1.0_wp) then

          x = p1 - q1 * a1 - a2 * (r1 - s1 * a1)

          a = a2
          b = a1
          c = a0
          xShift = 0.0_wp

      else if (a1 == 1.0_wp) then

          if (a0 > 0.0_wp) then
              x = a0 * (- q4 - s4 * a2)
          else
              x = a0 * (- q4 + s4 * a2)
          end if

          a = a2
          b = a1
          c = a0
          xShift = 0.0_wp

      else if (a1 == - 1.0_wp) then

          y = - two27th
          y = y * a2
          y = y * a2 - third
          y = y * a2

          if (a0 < y) then
              x = + p3 - q3 * a0 - a2 * (r3 + s3 * a0)               ! + guess
          else
              x = - p3 - q3 * a0 - a2 * (r3 - s3 * a0)               ! - guess
          end if

          a = a2
          b = a1
          c = a0
          xShift = 0.0_wp

      else if (a2 == 1.0_wp) then

          b = a1 - third
          c = a0 - one27th

          if (abs (b) < macheps .and. abs (c) < macheps) then        ! triple -1/3 root

              x = - third * k

              nReal = 3

              root (:,Re) = x
              root (:,Im) = 0.0_wp

              return

          else

              y = third * a1 - two27th

              if (a1 <= third) then
                  if (a0 > y) then
                      x = - p51 - q51 * a0 + a1 * (r51 - s51 * a0)   ! - guess
                  else
                      x = + p52 - q52 * a0 - a1 * (r52 + s52 * a0)   ! + guess
                  end if
              else
                  if (a0 > y) then
                      x = - p53 - q53 * a0 + a1 * (r53 - s53 * a0)   ! <-1/3 guess
                  else
                      x = + p54 - q54 * a0 - a1 * (r54 + s54 * a0)   ! >-1/3 guess
                  end if
              end if

              if (abs (b) < 1.e-2_wp .and. abs (c) < 1.e-2_wp) then  ! use shifted root
                  c = - third * b + c
                  if (abs (c) < macheps) c = 0.0_wp                  ! prevent random noise
                  a = 0.0_wp
                  xShift = third
                  x = x + xShift
              else
                  a = a2
                  b = a1
                  c = a0
                  xShift = 0.0_wp
              end if

          end if

      else if (a2 == - 1.0_wp) then

          b = a1 - third
          c = a0 + one27th

          if (abs (b) < macheps .and. abs (c) < macheps) then        ! triple 1/3 root

              x = third * k

              nReal = 3

              root (:,Re) = x
              root (:,Im) = 0.0_wp

              return

          else

              y = two27th - third * a1

              if (a1 <= third) then
                  if (a0 < y) then
                      x = + p51 - q51 * a0 - a1 * (r51 + s51 * a0)   ! +1 guess
                  else
                      x = - p52 - q52 * a0 + a1 * (r52 - s52 * a0)   ! -1 guess
                  end if
              else
                  if (a0 < y) then
                      x = + p53 - q53 * a0 - a1 * (r53 + s53 * a0)   ! >1/3 guess
                  else
                      x = - p54 - q54 * a0 + a1 * (r54 - s54 * a0)   ! <1/3 guess
                  end if
              end if

              if (abs (b) < 1.e-2_wp .and. abs (c) < 1.e-2_wp) then  ! use shifted root
                  c = third * b + c
                  if (abs (c) < macheps) c = 0.0_wp                  ! prevent random noise
                  a = 0.0_wp
                  xShift = - third
                  x = x + xShift
              else
                  a = a2
                  b = a1
                  c = a0
                  xShift = 0.0_wp
              end if

          end if

      end if
!
!
!     ...Perform Newton/Bisection iterations on x^3 + ax^2 + bx + c.
!
!
      z = x + a
      y = x + z
      z = z * x + b
      y = y * x + z       ! C'(x)
      z = z * x + c       ! C(x)
      t = z               ! save C(x) for sign comparison
      x = x - z / y       ! 1st improved root

      oscillate = 0
      bisection = .false.
      converged = .false.

      do while (.not.converged .and. .not.bisection)    ! Newton-Raphson iterates

         z = x + a
         y = x + z
         z = z * x + b
         y = y * x + z
         z = z * x + c

         if (z * t < 0.0_wp) then                       ! does Newton start oscillating ?
             if (z < 0.0_wp) then
                 oscillate = oscillate + 1              ! increment oscillation counter
                 s = x                                  ! save lower bisection bound
             else
                 u = x                                  ! save upper bisection bound
             end if
             t = z                                      ! save current C(x)
         end if

         y = z / y                                      ! Newton correction
         x = x - y                                      ! new Newton root

         bisection = oscillate > 2                      ! activate bisection
         converged = abs (y) <= abs (x) * macheps       ! Newton convergence indicator

         if (doPrint) write (*,wpformat) ' Newton root           = ',x

      end do

      if (bisection) then

          t = u - s                                     ! initial bisection interval
          do while (abs (t) > abs (x) * macheps)        ! bisection iterates

             z = x + a                                  !
             z = z * x + b                              ! C (x)
             z = z * x + c                              !

             if (z < 0.0_wp) then                       !
                 s = x                                  !
             else                                       ! keep bracket on root
                 u = x                                  !
             end if                                     !

             t = 0.5_wp * (u - s)                       ! new bisection interval
             x = s + t                                  ! new bisection root

             if (doPrint) write (*,wpformat) ' Bisection root        = ',x

          end do
      end if

      if (doPrint) write (*,wpformat) ' ------------------------------------------------'

      x = x - xShift                                    ! unshift root
!
!
!     ...Forward / backward deflate rescaled cubic (if needed) to check for other real roots.
!        The deflation analysis is performed on the rescaled cubic. The actual deflation must
!        be performed on the original cubic, not the rescaled one. Otherwise deflation errors
!        will be enhanced when undoing the rescaling on the extra roots.
!
!
      z = abs (x)
      s = abs (a2)
      t = abs (a1)
      u = abs (a0)

      y = z * max (s,z)           ! take maximum between |x^2|,|a2 * x|

      deflateCase = 1             ! up to now, the maximum is |x^3| or |a2 * x^2|

      if (y < t) then             ! check maximum between |x^2|,|a2 * x|,|a1|
          y = t * z               ! the maximum is |a1 * x|
          deflateCase = 2         ! up to now, the maximum is |a1 * x|
      else
          y = y * z               ! the maximum is |x^3| or |a2 * x^2|
      end if

      if (y < u) then             ! check maximum between |x^3|,|a2 * x^2|,|a1 * x|,|a0|
          deflateCase = 3         ! the maximum is |a0|
      end if

      y = x * k                   ! real root of original cubic

      select case (deflateCase)

      case (1)
        x = 1.0_wp / y
        t = - c0 * x              ! t -> backward deflation on unscaled cubic
        s = (t - c1) * x          ! s -> backward deflation on unscaled cubic
      case (2)
        s = c2 + y                ! s ->  forward deflation on unscaled cubic
        t = - c0 / y              ! t -> backward deflation on unscaled cubic
      case (3)
        s = c2 + y                ! s ->  forward deflation on unscaled cubic
        t = c1 + s * y            ! t ->  forward deflation on unscaled cubic
      end select

      if (doPrint) then
          write (*,wpformat) ' Residual quadratic q1 = ',s
          write (*,wpformat) ' Residual quadratic q0 = ',t
          write (*,wpformat) ' ------------------------------------------------'
      end if

      call quadraticRoots (s, t, nReal, root (1:2,1:2))

      if (nReal == 2) then

          x = root (1,Re)         ! real roots of quadratic are ordered x >= z
          z = root (2,Re)         ! use 'z', because 'y' is original cubic real root

          nReal = 3

          root (1,Re) = max (x, y)
          root (2,Re) = max (z, min (x, y))
          root (3,Re) = min (z, y)
          root (:,Im) = 0.0_wp

      else

          nReal = 1

          root (3,Re) = root (2,Re)
          root (2,Re) = root (1,Re)
          root (1,Re) = y
          root (3,Im) = root (2,Im)
          root (2,Im) = root (1,Im)
          root (1,Im) = 0.0_wp

      end if

  end select
!
!
!     ...Ready!
!
!
  return
end subroutine cubicRoots



!!-----------------------------------------------------------------------------------
!!
!! QUADRATIC POLYNOMIAL ROOT SOLVER
!!
!! SYNOPSIS
!!
!!  call quadraticRoots (real,    intent (in)  :: q1,
!!                       real,    intent (in)  :: q0,
!!                       integer, intent (out) :: nReal,
!!                       real,    intent (out) :: root (1:2,1:2))
!!
!! DESCRIPTION
!!
!!  Calculates all real + complex roots of the quadratic polynomial:
!!
!!                 x^2 + q1 * x + q0
!!
!!  The code checks internally, if rescaling of the coefficients is needed to
!!  avoid overflow.
!!
!!  The order of the roots is as follows:
!!
!!        1) For real roots, the order is according to their algebraic value
!!           on the number scale (largest positive first, largest negative last).
!!
!!        2) Since there can be only one complex conjugate pair root, no order
!!           is necessary.
!!
!! ARGUMENTS
!!
!!  q1         : coefficient of x term
!!  q0         : independent coefficient
!!  nReal      : number of real roots found
!!  root (n,1) : real part of n-th root
!!  root (n,2) : imaginary part of n-th root
!!
!! NOTES
!!
!!***

subroutine quadraticRoots (q1, q0, nReal, root)
  implicit none

  integer            , intent (out) :: nReal
  real    (kind = wp), intent (in)  :: q1, q0
  real    (kind = wp), intent (out) :: root (1:2,1:2)

  logical :: rescale

  real (kind = wp) :: a0, a1
  real (kind = wp) :: k, x, y, z

  integer, parameter :: Re = 1
  integer, parameter :: Im = 2

  real (kind = wp), parameter :: LPN     = huge (1.0_wp)   ! the (L)argest (P)ositive (N)umber
  real (kind = wp), parameter :: sqrtLPN = sqrt (LPN)      ! and the square root of it
!
!
!     ...Handle special cases.
!
!
  if (q0 == 0.0_wp .and. q1 == 0.0_wp) then

      nReal = 2

      root (:,Re) = 0.0_wp
      root (:,Im) = 0.0_wp

  else if (q0 == 0.0_wp) then

      nReal = 2

      root (1,Re) = max (0.0_wp, - q1)
      root (2,Re) = min (0.0_wp, - q1)
      root (:,Im) = 0.0_wp

  else if (q1 == 0.0_wp) then

      x = sqrt (abs (q0))

      if (q0 < 0.0_wp) then

          nReal = 2

          root (1,Re) = x
          root (2,Re) = - x
          root (:,Im) = 0.0_wp

      else

          nReal = 0

          root (:,Re) = 0.0_wp
          root (1,Im) = x
          root (2,Im) = - x

      end if

  else
!
!
!     ...The general case. Do rescaling, if either squaring of q1/2 or evaluation of
!        (q1/2)^2 - q0 will lead to overflow. This is better than to have the solver
!        crashed. Note, that rescaling might lead to loss of accuracy, so we only
!        invoke it when absolutely necessary.
!
!
      rescale = (q1 > sqrtLPN + sqrtLPN)     ! this detects overflow of (q1/2)^2

      if (.not.rescale) then
           x = q1 * 0.5                      ! we are sure here that x*x will not overflow
           rescale = (q0 < x * x - LPN)      ! this detects overflow of (q1/2)^2 - q0
      end if

      if (rescale) then

          x = abs (q1)
          y = sqrt (abs (q0))

          if (x > y) then
              k  = x
              z  = 1.0_wp / x
              a1 = sign (1.0_wp , q1)
              a0 = (q0 * z) * z
          else
              k  = y
              a1 = q1 / y
              a0 = sign (1.0_wp , q0)
          end if

      else
          a1 = q1
          a0 = q0
      end if
!
!
!     ...Determine the roots of the quadratic. Note, that either a1 or a0 might
!        have become equal to zero due to underflow. But both cannot be zero.
!
!
      x = a1 * 0.5_wp
      y = x * x - a0

      if (y >= 0.0_wp) then

          y = sqrt (y)

          if (x > 0.0_wp) then
              y = - x - y
          else
              y = - x + y
          end if

          if (rescale) then
              y = y * k                     ! very important to convert to original
              z = q0 / y                    ! root first, otherwise complete loss of
          else                              ! root due to possible a0 = 0 underflow
              z = a0 / y
          end if

          nReal = 2

          root (1,Re) = max (y,z)           ! 1st real root of x^2 + a1 * x + a0
          root (2,Re) = min (y,z)           ! 2nd real root of x^2 + a1 * x + a0
          root (:,Im) = 0.0_wp

      else

          y = sqrt (- y)

          nReal = 0

          root (1,Re) = - x
          root (2,Re) = - x
          root (1,Im) = y                   ! complex conjugate pair of roots
          root (2,Im) = - y                 ! of x^2 + a1 * x + a0

          if (rescale) then
              root = root * k
          end if

      end if

  end if
!
!
!     ...Ready!
!
!
  return
end subroutine quadraticRoots





!!-----------------------------------------------------------------------------------
!!
!! QUARTIC POLYNOMIAL ROOT SOLVER
!!
!! SYNOPSIS
!!
!!  call quarticRoots (real,              intent (in)  :: q3,
!!                     real,              intent (in)  :: q2,
!!                     real,              intent (in)  :: q1,
!!                     real,              intent (in)  :: q0,
!!                     integer,           intent (out) :: nReal,
!!                     real,              intent (out) :: root (1:4,1:2),
!!                     logical, optional, intent (in)  :: printInfo)
!!
!! DESCRIPTION
!!
!!  Calculates all real + complex roots of the quartic polynomial:
!!
!!                 x^4 + q3 * x^3 + q2 * x^2 + q1 * x + q0
!!
!!  An option for printing detailed info about the intermediate stages in solving
!!  the quartic is available. This enables a detailed check in case something went
!!  wrong and the roots obtained are not proper.
!!
!!  The quartic root solver can handle any size of quartic coefficients and there is
!!  no danger of overflow, due to proper rescaling of the quartic polynomial.
!!
!!  The order of the roots is as follows:
!!
!!        1) For real roots, the order is according to their algebraic value
!!           on the number scale (largest positive first, largest negative last).
!!
!!        2) For complex conjugate pair roots, the order is according to the
!!           algebraic value of their real parts (largest positive first). If
!!           the real parts are equal, the order is according to the algebraic
!!           value of their imaginary parts (largest first).
!!
!!        3) All real roots preceede the complex ones.
!!
!! ARGUMENTS
!!
!!  q3         : coefficient of x^3 term
!!  q2         : coefficient of x^2 term
!!  q1         : coefficient of x term
!!  q0         : independent coefficient
!!  nReal      : number of real roots found
!!  root (n,1) : real part of n-th root
!!  root (n,2) : imaginary part of n-th root
!!  printInfo  : if given and true, detailed info will be printed about intermediate stages
!!
!! NOTES
!!
!!***

subroutine quarticRoots (q3, q2, q1, q0, nReal, root, printInfo)
  
  implicit none

  logical, optional  , intent (in)  :: printInfo
  integer            , intent (out) :: nReal
  real    (kind = wp), intent (in)  :: q3, q2, q1, q0
  real    (kind = wp), intent (out) :: root (1:4,1:2)

  logical :: bisection
  logical :: converged
  logical :: doPrint
  logical :: iterate
  logical :: minimum
  logical :: notZero

  integer :: deflateCase
  integer :: oscillate
  integer :: quarticType

  integer, parameter :: Re = 1
  integer, parameter :: Im = 2

  integer, parameter :: biquadratic = 2
  integer, parameter :: cubic       = 3
  integer, parameter :: general     = 4

  real (kind = wp) :: a0, a1, a2, a3
  real (kind = wp) :: a, b, c, d, k, s, t, u, x, y, z

  real (kind = wp), parameter :: macheps = epsilon (1.0_wp)
  real (kind = wp), parameter :: third   = 1.0_wp / 3.0_wp
!
!
!     ...Start.
!
!
  if (present (printInfo)) then
      doPrint = printInfo
  else
      doPrint = .false.
  end if

  if (doPrint) then
      write (*,wpformat) ' initial quartic q3    = ',q3
      write (*,wpformat) ' initial quartic q2    = ',q2
      write (*,wpformat) ' initial quartic q1    = ',q1
      write (*,wpformat) ' initial quartic q0    = ',q0
      write (*,wpformat) ' ------------------------------------------------'
  end if
!
!
!     ...Handle special cases. Since the cubic solver handles all its
!        special cases by itself, we need to check only for two cases:
!
!            1) independent term is zero -> solve cubic and include
!               the zero root
!
!            2) the biquadratic case.
!
!
  if (q0 == 0.0_wp) then

      k  = 1.0_wp
      a3 = q3
      a2 = q2
      a1 = q1

      quarticType = cubic

  else if (q3 == 0.0_wp .and. q1 == 0.0_wp) then

      k  = 1.0_wp
      a2 = q2
      a0 = q0

      quarticType = biquadratic

  else
!
!
!     ...The general case. Rescale quartic polynomial, such that largest absolute coefficient
!        is (exactly!) equal to 1. Honor the presence of a special quartic case that might have
!        been obtained during the rescaling process (due to underflow in the coefficients).
!
!
      s = abs (q3)
      t = sqrt (abs (q2))
      u = abs (q1) ** third
      x = sqrt (sqrt (abs (q0)))
      y = max (s,t,u,x)

      if (y == s) then

          k  = 1.0_wp / s
          a3 = sign (1.0_wp , q3)
          a2 = (q2 * k) * k
          a1 = ((q1 * k) * k) * k
          a0 = (((q0 * k) * k) * k) * k

      else if (y == t) then

          k  = 1.0_wp / t
          a3 = q3 * k
          a2 = sign (1.0_wp , q2)
          a1 = ((q1 * k) * k) * k
          a0 = (((q0 * k) * k) * k) * k

      else if (y == u) then

          k  = 1.0_wp / u
          a3 = q3 * k
          a2 = (q2 * k) * k
          a1 = sign (1.0_wp , q1)
          a0 = (((q0 * k) * k) * k) * k

      else

          k  = 1.0_wp / x
          a3 = q3 * k
          a2 = (q2 * k) * k
          a1 = ((q1 * k) * k) * k
          a0 = sign (1.0_wp , q0)

      end if

      k = 1.0_wp / k

      if (doPrint) then
          write (*,wpformat) ' rescaling factor      = ',k
          write (*,wpformat) ' ------------------------------------------------'
          write (*,wpformat) ' rescaled quartic q3   = ',a3
          write (*,wpformat) ' rescaled quartic q2   = ',a2
          write (*,wpformat) ' rescaled quartic q1   = ',a1
          write (*,wpformat) ' rescaled quartic q0   = ',a0
          write (*,wpformat) ' ------------------------------------------------'
      end if

      if (a0 == 0.0_wp) then
          quarticType = cubic
      else if (a3 == 0.0_wp .and. a1 == 0.0_wp) then
          quarticType = biquadratic
      else
          quarticType = general
      end if

  end if
!
!
!     ...Select the case.
!
!        1) The quartic with independent term = 0 -> solve cubic and add a zero root.
!
!
  select case (quarticType)

    case (cubic)

      call cubicRoots (a3, a2, a1, nReal, root (1:3,1:2), printInfo)

      if (nReal == 3) then

          x = root (1,Re) * k       ! real roots of cubic are ordered x >= y >= z
          y = root (2,Re) * k
          z = root (3,Re) * k

          nReal = 4

          root (1,Re) = max (x, 0.0_wp)
          root (2,Re) = max (y, min (x, 0.0_wp))
          root (3,Re) = max (z, min (y, 0.0_wp))
          root (4,Re) = min (z, 0.0_wp)
          root (:,Im) = 0.0_wp

      else                          ! there is only one real cubic root here

          x = root (1,Re) * k

          nReal = 2

          root (4,Re) = root (3,Re) * k
          root (3,Re) = root (2,Re) * k
          root (2,Re) = min (x, 0.0_wp)
          root (1,Re) = max (x, 0.0_wp)

          root (4,Im) = root (3,Im) * k
          root (3,Im) = root (2,Im) * k
          root (2,Im) = 0.0_wp
          root (1,Im) = 0.0_wp

      end if
!
!
!     ...2) The quartic with x^3 and x terms = 0 -> solve biquadratic.
!
!
    case (biquadratic)

      call quadraticRoots (q2, q0, nReal, root (1:2,1:2))

      if (nReal == 2) then

          x = root (1,Re)         ! real roots of quadratic are ordered x >= y
          y = root (2,Re)

          if (y >= 0.0_wp) then

              x = sqrt (x) * k
              y = sqrt (y) * k

              nReal = 4

              root (1,Re) = x
              root (2,Re) = y
              root (3,Re) = - y
              root (4,Re) = - x
              root (:,Im) = 0.0_wp

          else if (x >= 0.0_wp .and. y < 0.0_wp) then

              x = sqrt (x)       * k
              y = sqrt (abs (y)) * k

              nReal = 2

              root (1,Re) = x
              root (2,Re) = - x
              root (3,Re) = 0.0_wp
              root (4,Re) = 0.0_wp
              root (1,Im) = 0.0_wp
              root (2,Im) = 0.0_wp
              root (3,Im) = y
              root (4,Im) = - y

          else if (x < 0.0_wp) then

              x = sqrt (abs (x)) * k
              y = sqrt (abs (y)) * k

              nReal = 0

              root (:,Re) = 0.0_wp
              root (1,Im) = y
              root (2,Im) = x
              root (3,Im) = - x
              root (4,Im) = - y

          end if

      else          ! complex conjugate pair biquadratic roots x +/- iy.
              
          x = root (1,Re) * 0.5_wp
          y = root (1,Im) * 0.5_wp
          z = sqrt (x * x + y * y)
          y = sqrt (z - x) * k
          x = sqrt (z + x) * k

          nReal = 0

          root (1,Re) = x
          root (2,Re) = x
          root (3,Re) = - x
          root (4,Re) = - x
          root (1,Im) = y
          root (2,Im) = - y
          root (3,Im) = y
          root (4,Im) = - y

      end if
!
!
!     ...3) The general quartic case. Search for stationary points. Set the first
!           derivative polynomial (cubic) equal to zero and find its roots.
!           Check, if any minimum point of Q(x) is below zero, in which case we
!           must have real roots for Q(x). Hunt down only the real root, which
!           will potentially converge fastest during Newton iterates. The remaining
!           roots will be determined by deflation Q(x) -> cubic.
!
!           The best roots for the Newton iterations are the two on the opposite
!           ends, i.e. those closest to the +2 and -2. Which of these two roots
!           to take, depends on the location of the Q(x) minima x = s and x = u,
!           with s > u. There are three cases:
!
!              1) both Q(s) and Q(u) < 0
!                 ----------------------
!
!                 The best root is the one that corresponds to the lowest of
!                 these minima. If Q(s) is lowest -> start Newton from +2
!                 downwards (or zero, if s < 0 and a0 > 0). If Q(u) is lowest
!                 -> start Newton from -2 upwards (or zero, if u > 0 and a0 > 0).
!
!              2) only Q(s) < 0
!                 -------------
!
!                 With both sides +2 and -2 possible as a Newton starting point,
!                 we have to avoid the area in the Q(x) graph, where inflection
!                 points are present. Solving Q''(x) = 0, leads to solutions
!                 x = -a3/4 +/- discriminant, i.e. they are centered around -a3/4.
!                 Since both inflection points must be either on the r.h.s or l.h.s.
!                 from x = s, a simple test where s is in relation to -a3/4 allows
!                 us to avoid the inflection point area.
!
!              3) only Q(u) < 0
!                 -------------
!
!                 Same of what has been said under 2) but with x = u.
!
!
    case (general)

      x = 0.75_wp * a3
      y = 0.50_wp * a2
      z = 0.25_wp * a1

      if (doPrint) then
          write (*,wpformat) ' dQ(x)/dx cubic c2     = ',x
          write (*,wpformat) ' dQ(x)/dx cubic c1     = ',y
          write (*,wpformat) ' dQ(x)/dx cubic c0     = ',z
          write (*,wpformat) ' ------------------------------------------------'
      end if

      call cubicRoots (x, y, z, nReal, root (1:3,1:2), printInfo)

      s = root (1,Re)        ! Q'(x) root s (real for sure)
      x = s + a3
      x = x * s + a2
      x = x * s + a1
      x = x * s + a0         ! Q(s)

      y = 1.0_wp             ! dual info: Q'(x) has more real roots, and if so, is Q(u) < 0 ? 

      if (nReal > 1) then
          u = root (3,Re)    ! Q'(x) root u
          y = u + a3
          y = y * u + a2
          y = y * u + a1
          y = y * u + a0     ! Q(u)
      end if

      if (doPrint) then
          write (*,wpformat) ' dQ(x)/dx root s       = ',s
          write (*,wpformat) ' Q(s)                  = ',x
          write (*,wpformat) ' dQ(x)/dx root u       = ',u
          write (*,wpformat) ' Q(u)                  = ',y
          write (*,wpformat) ' ------------------------------------------------'
      end if

      if (x < 0.0_wp .and. y < 0.0_wp) then

          if (x < y) then
              if (s < 0.0_wp) then
                  x = 1.0_wp - sign (1.0_wp,a0)
              else
                  x = 2.0_wp
              end if
          else
              if (u > 0.0_wp) then
                  x = - 1.0_wp + sign (1.0_wp,a0)
              else
                  x = - 2.0_wp
              end if
          end if

          nReal = 1

      else if (x < 0.0_wp) then

          if (s < - a3 * 0.25_wp) then
              if (s > 0.0_wp) then
                  x = - 1.0_wp + sign (1.0_wp,a0)
              else
                  x = - 2.0_wp
              end if
          else
              if (s < 0.0_wp) then
                  x = 1.0_wp - sign (1.0_wp,a0)
              else
                  x = 2.0_wp
              end if
          end if

          nReal = 1

      else if (y < 0.0_wp) then

          if (u < - a3 * 0.25_wp) then
              if (u > 0.0_wp) then
                  x = - 1.0_wp + sign (1.0_wp,a0)
              else
                  x = - 2.0_wp
              end if
          else
              if (u < 0.0_wp) then
                  x = 1.0_wp - sign (1.0_wp,a0)
              else
                  x = 2.0_wp
              end if
          end if

          nReal = 1
      else
          nReal = 0
      end if
!
!
!     ...Do all necessary Newton iterations. In case we have more than 2 oscillations,
!        exit the Newton iterations and switch to bisection. Note, that from the
!        definition of the Newton starting point, we always have Q(x) > 0 and Q'(x)
!        starts (-ve/+ve) for the (-2/+2) starting points and (increase/decrease) smoothly
!        and staying (< 0 / > 0). In practice, for extremely shallow Q(x) curves near the
!        root, the Newton procedure can overshoot slightly due to rounding errors when
!        approaching the root. The result are tiny oscillations around the root. If such
!        a situation happens, the Newton iterations are abandoned after 3 oscillations
!        and further location of the root is done using bisection starting with the
!        oscillation brackets.
!
!
      if (nReal > 0) then

          oscillate = 0
          bisection = .false.
          converged = .false.

          do while (.not.converged .and. .not.bisection)    ! Newton-Raphson iterates

             y = x + a3                                     !
             z = x + y                                      !
             y = y * x + a2                                 ! y = Q(x)
             z = z * x + y                                  !
             y = y * x + a1                                 ! z = Q'(x)
             z = z * x + y                                  !
             y = y * x + a0                                 !

             if (y < 0.0_wp) then                           ! does Newton start oscillating ?
                 oscillate = oscillate + 1                  ! increment oscillation counter
                 s = x                                      ! save lower bisection bound
             else
                 u = x                                      ! save upper bisection bound
             end if

             y = y / z                                      ! Newton correction
             x = x - y                                      ! new Newton root

             bisection = oscillate > 2                      ! activate bisection
             converged = abs (y) <= abs (x) * macheps       ! Newton convergence indicator

             if (doPrint) write (*,wpformat) ' Newton root           = ',x

          end do

          if (bisection) then

              t = u - s                                     ! initial bisection interval
              do while (abs (t) > abs (x) * macheps)        ! bisection iterates

                 y = x + a3                                 !
                 y = y * x + a2                             ! y = Q(x)
                 y = y * x + a1                             !
                 y = y * x + a0                             !

                 if (y < 0.0_wp) then                       !
                     s = x                                  !
                 else                                       ! keep bracket on root
                     u = x                                  !
                 end if                                     !

                 t = 0.5_wp * (u - s)                       ! new bisection interval
                 x = s + t                                  ! new bisection root

                 if (doPrint) write (*,wpformat) ' Bisection root        = ',x

              end do
          end if

          if (doPrint) write (*,wpformat) ' ------------------------------------------------'
!
!
!     ...Find remaining roots -> reduce to cubic. The reduction to a cubic polynomial
!        is done using composite deflation to minimize rounding errors. Also, while
!        the composite deflation analysis is done on the reduced quartic, the actual
!        deflation is being performed on the original quartic again to avoid enhanced
!        propagation of root errors.
!
!
          z = abs (x)            !
          a = abs (a3)           !
          b = abs (a2)           ! prepare for composite deflation
          c = abs (a1)           !
          d = abs (a0)           !

          y = z * max (a,z)      ! take maximum between |x^2|,|a3 * x|

          deflateCase = 1        ! up to now, the maximum is |x^4| or |a3 * x^3|

          if (y < b) then        ! check maximum between |x^2|,|a3 * x|,|a2|
              y = b * z          ! the maximum is |a2| -> form |a2 * x|
              deflateCase = 2    ! up to now, the maximum is |a2 * x^2|
          else
              y = y * z          ! the maximum is |x^3| or |a3 * x^2|
          end if

          if (y < c) then        ! check maximum between |x^3|,|a3 * x^2|,|a2 * x|,|a1|
              y = c * z          ! the maximum is |a1| -> form |a1 * x|
              deflateCase = 3    ! up to now, the maximum is |a1 * x|
          else
              y = y * z          ! the maximum is |x^4|,|a3 * x^3| or |a2 * x^2|
          end if

          if (y < d) then        ! check maximum between |x^4|,|a3 * x^3|,|a2 * x^2|,|a1 * x|,|a0|
              deflateCase = 4    ! the maximum is |a0|
          end if

          x = x * k              ! 1st real root of original Q(x)

          select case (deflateCase)

          case (1)
            z = 1.0_wp / x
            u = - q0 * z         ! u -> backward deflation on original Q(x)
            t = (u - q1) * z     ! t -> backward deflation on original Q(x)
            s = (t - q2) * z     ! s -> backward deflation on original Q(x)
          case (2)
            z = 1.0_wp / x
            u = - q0 * z         ! u -> backward deflation on original Q(x)
            t = (u - q1) * z     ! t -> backward deflation on original Q(x)
            s = q3 + x           ! s ->  forward deflation on original Q(x)
          case (3)
            s = q3 + x           ! s ->  forward deflation on original Q(x)
            t = q2 + s * x       ! t ->  forward deflation on original Q(x)
            u = - q0 / x         ! u -> backward deflation on original Q(x)
          case (4)
            s = q3 + x           ! s ->  forward deflation on original Q(x)
            t = q2 + s * x       ! t ->  forward deflation on original Q(x)
            u = q1 + t * x       ! u ->  forward deflation on original Q(x)
          end select

          if (doPrint) then
              write (*,wpformat) ' Residual cubic c2     = ',s
              write (*,wpformat) ' Residual cubic c1     = ',t
              write (*,wpformat) ' Residual cubic c0     = ',u
              write (*,wpformat) ' ------------------------------------------------'
          end if

          call cubicRoots (s, t, u, nReal, root (1:3,1:2), printInfo)

          if (nReal == 3) then

              s = root (1,Re)    !
              t = root (2,Re)    ! real roots of cubic are ordered s >= t >= u
              u = root (3,Re)    !

              root (1,Re) = max (s, x)
              root (2,Re) = max (t, min (s, x))
              root (3,Re) = max (u, min (t, x))
              root (4,Re) = min (u, x)
              root (:,Im) = 0.0_wp

              nReal = 4

          else                   ! there is only one real cubic root here

              s = root (1,Re)

              root (4,Re) = root (3,Re)
              root (3,Re) = root (2,Re)
              root (2,Re) = min (s, x)
              root (1,Re) = max (s, x)
              root (4,Im) = root (3,Im)
              root (3,Im) = root (2,Im)
              root (2,Im) = 0.0_wp
              root (1,Im) = 0.0_wp

              nReal = 2

          end if

      else
!
!
!     ...If no real roots have been found by now, only complex roots are possible.
!        Find real parts of roots first, followed by imaginary components.
!
!
          s = a3 * 0.5_wp
          t =  s * s - a2
          u =  s * t + a1                   ! value of Q'(-a3/4) at stationary point -a3/4

          notZero = (abs (u) >= macheps)    ! H(-a3/4) is considered > 0 at stationary point

          if (doPrint) then
              write (*,wpformat) ' dQ/dx (-a3/4) value   = ',u
              write (*,wpformat) ' ------------------------------------------------'
          end if

          if (a3 /= 0.0_wp) then
              s = a1 / a3
              minimum = (a0 > s * s)                            ! H''(-a3/4) > 0 -> minimum
          else
              minimum = (4 * a0 > a2 * a2)                      ! H''(-a3/4) > 0 -> minimum
          end if

          iterate = notZero .or. (.not.notZero .and. minimum)

          if (iterate) then

              x = sign (2.0_wp,a3)                              ! initial root -> target = smaller mag root

              oscillate = 0
              bisection = .false.
              converged = .false.

              do while (.not.converged .and. .not.bisection)    ! Newton-Raphson iterates

                 a = x + a3                                     !
                 b = x + a                                      ! a = Q(x)
                 c = x + b                                      !
                 d = x + c                                      ! b = Q'(x)
                 a = a * x + a2                                 !
                 b = b * x + a                                  ! c = Q''(x) / 2
                 c = c * x + b                                  !
                 a = a * x + a1                                 ! d = Q'''(x) / 6
                 b = b * x + a                                  !
                 a = a * x + a0                                 !
                 y = a * d * d - b * c * d + b * b              ! y = H(x), usually < 0
                 z = 2 * d * (4 * a - b * d - c * c)            ! z = H'(x)

                 if (y > 0.0_wp) then                           ! does Newton start oscillating ?
                     oscillate = oscillate + 1                  ! increment oscillation counter
                     s = x                                      ! save upper bisection bound
                 else
                     u = x                                      ! save lower bisection bound
                 end if

                 y = y / z                                      ! Newton correction
                 x = x - y                                      ! new Newton root

                 bisection = oscillate > 2                      ! activate bisection
                 converged = abs (y) <= abs (x) * macheps       ! Newton convergence criterion

                 if (doPrint) write (*,wpformat) ' Newton H(x) root      = ',x

              end do

              if (bisection) then

                  t = u - s                                     ! initial bisection interval
                  do while (abs (t) > abs (x * macheps))        ! bisection iterates

                     a = x + a3                                 !
                     b = x + a                                  ! a = Q(x)
                     c = x + b                                  !
                     d = x + c                                  ! b = Q'(x)
                     a = a * x + a2                             !
                     b = b * x + a                              ! c = Q''(x) / 2
                     c = c * x + b                              !
                     a = a * x + a1                             ! d = Q'''(x) / 6
                     b = b * x + a                              !
                     a = a * x + a0                             !
                     y = a * d * d - b * c * d + b * b          ! y = H(x)

                     if (y > 0.0_wp) then                       !
                         s = x                                  !
                     else                                       ! keep bracket on root
                         u = x                                  !
                     end if                                     !

                     t = 0.5_wp * (u - s)                       ! new bisection interval
                     x = s + t                                  ! new bisection root

                     if (doPrint) write (*,wpformat) ' Bisection H(x) root   = ',x

                  end do
              end if

              if (doPrint) write (*,wpformat) ' ------------------------------------------------'

              a = x * k                                         ! 1st real component -> a
              b = - 0.5_wp * q3 - a                             ! 2nd real component -> b

              x = 4 * a + q3                                    ! Q'''(a)
              y = x + q3 + q3                                   !
              y = y * a + q2 + q2                               ! Q'(a)
              y = y * a + q1                                    !
              y = max (y / x, 0.0_wp)                           ! ensure Q'(a) / Q'''(a) >= 0
              x = 4 * b + q3                                    ! Q'''(b)
              z = x + q3 + q3                                   !
              z = z * b + q2 + q2                               ! Q'(b)
              z = z * b + q1                                    !
              z = max (z / x, 0.0_wp)                           ! ensure Q'(b) / Q'''(b) >= 0
              c = a * a                                         ! store a^2 for later
              d = b * b                                         ! store b^2 for later
              s = c + y                                         ! magnitude^2 of (a + iy) root
              t = d + z                                         ! magnitude^2 of (b + iz) root

              if (s > t) then                                   ! minimize imaginary error
                  c = sqrt (y)                                  ! 1st imaginary component -> c
                  d = sqrt (q0 / s - d)                         ! 2nd imaginary component -> d
              else
                  c = sqrt (q0 / t - c)                         ! 1st imaginary component -> c
                  d = sqrt (z)                                  ! 2nd imaginary component -> d
              end if

          else                                                  ! no bisection -> real components equal

              a = - 0.25_wp * q3                                ! 1st real component -> a
              b = a                                             ! 2nd real component -> b = a

              x = a + q3                                        !
              x = x * a + q2                                    ! Q(a)
              x = x * a + q1                                    !
              x = x * a + q0                                    !
              y = - 0.1875_wp * q3 * q3 + 0.5_wp * q2           ! Q''(a) / 2
              z = max (y * y - x, 0.0_wp)                       ! force discriminant to be >= 0
              z = sqrt (z)                                      ! square root of discriminant
              y = y + sign (z,y)                                ! larger magnitude root
              x = x / y                                         ! smaller magnitude root
              c = max (y, 0.0_wp)                               ! ensure root of biquadratic > 0
              d = max (x, 0.0_wp)                               ! ensure root of biquadratic > 0
              c = sqrt (c)                                      ! large magnitude imaginary component
              d = sqrt (d)                                      ! small magnitude imaginary component

          end if

          if (a > b) then

              root (1,Re) = a
              root (2,Re) = a
              root (3,Re) = b
              root (4,Re) = b
              root (1,Im) = c
              root (2,Im) = - c
              root (3,Im) = d
              root (4,Im) = - d

          else if (a < b) then

              root (1,Re) = b
              root (2,Re) = b
              root (3,Re) = a
              root (4,Re) = a
              root (1,Im) = d
              root (2,Im) = - d
              root (3,Im) = c
              root (4,Im) = - c

          else

              root (1,Re) = a
              root (2,Re) = a
              root (3,Re) = a
              root (4,Re) = a
              root (1,Im) = c
              root (2,Im) = - c
              root (3,Im) = d
              root (4,Im) = - d

          end if

      end if    ! # of real roots 'if'

  end select    ! quartic type select
!
!
!     ...Ready!
!
!
  return
end subroutine quarticRoots

end module Polynomial234RootSolvers
