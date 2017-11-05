program PolynomialRootsCompare
!
!
!   ...This program solves a set of 8 cubic and 13 quartic polynomials using
!      the cubic and quartic solvers as described in the manuscript.
!
!  
  use Polynomial234RootSolvers, ONLY : cubicRoots, quarticRoots
  use JenkinsTraubSolver,       ONLY : rpoly
  use SetWorkingPrecision,      ONLY : wp, wpformat, dpKind, qpKind

  implicit none

  logical :: fail
  logical :: printInfo = .false.   ! change this to true, if detailed info wanted

  integer :: i
  integer :: nReal

  integer, parameter :: Re = 1
  integer, parameter :: Im = 2

  real (kind = wp) :: A,B,C,D
  real (kind = wp) :: d1r,d2r,d3r,d4r,drMax
  real (kind = wp) :: d1i,d3i,diMax

  real (kind = wp) :: root              (1:4,1:2)
  real (kind = wp) :: exactCubicRoot    (1:3,1:2,1:9)
  real (kind = wp) :: exactQuarticRoot  (1:4,1:2,1:13)
  real (kind = wp) :: NSolveCubicRoot   (1:3,1:2,1:9)
  real (kind = wp) :: NSolveQuarticRoot (1:4,1:2,1:13)

  real (kind = wp) :: p (101), zr (100), zi (100)       ! Jenkins-Traub arrays

  real (kind = wp) :: cubicA (1:9)
  real (kind = wp) :: cubicB (1:9)
  real (kind = wp) :: cubicC (1:9)

  real (kind = wp) :: quarticA (1:13)
  real (kind = wp) :: quarticB (1:13)
  real (kind = wp) :: quarticC (1:13)
  real (kind = wp) :: quarticD (1:13)
!
!
!   ...Set the exact cubic roots.
!
!  
  exactCubicRoot (1,1,1) = 1.0e+14_wp
  exactCubicRoot (2,1,1) = 1.0e+7_wp
  exactCubicRoot (3,1,1) = 1.0_wp
  exactCubicRoot (1,2,1) = 0.0_wp
  exactCubicRoot (2,2,1) = 0.0_wp
  exactCubicRoot (3,2,1) = 0.0_wp

  exactCubicRoot (1,1,2) = 1.000002_wp
  exactCubicRoot (2,1,2) = 1.000001_wp
  exactCubicRoot (3,1,2) = 1.000000_wp
  exactCubicRoot (1,2,2) = 0.0_wp
  exactCubicRoot (2,2,2) = 0.0_wp
  exactCubicRoot (3,2,2) = 0.0_wp

  exactCubicRoot (1,1,3) = 1.0e+80_wp
  exactCubicRoot (2,1,3) = 1.0e+77_wp
  exactCubicRoot (3,1,3) = -1.0e+81_wp
  exactCubicRoot (1,2,3) = 0.0_wp
  exactCubicRoot (2,2,3) = 0.0_wp
  exactCubicRoot (3,2,3) = 0.0_wp

  exactCubicRoot (1,1,4) = 1.0_wp
  exactCubicRoot (2,1,4) = -1.0_wp
  exactCubicRoot (3,1,4) = -1.0e+24_wp
  exactCubicRoot (1,2,4) = 0.0_wp
  exactCubicRoot (2,2,4) = 0.0_wp
  exactCubicRoot (3,2,4) = 0.0_wp

  exactCubicRoot (1,1,5) = 1.0e+14_wp
  exactCubicRoot (2,1,5) = 1.0e+14_wp
  exactCubicRoot (3,1,5) = -1.0_wp
  exactCubicRoot (1,2,5) = 0.0_wp
  exactCubicRoot (2,2,5) = 0.0_wp
  exactCubicRoot (3,2,5) = 0.0_wp

  exactCubicRoot (1,1,6) = 1.0e+5_wp
  exactCubicRoot (2,1,6) = 1.0e+5_wp
  exactCubicRoot (3,1,6) = 1.0e+5_wp
  exactCubicRoot (1,2,6) = 0.0_wp
  exactCubicRoot (2,2,6) = 1.0_wp
  exactCubicRoot (3,2,6) = -1.0_wp

  exactCubicRoot (1,1,7) = 1.0_wp
  exactCubicRoot (2,1,7) = 1.0_wp
  exactCubicRoot (3,1,7) = 1.0_wp
  exactCubicRoot (1,2,7) = 0.0_wp
  exactCubicRoot (2,2,7) = 1.0e+7_wp
  exactCubicRoot (3,2,7) = -1.0e+7_wp

  exactCubicRoot (1,1,8) = 1.0_wp
  exactCubicRoot (2,1,8) = 1.0e+7_wp
  exactCubicRoot (3,1,8) = 1.0e+7_wp
  exactCubicRoot (1,2,8) = 0.0_wp
  exactCubicRoot (2,2,8) = 1.0_wp
  exactCubicRoot (3,2,8) = -1.0_wp

  exactCubicRoot (1,1,9) = -1.0e+14_wp
  exactCubicRoot (2,1,9) = 1.0_wp
  exactCubicRoot (3,1,9) = 1.0_wp
  exactCubicRoot (1,2,9) = 0.0_wp
  exactCubicRoot (2,2,9) = 1.0_wp
  exactCubicRoot (3,2,9) = -1.0_wp
!
!
!   ...Set the exact quartic roots.
!
!  
  exactQuarticRoot (1,1,1) = 1.0e+9_wp
  exactQuarticRoot (2,1,1) = 1.0e+6_wp
  exactQuarticRoot (3,1,1) = 1.0e+3_wp
  exactQuarticRoot (4,1,1) = 1.0_wp
  exactQuarticRoot (1,2,1) = 0.0_wp
  exactQuarticRoot (2,2,1) = 0.0_wp
  exactQuarticRoot (3,2,1) = 0.0_wp
  exactQuarticRoot (4,2,1) = 0.0_wp

  exactQuarticRoot (1,1,2) = 1.003_wp
  exactQuarticRoot (2,1,2) = 1.002_wp
  exactQuarticRoot (3,1,2) = 1.001_wp
  exactQuarticRoot (4,1,2) = 1.000_wp
  exactQuarticRoot (1,2,2) = 0.0_wp
  exactQuarticRoot (2,2,2) = 0.0_wp
  exactQuarticRoot (3,2,2) = 0.0_wp
  exactQuarticRoot (4,2,2) = 0.0_wp

  exactQuarticRoot (1,1,3) = 1.0e+80_wp
  exactQuarticRoot (2,1,3) = -1.0e+74_wp
  exactQuarticRoot (3,1,3) = -1.0e+76_wp
  exactQuarticRoot (4,1,3) = -1.0e+77_wp
  exactQuarticRoot (1,2,3) = 0.0_wp
  exactQuarticRoot (2,2,3) = 0.0_wp
  exactQuarticRoot (3,2,3) = 0.0_wp
  exactQuarticRoot (4,2,3) = 0.0_wp

  exactQuarticRoot (1,1,4) = 1.0e+14_wp
  exactQuarticRoot (2,1,4) = 2.0_wp
  exactQuarticRoot (3,1,4) = 1.0_wp
  exactQuarticRoot (4,1,4) = -1.0_wp
  exactQuarticRoot (1,2,4) = 0.0_wp
  exactQuarticRoot (2,2,4) = 0.0_wp
  exactQuarticRoot (3,2,4) = 0.0_wp
  exactQuarticRoot (4,2,4) = 0.0_wp

  exactQuarticRoot (1,1,5) = 1.0e+7_wp
  exactQuarticRoot (2,1,5) = 1.0_wp
  exactQuarticRoot (3,1,5) = -1.0_wp
  exactQuarticRoot (4,1,5) = -2.0e+7_wp
  exactQuarticRoot (1,2,5) = 0.0_wp
  exactQuarticRoot (2,2,5) = 0.0_wp
  exactQuarticRoot (3,2,5) = 0.0_wp
  exactQuarticRoot (4,2,5) = 0.0_wp

  exactQuarticRoot (1,1,6) = 1.0e+7_wp
  exactQuarticRoot (2,1,6) = -1.0e+6_wp
  exactQuarticRoot (3,1,6) = 1.0_wp
  exactQuarticRoot (4,1,6) = 1.0_wp
  exactQuarticRoot (1,2,6) = 0.0_wp
  exactQuarticRoot (2,2,6) = 0.0_wp
  exactQuarticRoot (3,2,6) = 1.0_wp
  exactQuarticRoot (4,2,6) = -1.0_wp

  exactQuarticRoot (1,1,7) = -4.0_wp
  exactQuarticRoot (2,1,7) = -7.0_wp
  exactQuarticRoot (3,1,7) = -1.0e+6_wp
  exactQuarticRoot (4,1,7) = -1.0e+6_wp
  exactQuarticRoot (1,2,7) = 0.0_wp
  exactQuarticRoot (2,2,7) = 0.0_wp
  exactQuarticRoot (3,2,7) = 1.0e+5_wp
  exactQuarticRoot (4,2,7) = -1.0e+5_wp

  exactQuarticRoot (1,1,8) = 1.0e+8_wp
  exactQuarticRoot (2,1,8) = 11.0_wp
  exactQuarticRoot (3,1,8) = 1.0e+3_wp
  exactQuarticRoot (4,1,8) = 1.0e+3_wp
  exactQuarticRoot (1,2,8) = 0.0_wp
  exactQuarticRoot (2,2,8) = 0.0_wp
  exactQuarticRoot (3,2,8) = 1.0_wp
  exactQuarticRoot (4,2,8) = -1.0_wp

  exactQuarticRoot (1,1,9) = 1.0e+7_wp
  exactQuarticRoot (2,1,9) = 1.0e+7_wp
  exactQuarticRoot (3,1,9) = 1.0_wp
  exactQuarticRoot (4,1,9) = 1.0_wp
  exactQuarticRoot (1,2,9) = 1.0e+6_wp
  exactQuarticRoot (2,2,9) = -1.0e+6_wp
  exactQuarticRoot (3,2,9) = 2.0_wp
  exactQuarticRoot (4,2,9) = -2.0_wp

  exactQuarticRoot (1,1,10) = 1.0e+4_wp
  exactQuarticRoot (2,1,10) = 1.0e+4_wp
  exactQuarticRoot (3,1,10) = -7.0_wp
  exactQuarticRoot (4,1,10) = -7.0_wp
  exactQuarticRoot (1,2,10) = 3.0_wp
  exactQuarticRoot (2,2,10) = -3.0_wp
  exactQuarticRoot (3,2,10) = 1.0e+3_wp
  exactQuarticRoot (4,2,10) = -1.0e+3_wp

  exactQuarticRoot (1,1,11) = 1.002_wp
  exactQuarticRoot (2,1,11) = 1.002_wp
  exactQuarticRoot (3,1,11) = 1.001_wp
  exactQuarticRoot (4,1,11) = 1.001_wp
  exactQuarticRoot (1,2,11) = 4.998_wp
  exactQuarticRoot (2,2,11) = -4.998_wp
  exactQuarticRoot (3,2,11) = 5.001_wp
  exactQuarticRoot (4,2,11) = -5.001_wp

  exactQuarticRoot (1,1,12) = 1.0e+3_wp
  exactQuarticRoot (2,1,12) = 1.0e+3_wp
  exactQuarticRoot (3,1,12) = 1.0e+3_wp
  exactQuarticRoot (4,1,12) = 1.0e+3_wp
  exactQuarticRoot (1,2,12) = 3.0_wp
  exactQuarticRoot (2,2,12) = -3.0_wp
  exactQuarticRoot (3,2,12) = 1.0_wp
  exactQuarticRoot (4,2,12) = -1.0_wp

  exactQuarticRoot (1,1,13) = 2.0_wp
  exactQuarticRoot (2,1,13) = 2.0_wp
  exactQuarticRoot (3,1,13) = 1.0_wp
  exactQuarticRoot (4,1,13) = 1.0_wp
  exactQuarticRoot (1,2,13) = 1.0e+4_wp
  exactQuarticRoot (2,2,13) = -1.0e+4_wp
  exactQuarticRoot (3,2,13) = 1.0e+3_wp
  exactQuarticRoot (4,2,13) = -1.0e+3_wp
!
!
!   ...Set the Mathematica NSolve cubic roots.
!
!  
  if (wp == dpKind) then

  NSolveCubicRoot (1,1,1) = 1.0000000000000000e+14_wp
  NSolveCubicRoot (2,1,1) = 1.0000000000000000e+7_wp
  NSolveCubicRoot (3,1,1) = 1.0000000000000000_wp
  NSolveCubicRoot (1,2,1) = 0.0_wp
  NSolveCubicRoot (2,2,1) = 0.0_wp
  NSolveCubicRoot (3,2,1) = 0.0_wp

  NSolveCubicRoot (1,1,2) = 1.0000010000000001_wp
  NSolveCubicRoot (2,1,2) = 1.0000010000000001_wp
  NSolveCubicRoot (3,1,2) = 1.0000010000000001_wp
  NSolveCubicRoot (1,2,2) = 0.0_wp
  NSolveCubicRoot (2,2,2) = 0.0_wp
  NSolveCubicRoot (3,2,2) = 0.0_wp

  NSolveCubicRoot (1,1,3) = 9.99999999999999745852577533e+79_wp
  NSolveCubicRoot (2,1,3) = 1.000000000000000285525691e+77_wp
  NSolveCubicRoot (3,1,3) = -1.0000000000000000520161612e+81_wp
  NSolveCubicRoot (1,2,3) = 0.0_wp
  NSolveCubicRoot (2,2,3) = 0.0_wp
  NSolveCubicRoot (3,2,3) = 0.0_wp

  NSolveCubicRoot (1,1,4) = 1.0000000000000001_wp
  NSolveCubicRoot (2,1,4) = -1.0000000000000001_wp
  NSolveCubicRoot (3,1,4) = -1.000000000000000e+24_wp
  NSolveCubicRoot (1,2,4) = 0.0_wp
  NSolveCubicRoot (2,2,4) = 0.0_wp
  NSolveCubicRoot (3,2,4) = 0.0_wp

  NSolveCubicRoot (1,1,5) = 1.000000000000000e+14_wp
  NSolveCubicRoot (2,1,5) = 1.000000000000000e+14_wp
  NSolveCubicRoot (3,1,5) = -1.0000000000000002819362008982_wp
  NSolveCubicRoot (1,2,5) = 0.0_wp
  NSolveCubicRoot (2,2,5) = 0.0_wp
  NSolveCubicRoot (3,2,5) = 0.0_wp

  NSolveCubicRoot (1,1,6) = 1.0000000000000003e+5_wp
  NSolveCubicRoot (2,1,6) = 1.0000000000000003e+5_wp
  NSolveCubicRoot (3,1,6) = 1.0000000000000003e+5_wp
  NSolveCubicRoot (1,2,6) = 0.0_wp
  NSolveCubicRoot (2,2,6) = 0.0_wp
  NSolveCubicRoot (3,2,6) = 0.0_wp

  NSolveCubicRoot (1,1,7) = 0.999999999999999824865906_wp
  NSolveCubicRoot (2,1,7) = 1.000000001564227064228870_wp
  NSolveCubicRoot (3,1,7) = 1.000000001564227064228870_wp
  NSolveCubicRoot (1,2,7) = 0.0_wp
  NSolveCubicRoot (2,2,7) = 1.000000000000000014347301e+7_wp
  NSolveCubicRoot (3,2,7) = -1.000000000000000014347301e+7_wp

  NSolveCubicRoot (1,1,8) = 0.9999999999999998644747_wp
  NSolveCubicRoot (2,1,8) = 1.00000000307502287028000e+7_wp
  NSolveCubicRoot (3,1,8) = 1.00000000307502287028000e+7_wp
  NSolveCubicRoot (1,2,8) = 0.0_wp
  NSolveCubicRoot (2,2,8) = 1.0333923459768027613243576_wp
  NSolveCubicRoot (3,2,8) = -1.0333923459768027613243576_wp

  NSolveCubicRoot (1,1,9) = -0.99999999999999986648559570e+14_wp
  NSolveCubicRoot (2,1,9) = 0.999999999999999986122212192_wp
  NSolveCubicRoot (3,1,9) = 0.999999999999999986122212192_wp
  NSolveCubicRoot (1,2,9) = 0.0_wp
  NSolveCubicRoot (2,2,9) = 1.0000000000000001842601592_wp
  NSolveCubicRoot (3,2,9) = -1.0000000000000001842601592_wp

  else if (wp == qpKind) then

  NSolveCubicRoot (1,1,1) = 1.000000000000000000000000000000000e+14_wp
  NSolveCubicRoot (2,1,1) = 1.000000000000000000000000000000000e+7_wp
  NSolveCubicRoot (3,1,1) = 1.000000000000000000000000000000000_wp
  NSolveCubicRoot (1,2,1) = 0.0_wp
  NSolveCubicRoot (2,2,1) = 0.0_wp
  NSolveCubicRoot (3,2,1) = 0.0_wp

  NSolveCubicRoot (1,1,2) = 1.000001000001_wp
  NSolveCubicRoot (2,1,2) = 1.000001000001_wp
  NSolveCubicRoot (3,1,2) = 1.000001000001_wp
  NSolveCubicRoot (1,2,2) = 0.0_wp
  NSolveCubicRoot (2,2,2) = 0.0_wp
  NSolveCubicRoot (3,2,2) = 0.0_wp

  NSolveCubicRoot (1,1,3) = 1.0000000000000000000000000000000001e+80_wp
  NSolveCubicRoot (2,1,3) = 1.0000000000000000000000000000000001e+77_wp
  NSolveCubicRoot (3,1,3) = -1.00000000000000000000000000000000001e+81_wp
  NSolveCubicRoot (1,2,3) = 0.0_wp
  NSolveCubicRoot (2,2,3) = 0.0_wp
  NSolveCubicRoot (3,2,3) = 0.0_wp

  NSolveCubicRoot (1,1,4) = 1.00000000000000000000000000000000001_wp
  NSolveCubicRoot (2,1,4) = -1.00000000000000000000000000000000001_wp
  NSolveCubicRoot (3,1,4) = -1.00000000000000000000000000000000001e+24_wp
  NSolveCubicRoot (1,2,4) = 0.0_wp
  NSolveCubicRoot (2,2,4) = 0.0_wp
  NSolveCubicRoot (3,2,4) = 0.0_wp

  NSolveCubicRoot (1,1,5) = 1.0000000000000000000000000000000000e+14_wp
  NSolveCubicRoot (2,1,5) = 1.0000000000000000000000000000000000e+14_wp
  NSolveCubicRoot (3,1,5) = -1.0000000000000000000000000000000000_wp
  NSolveCubicRoot (1,2,5) = 0.0_wp
  NSolveCubicRoot (2,2,5) = 0.0_wp
  NSolveCubicRoot (3,2,5) = 0.0_wp

  NSolveCubicRoot (1,1,6) = 1.0000000000000001e+5_wp
  NSolveCubicRoot (2,1,6) = 1.0000000000000001e+5_wp
  NSolveCubicRoot (3,1,6) = 1.0000000000000001e+5_wp
  NSolveCubicRoot (1,2,6) = 0.0_wp
  NSolveCubicRoot (2,2,6) = 0.0_wp
  NSolveCubicRoot (3,2,6) = 0.0_wp

  NSolveCubicRoot (1,1,7) = 1.0000000000000000000000000000000001_wp
  NSolveCubicRoot (2,1,7) = 1.000000000000000000000000001_wp
  NSolveCubicRoot (3,1,7) = 1.000000000000000000000000001_wp
  NSolveCubicRoot (1,2,7) = 0.0_wp
  NSolveCubicRoot (2,2,7) = 1.0000000000000000000000000000000001e+7_wp
  NSolveCubicRoot (3,2,7) = -1.000000000000000000000000000000001e+7_wp

  NSolveCubicRoot (1,1,8) = 1.0000000000000000000000000000000001_wp
  NSolveCubicRoot (2,1,8) = 1.000000000000000000000000001e+7_wp
  NSolveCubicRoot (3,1,8) = 1.000000000000000000000000001e+7_wp
  NSolveCubicRoot (1,2,8) = 0.0_wp
  NSolveCubicRoot (2,2,8) = 1.00000000000000000001_wp
  NSolveCubicRoot (3,2,8) = -1.00000000000000000001_wp

  NSolveCubicRoot (1,1,9) = -1.0000000000000000000000000000000001e+14_wp
  NSolveCubicRoot (2,1,9) = 1.000000000000000000000000000000001_wp
  NSolveCubicRoot (3,1,9) = 1.000000000000000000000000000000001_wp
  NSolveCubicRoot (1,2,9) = 0.0_wp
  NSolveCubicRoot (2,2,9) = 1.000000000000000000000000000000001_wp
  NSolveCubicRoot (3,2,9) = -1.000000000000000000000000000000001_wp

  end if
!
!
!   ...Set the Mathematica NSolve quartic roots.
!
!  
  if (wp == dpKind) then

  NSolveQuarticRoot (1,1,1) = 1.0000000000000000471336534e+9_wp
  NSolveQuarticRoot (2,1,1) = 9.99999999999999875466727417e+5_wp
  NSolveQuarticRoot (3,1,1) = 9.99999999999999970259900727e+2_wp
  NSolveQuarticRoot (4,1,1) = 1.000000000000000091974225_wp
  NSolveQuarticRoot (1,2,1) = 0.0_wp
  NSolveQuarticRoot (2,2,1) = 0.0_wp
  NSolveQuarticRoot (3,2,1) = 0.0_wp
  NSolveQuarticRoot (4,2,1) = 0.0_wp

  NSolveQuarticRoot (1,1,2) = 1.0029996719053184683900781237_wp
  NSolveQuarticRoot (2,1,2) = 1.00200012440260177726258916_wp
  NSolveQuarticRoot (3,1,2) = 1.00099956089539365855500818_wp
  NSolveQuarticRoot (4,1,2) = 1.000000141818917054337134686_wp
  NSolveQuarticRoot (1,2,2) = 0.0_wp
  NSolveQuarticRoot (2,2,2) = 0.0_wp
  NSolveQuarticRoot (3,2,2) = 0.0_wp
  NSolveQuarticRoot (4,2,2) = 0.0_wp

  NSolveQuarticRoot (1,1,3) = 9.9999999999999982973851428976e+79_wp
  NSolveQuarticRoot (2,1,3) = -9.99999999999999818399224428496e+73_wp
  NSolveQuarticRoot (3,1,3) = -9.99999999999999959847695684840e+75_wp
  NSolveQuarticRoot (4,1,3) = -1.00000000000000005349633e+77_wp
  NSolveQuarticRoot (1,2,3) = 0.0_wp
  NSolveQuarticRoot (2,2,3) = 0.0_wp
  NSolveQuarticRoot (3,2,3) = 0.0_wp
  NSolveQuarticRoot (4,2,3) = 0.0_wp

  NSolveQuarticRoot (1,1,4) = 1.000000000000000147209167e+14_wp
  NSolveQuarticRoot (2,1,4) = 1.9999999999999997615839422704_wp
  NSolveQuarticRoot (3,1,4) = 0.9999999999999998807919711352187_wp
  NSolveQuarticRoot (4,1,4) = -1.0000000000000000846761896_wp
  NSolveQuarticRoot (1,2,4) = 0.0_wp
  NSolveQuarticRoot (2,2,4) = 0.0_wp
  NSolveQuarticRoot (3,2,4) = 0.0_wp
  NSolveQuarticRoot (4,2,4) = 0.0_wp

  NSolveQuarticRoot (1,1,5) = 1.00000000000000008003553375e+7_wp
  NSolveQuarticRoot (2,1,5) = 0.9999999999999999861222121921855_wp
  NSolveQuarticRoot (3,1,5) = -0.99999999999999998612221219218_wp
  NSolveQuarticRoot (4,1,5) = -1.9999999999999998472048901e+7_wp
  NSolveQuarticRoot (1,2,5) = 0.0_wp
  NSolveQuarticRoot (2,2,5) = 0.0_wp
  NSolveQuarticRoot (3,2,5) = 0.0_wp
  NSolveQuarticRoot (4,2,5) = 0.0_wp

  NSolveQuarticRoot (1,1,6) = 1.00000000000000008776623872108757e+7_wp
  NSolveQuarticRoot (2,1,6) = -1.00000000000000013585577107733115e+6_wp
  NSolveQuarticRoot (3,1,6) = 0.99999999999999995278299538825_wp
  NSolveQuarticRoot (4,1,6) = 0.99999999999999995278299538825_wp
  NSolveQuarticRoot (1,2,6) = 0.0_wp
  NSolveQuarticRoot (2,2,6) = 0.0_wp
  NSolveQuarticRoot (3,2,6) = 0.999999999999999952782995388256_wp
  NSolveQuarticRoot (4,2,6) = -0.999999999999999952782995388256_wp

  NSolveQuarticRoot (1,1,7) = -4.00000000000000136349265211777_wp
  NSolveQuarticRoot (2,1,7) = -6.99999999999999838583980560358099_wp
  NSolveQuarticRoot (3,1,7) = -9.9999999999999984413534548366e+5_wp
  NSolveQuarticRoot (4,1,7) = -9.9999999999999984413534548366e+5_wp
  NSolveQuarticRoot (1,2,7) = 0.0_wp
  NSolveQuarticRoot (2,2,7) = 0.0_wp
  NSolveQuarticRoot (3,2,7) = 9.999999999999902752279012929648e+4_wp
  NSolveQuarticRoot (4,2,7) = -9.999999999999902752279012929648e+4_wp

  NSolveQuarticRoot (1,1,8) = 9.9999999999999969033524394035e+7_wp
  NSolveQuarticRoot (2,1,8) = 11.0000000000000031138286393783_wp
  NSolveQuarticRoot (3,1,8) = 1.00000000000000024674706722294e+3_wp
  NSolveQuarticRoot (4,1,8) = 1.00000000000000024674706722294e+3_wp
  NSolveQuarticRoot (1,2,8) = 0.0_wp
  NSolveQuarticRoot (2,2,8) = 0.0_wp
  NSolveQuarticRoot (3,2,8) = 0.9999999996991161792660000_wp
  NSolveQuarticRoot (4,2,8) = -0.9999999996991161792660000_wp

  NSolveQuarticRoot (1,1,9) = 9.99999999999999895408109296113252e+6_wp
  NSolveQuarticRoot (2,1,9) = 9.99999999999999895408109296113252e+6_wp
  NSolveQuarticRoot (3,1,9) = 1.0000000000000001471262348062_wp
  NSolveQuarticRoot (4,1,9) = 1.0000000000000001471262348062_wp
  NSolveQuarticRoot (1,2,9) = 1.000000000000007035509952402208e+6_wp
  NSolveQuarticRoot (2,2,9) = -1.000000000000007035509952402208e+6_wp
  NSolveQuarticRoot (3,2,9) = 2.00000000000000003729655473_wp
  NSolveQuarticRoot (4,2,9) = -2.00000000000000003729655473_wp

  NSolveQuarticRoot (1,1,10) = 9.99999999999999895639035685235e+3_wp
  NSolveQuarticRoot (2,1,10) = 9.99999999999999895639035685235e+3_wp
  NSolveQuarticRoot (3,1,10) = -6.9999999999999614396992142495_wp
  NSolveQuarticRoot (4,1,10) = -6.9999999999999614396992142495_wp
  NSolveQuarticRoot (1,2,10) = 3.0000000009772444004192692990073_wp
  NSolveQuarticRoot (2,2,10) = -3.0000000009772444004192692990073_wp
  NSolveQuarticRoot (3,2,10) = 9.9999999999999977795539507496e+2_wp
  NSolveQuarticRoot (4,2,10) = -9.9999999999999977795539507496e+2_wp

  NSolveQuarticRoot (1,1,11) = 1.001999999999917817516958962_wp
  NSolveQuarticRoot (2,1,11) = 1.001999999999917817516958962_wp
  NSolveQuarticRoot (3,1,11) = 1.00100000000061362115388874_wp
  NSolveQuarticRoot (4,1,11) = 1.00100000000061362115388874_wp
  NSolveQuarticRoot (1,2,11) = 4.99800000000033106672958638_wp
  NSolveQuarticRoot (2,2,11) = -4.99800000000033106672958638_wp
  NSolveQuarticRoot (3,2,11) = 5.0010000000000598419092057_wp
  NSolveQuarticRoot (4,2,11) = -5.0010000000000598419092057_wp

  NSolveQuarticRoot (1,1,12) = 9.9999999633947100896591564378e+2_wp
  NSolveQuarticRoot (2,1,12) = 9.9999999633947100896591564378e+2_wp
  NSolveQuarticRoot (3,1,12) = 1.00000000058396976143626488e+3_wp
  NSolveQuarticRoot (4,1,12) = 1.00000000058396976143626488e+3_wp
  NSolveQuarticRoot (1,2,12) = 3.00001177663677259799257512895_wp
  NSolveQuarticRoot (2,2,12) = -3.00001177663677259799257512895_wp
  NSolveQuarticRoot (3,2,12) = 0.999961997284177714351294241_wp
  NSolveQuarticRoot (4,2,12) = -0.999961997284177714351294241_wp

  NSolveQuarticRoot (1,1,13) = 2.000000000000002571077031_wp
  NSolveQuarticRoot (2,1,13) = 2.000000000000002571077031_wp
  NSolveQuarticRoot (3,1,13) = 1.000000000000000021141942363_wp
  NSolveQuarticRoot (4,1,13) = 1.000000000000000021141942363_wp
  NSolveQuarticRoot (1,2,13) = 1.000000000000000064637184493e+4_wp
  NSolveQuarticRoot (2,2,13) = -1.000000000000000064637184493e+4_wp
  NSolveQuarticRoot (3,2,13) = 1.000000000000000097005736776623e+3_wp
  NSolveQuarticRoot (4,2,13) = -1.000000000000000097005736776623e+3_wp

  else if (wp == qpKind) then

  NSolveQuarticRoot (1,1,1) = 1.0000000000000000000000000000000000e+9_wp
  NSolveQuarticRoot (2,1,1) = 1.0000000000000000000000000000000000e+6_wp
  NSolveQuarticRoot (3,1,1) = 1.0000000000000000000000000000000000e+3_wp
  NSolveQuarticRoot (4,1,1) = 1.0000000000000000000000000000000000_wp
  NSolveQuarticRoot (1,2,1) = 0.0_wp
  NSolveQuarticRoot (2,2,1) = 0.0_wp
  NSolveQuarticRoot (3,2,1) = 0.0_wp
  NSolveQuarticRoot (4,2,1) = 0.0_wp

  NSolveQuarticRoot (1,1,2) = 1.0029999999999999999999999_wp
  NSolveQuarticRoot (2,1,2) = 1.002000000000000000000001_wp
  NSolveQuarticRoot (3,1,2) = 1.000999999999999999999999_wp
  NSolveQuarticRoot (4,1,2) = 1.0000000000000000000000001_wp
  NSolveQuarticRoot (1,2,2) = 0.0_wp
  NSolveQuarticRoot (2,2,2) = 0.0_wp
  NSolveQuarticRoot (3,2,2) = 0.0_wp
  NSolveQuarticRoot (4,2,2) = 0.0_wp

  NSolveQuarticRoot (1,1,3) = 9.99999999999999999999999999999999999e+79_wp
  NSolveQuarticRoot (2,1,3) = -1.0000000000000000000000000000000001e+74_wp
  NSolveQuarticRoot (3,1,3) = -1.0000000000000000000000000000000001e+76_wp
  NSolveQuarticRoot (4,1,3) = -1.0000000000000000000000000000000001e+77_wp
  NSolveQuarticRoot (1,2,3) = 0.0_wp
  NSolveQuarticRoot (2,2,3) = 0.0_wp
  NSolveQuarticRoot (3,2,3) = 0.0_wp
  NSolveQuarticRoot (4,2,3) = 0.0_wp

  NSolveQuarticRoot (1,1,4) = 1.0000000000000000000000000000000001e+14_wp
  NSolveQuarticRoot (2,1,4) = 1.9999999999999999999999999999999999_wp
  NSolveQuarticRoot (3,1,4) = 0.9999999999999999999999999999999999_wp
  NSolveQuarticRoot (4,1,4) = -0.9999999999999999999999999999999999_wp
  NSolveQuarticRoot (1,2,4) = 0.0_wp
  NSolveQuarticRoot (2,2,4) = 0.0_wp
  NSolveQuarticRoot (3,2,4) = 0.0_wp
  NSolveQuarticRoot (4,2,4) = 0.0_wp

  NSolveQuarticRoot (1,1,5) = 9.9999999999999999999999999999999999e+6_wp
  NSolveQuarticRoot (2,1,5) = 1.0000000000000000000000000000000001_wp
  NSolveQuarticRoot (3,1,5) = -1.0000000000000000000000000000000001_wp
  NSolveQuarticRoot (4,1,5) = -1.9999999999999999999999999999999999e+7_wp
  NSolveQuarticRoot (1,2,5) = 0.0_wp
  NSolveQuarticRoot (2,2,5) = 0.0_wp
  NSolveQuarticRoot (3,2,5) = 0.0_wp
  NSolveQuarticRoot (4,2,5) = 0.0_wp

  NSolveQuarticRoot (1,1,6) = 1.00000000000000000000000000000000001e+7_wp
  NSolveQuarticRoot (2,1,6) = -1.00000000000000000000000000000000001e+6_wp
  NSolveQuarticRoot (3,1,6) = 1.00000000000000000000000000000000001_wp
  NSolveQuarticRoot (4,1,6) = 1.00000000000000000000000000000000001_wp
  NSolveQuarticRoot (1,2,6) = 0.0_wp
  NSolveQuarticRoot (2,2,6) = 0.0_wp
  NSolveQuarticRoot (3,2,6) = 1.00000000000000000000000000000000001_wp
  NSolveQuarticRoot (4,2,6) = -1.00000000000000000000000000000000001_wp

  NSolveQuarticRoot (1,1,7) = -4.000000000000000000000000000000001_wp
  NSolveQuarticRoot (2,1,7) = -7.000000000000000000000000000000001_wp
  NSolveQuarticRoot (3,1,7) = -1.000000000000000000000000000000001e+6_wp
  NSolveQuarticRoot (4,1,7) = -1.000000000000000000000000000000001e+6_wp
  NSolveQuarticRoot (1,2,7) = 0.0_wp
  NSolveQuarticRoot (2,2,7) = 0.0_wp
  NSolveQuarticRoot (3,2,7) =  9.99999999999999999999999999999999e+4_wp
  NSolveQuarticRoot (4,2,7) = -9.99999999999999999999999999999999e+4_wp

  NSolveQuarticRoot (1,1,8) = 9.9999999999999999999999999999999999e+7_wp
  NSolveQuarticRoot (2,1,8) = 10.9999999999999999999999999999999999_wp
  NSolveQuarticRoot (3,1,8) = 9.99999999999999999999999999999999e+2_wp
  NSolveQuarticRoot (4,1,8) = 9.99999999999999999999999999999999e+2_wp
  NSolveQuarticRoot (1,2,8) = 0.0_wp
  NSolveQuarticRoot (2,2,8) = 0.0_wp
  NSolveQuarticRoot (3,2,8) = 1.000000000000000000000000001_wp
  NSolveQuarticRoot (4,2,8) = -1.000000000000000000000000001_wp

  NSolveQuarticRoot (1,1,9) = 1.000000000000000000000000000000001e+7_wp
  NSolveQuarticRoot (2,1,9) = 1.000000000000000000000000000000001e+7_wp
  NSolveQuarticRoot (3,1,9) = 1.000000000000000000000000000000001_wp
  NSolveQuarticRoot (4,1,9) = 1.000000000000000000000000000000001_wp
  NSolveQuarticRoot (1,2,9) = 9.999999999999999999999999999999999999e+5_wp
  NSolveQuarticRoot (2,2,9) = -9.999999999999999999999999999999999999e+5_wp
  NSolveQuarticRoot (3,2,9) = 2.0000000000000000000000000000000001_wp
  NSolveQuarticRoot (4,2,9) = -2.0000000000000000000000000000000001_wp

  NSolveQuarticRoot (1,1,10) = 1.0000000000000000000000000000000001e+4_wp
  NSolveQuarticRoot (2,1,10) = 1.0000000000000000000000000000000001e+4_wp
  NSolveQuarticRoot (3,1,10) = -7.0000000000000000000000000000001_wp
  NSolveQuarticRoot (4,1,10) = -7.0000000000000000000000000000001_wp
  NSolveQuarticRoot (1,2,10) = 3.000000000000000000000000001_wp
  NSolveQuarticRoot (2,2,10) = -3.000000000000000000000000001_wp
  NSolveQuarticRoot (3,2,10) = 9.999999999999999999999999999999999999e+2_wp
  NSolveQuarticRoot (4,2,10) = -9.999999999999999999999999999999999999e+2_wp

  NSolveQuarticRoot (1,1,11) = 1.002000000000000000000000000001_wp
  NSolveQuarticRoot (2,1,11) = 1.002000000000000000000000000001_wp
  NSolveQuarticRoot (3,1,11) = 1.001000000000000000000000000001_wp
  NSolveQuarticRoot (4,1,11) = 1.001000000000000000000000000001_wp
  NSolveQuarticRoot (1,2,11) = 4.9980000000000000000000000000001_wp
  NSolveQuarticRoot (2,2,11) = -4.9980000000000000000000000000001_wp
  NSolveQuarticRoot (3,2,11) = 5.0010000000000000000000000000001_wp
  NSolveQuarticRoot (4,2,11) = -5.0010000000000000000000000000001_wp

  NSolveQuarticRoot (1,1,12) = 9.9999999999999999999999999999e+2_wp
  NSolveQuarticRoot (2,1,12) = 9.9999999999999999999999999999e+2_wp
  NSolveQuarticRoot (3,1,12) = 9.999999999999999999999999999e+2_wp
  NSolveQuarticRoot (4,1,12) = 9.999999999999999999999999999e+2_wp
  NSolveQuarticRoot (1,2,12) = 3.00000000000000000000000_wp
  NSolveQuarticRoot (2,2,12) = -3.00000000000000000000000_wp
  NSolveQuarticRoot (3,2,12) = 1.0000000000000000000000_wp
  NSolveQuarticRoot (4,2,12) = -1.0000000000000000000000_wp

  NSolveQuarticRoot (1,1,13) = 2.000000000000000000000000000001_wp
  NSolveQuarticRoot (2,1,13) = 2.000000000000000000000000000001_wp
  NSolveQuarticRoot (3,1,13) = 1.0000000000000000000000000000001_wp
  NSolveQuarticRoot (4,1,13) = 1.0000000000000000000000000000001_wp
  NSolveQuarticRoot (1,2,13) = 9.9999999999999999999999999999999999e+3_wp
  NSolveQuarticRoot (2,2,13) = -9.9999999999999999999999999999999999e+3_wp
  NSolveQuarticRoot (3,2,13) = 1.00000000000000000000000000000000001e+3_wp
  NSolveQuarticRoot (4,2,13) = -1.00000000000000000000000000000000001e+3_wp

  end if
!
!
!   ...Set the polynomial coefficients corresponding to the exact roots.
!
!  
  cubicA (1:9) = (/ -1.00000010000001e+14_wp, &
                    -3.000003_wp,             &
                    +8.999e+80_wp,            &
                    +1.e+24_wp,               &
                    -1.99999999999999e+14_wp, &
                    -3.e+5_wp,                &
                    -3.0_wp,                  &
                    -2.0000001e+7_wp,         &
                    +0.99999999999998e+14_wp  /)

  cubicB (1:9) = (/ +1.00000010000001e+21_wp, &
                    +3.000006000002_wp,       &
                    -1.0009e+161_wp,          &
                    -1.0_wp,                  &
                    +0.99999999999998e+28_wp, &
                    +3.0000000001e+10_wp,     &
                    +1.00000000000003e+14_wp, &
                    +1.00000020000001e+14_wp, &
                    -1.99999999999998e+14_wp  /)

  cubicC (1:9) = (/ -1.e+21_wp,               &
                    -1.000003000002_wp,       &
                    +1.e+238_wp,              &
                    -1.e+24_wp,               &
                    +1.e+28_wp,               &
                    -1.0000000001e+15_wp,     &
                    -1.00000000000001e+14_wp, &
                    -1.00000000000001e+14_wp, &
                    +2.e+14_wp                /)

  quarticA (1:13) = (/ -1.001001001e+9_wp,       &
                       -4.006_wp,                &
                       -9.98899e+79_wp,          &
                       -1.00000000000002e+14_wp, &
                       +1.e+7_wp,                &
                       -9.000002e+6_wp,          &
                       +2.000011e+6_wp,          &
                       -1.00002011e+8_wp,        &
                       -2.0000002e+7_wp,         &
                       -1.9986e+4_wp,            &
                       -4.006_wp,                &
                       -4.0e+3_wp,               &
                       -6.0_wp                   /)

  quarticB (1:13) = (/ +1.001002001001e+15_wp,   &
                       +6.018011_wp,             &
                       -1.1008989e+157_wp,       &
                       +1.99999999999999e+14_wp, &
                       -2.00000000000001e+14_wp, &
                       -0.9999981999998e+13_wp,  &
                       +1.010022000028e+12_wp,   &
                       +2.01101022001e+11_wp,    &
                       +1.01000040000005e+14_wp, &
                       +1.00720058e+8_wp,        &
                       +5.6008018e+1_wp,         &
                       +6.00001e+6_wp,           &
                       +1.01000013e+8_wp         /)

  quarticC (1:13) = (/ -1.001001001e+18_wp,      &
                       -4.018022006_wp,          &
                       -1.010999e+233_wp,        &
                       +1.00000000000002e+14_wp, &
                       -1.e+7_wp,                &
                       +1.9999982e+13_wp,        &
                       +1.1110056e+13_wp,        &
                       -1.02200111000011e+14_wp, &
                       -2.020001e+14_wp,         &
                       -1.8600979874e+10_wp,     &
                       -1.04148036024e+2_wp,     &
                       -4.00002e+9_wp,           &
                       -2.04000012e+8_wp         /)

  quarticD (1:13) = (/ +1.e+18_wp,               &
                       +1.006011006_wp,          &
                       -1.e+307_wp,              &
                       -2.e+14_wp,               &
                       +2.e+14_wp,               &
                       -2.e+13_wp,               &
                       +2.828e+13_wp,            &
                       +1.1000011e+15_wp,        &
                       +5.05e+14_wp,             &
                       +1.00004909000441e+14_wp, &
                       +6.75896068064016e+2_wp,  &
                       +1.000010000009e+12_wp,   &
                       +1.00000104000004e+14_wp  /)
!
!
!   ...Solve a set of 8 cubic polynomials.
!
!
  write (*,wpformat) ' -----------------------------------------------------------------------'
  write (*,wpformat) '              CUBIC POLYNOMIALS: x^3 + A x^2 + B x + C'
  write (*,wpformat) ' -----------------------------------------------------------------------'

  do i = 1,9

     p (1) = 1.0_wp
     p (2) = cubicA (i)
     p (3) = cubicB (i)
     p (4) = cubicC (i)
     
     A = cubicA (i)
     B = cubicB (i)
     C = cubicC (i)

     write (*,wpformat) ' '
     write (*,wpformat) ' A = ',A
     write (*,wpformat) ' B = ',B
     write (*,wpformat) ' C = ',C
     write (*,wpformat) ' '

     call cubicRoots (A,B,C, nReal, root (1:3,1:2), printInfo)

     write (*,wpformat) ' '
     write (*,wpformat) '  Cubic Solver Root 1 (x + iy) = ',root (1,Re),' + ',root (1,Im),' i'
     write (*,wpformat) '  Cubic Solver Root 2 (x + iy) = ',root (2,Re),' + ',root (2,Im),' i'
     write (*,wpformat) '  Cubic Solver Root 3 (x + iy) = ',root (3,Re),' + ',root (3,Im),' i'
     write (*,wpformat) ' '

     d1r = abs ((exactCubicRoot (1,1,i) - root (1,Re)) / exactCubicRoot (1,1,i))
     d2r = abs ((exactCubicRoot (2,1,i) - root (2,Re)) / exactCubicRoot (2,1,i))
     d3r = abs ((exactCubicRoot (3,1,i) - root (3,Re)) / exactCubicRoot (3,1,i))

     drMax = max (d1r,d2r,d3r)
     write (*,wpformat) '  Cubic Solver maximum relative accuracy (real) = ',drMax 

     if (root (2,Im) /= 0.0_wp) then
         diMax = abs ((exactCubicRoot (2,2,i) - root (2,Im)) / exactCubicRoot (2,2,i))
         write (*,wpformat) '  Cubic Solver maximum relative accuracy (imag) = ',diMax 
     end if

     call rpoly (p,3,  zr,zi,fail)
!
!
!   ...Reorders Jenkins-Traub roots to correspond to roots from the cubic solver.
!
!  
     if (zr (2) > zr (1)) then
         A = zr (2)
         B = zi (2)
         zr (2) = zr (1)
         zi (2) = zi (1)
         zr (1) = A
         zi (1) = B
     end if

     if (zr (3) > zr (1)) then
         A = zr (3)
         B = zi (3)
         zr (3) = zr (2)
         zi (3) = zi (2)
         zr (2) = zr (1)
         zi (2) = zi (1)
         zr (1) = A
         zi (1) = B
     else if (zr (3) > zr (2)) then
         A = zr (3)
         B = zi (3)
         zr (3) = zr (2)
         zi (3) = zi (2)
         zr (2) = A
         zi (2) = B
     end if

     if (zi (3) == 0.0_wp .and. zi (1) /= 0.0_wp) then
         A = zr (3)
         B = zi (3)
         zr (3) = zr (2)
         zi (3) = zi (2)
         zr (2) = zr (1)
         zi (2) = zi (1)
         zr (1) = A
         zi (1) = B
     end if

     write (*,wpformat) ' '
     write (*,wpformat) ' Jenkins Traub Root 1 (x + iy) = ',zr (1),' + ',zi (1),' i'
     write (*,wpformat) ' Jenkins Traub Root 2 (x + iy) = ',zr (2),' + ',zi (2),' i'
     write (*,wpformat) ' Jenkins Traub Root 3 (x + iy) = ',zr (3),' + ',zi (3),' i'
     write (*,wpformat) ' '

     d1r = abs ((exactCubicRoot (1,1,i) - zr (1)) / exactCubicRoot (1,1,i))
     d2r = abs ((exactCubicRoot (2,1,i) - zr (2)) / exactCubicRoot (2,1,i))
     d3r = abs ((exactCubicRoot (3,1,i) - zr (3)) / exactCubicRoot (3,1,i))

     drMax = max (d1r,d2r,d3r)
     write (*,wpformat) ' Jenkins Traub maximum relative accuracy (real) = ',drMax 

     if (zi (2) /= 0.0_wp) then
         diMax = abs ((exactCubicRoot (2,2,i) - zi (2)) / exactCubicRoot (2,2,i))
         write (*,wpformat) ' Jenkins Traub maximum relative accuracy (imag) = ',diMax 
     end if
     
     write (*,wpformat) ' '
     write (*,wpformat) '   Mathematica Root 1 (x + iy) = ',NSolveCubicRoot (1,1,i), &
                                                             ' + ',NSolveCubicRoot (1,2,i),' i'
     write (*,wpformat) '   Mathematica Root 2 (x + iy) = ',NSolveCubicRoot (2,1,i), &
                                                             ' + ',NSolveCubicRoot (2,2,i),' i'
     write (*,wpformat) '   Mathematica Root 3 (x + iy) = ',NSolveCubicRoot (3,1,i), &
                                                             ' + ',NSolveCubicRoot (3,2,i),' i'
     write (*,wpformat) ' '

     d1r = abs ((exactCubicRoot (1,1,i) - NSolveCubicRoot (1,1,i)) / exactCubicRoot (1,1,i))
     d2r = abs ((exactCubicRoot (2,1,i) - NSolveCubicRoot (2,1,i)) / exactCubicRoot (2,1,i))
     d3r = abs ((exactCubicRoot (3,1,i) - NSolveCubicRoot (3,1,i)) / exactCubicRoot (3,1,i))

     drMax = max (d1r,d2r,d3r)
     write (*,wpformat) '   Mathematica maximum relative accuracy (real) = ',drMax 

     if (NSolveCubicRoot (2,2,i) /= 0.0_wp) then
         diMax = abs ((exactCubicRoot (2,2,i) - NSolveCubicRoot (2,2,i)) / exactCubicRoot (2,2,i))
         write (*,wpformat) '   Mathematica maximum relative accuracy (imag) = ',diMax 
     end if

     write (*,wpformat) ' -----------------------------------------------------------------------'

  end do
!
!
!   ...Solve a set of 13 quartic polynomials.
!
!  
  write (*,wpformat) ' -----------------------------------------------------------------------'
  write (*,wpformat) '            QUARTIC POLYNOMIALS: x^4 + A x^3 + B x^2 + C x + D'
  write (*,wpformat) ' -----------------------------------------------------------------------'

  do i = 1,13

     p (1) = 1.0_wp
     p (2) = quarticA (i)
     p (3) = quarticB (i)
     p (4) = quarticC (i)
     p (5) = quarticD (i)

     A = quarticA (i)
     B = quarticB (i)
     C = quarticC (i)
     D = quarticD (i)

     write (*,wpformat) ' '
     write (*,wpformat) ' A = ',A
     write (*,wpformat) ' B = ',B
     write (*,wpformat) ' C = ',C
     write (*,wpformat) ' D = ',D
     write (*,wpformat) ' '

     call quarticRoots (A,B,C,D, nReal,root (1:4,1:2), printInfo)

     write (*,wpformat) ' '
     write (*,wpformat) ' Quartic Solver Root 1 (x + iy) = ',root (1,Re),' + ',root (1,Im),' i'
     write (*,wpformat) ' Quartic Solver Root 2 (x + iy) = ',root (2,Re),' + ',root (2,Im),' i'
     write (*,wpformat) ' Quartic Solver Root 3 (x + iy) = ',root (3,Re),' + ',root (3,Im),' i'
     write (*,wpformat) ' Quartic Solver Root 4 (x + iy) = ',root (4,Re),' + ',root (4,Im),' i'
     write (*,wpformat) ' '

     d1r = abs ((exactQuarticRoot (1,1,i) - root (1,Re)) / exactQuarticRoot (1,1,i))
     d2r = abs ((exactQuarticRoot (2,1,i) - root (2,Re)) / exactQuarticRoot (2,1,i))
     d3r = abs ((exactQuarticRoot (3,1,i) - root (3,Re)) / exactQuarticRoot (3,1,i))
     d4r = abs ((exactQuarticRoot (4,1,i) - root (4,Re)) / exactQuarticRoot (4,1,i))

     drMax = max (d1r,d2r,d3r,d4r)
     write (*,wpformat) ' Quartic Solver maximum relative accuracy (real) = ',drMax 

     d1i = -1.0_wp
     if (root (1,Im) /= 0.0_wp) then
         d1i = abs ((exactQuarticRoot (1,2,i) - root (1,Im)) / exactQuarticRoot (1,2,i))
     end if

     d3i = -1.0_wp
     if (root (3,Im) /= 0.0_wp) then
         d3i = abs ((exactQuarticRoot (3,2,i) - root (3,Im)) / exactQuarticRoot (3,2,i))
     end if

     if (d1i >= 0.0_wp .or. d3i >= 0.0_wp) then
         diMax = max (d1i,d3i)
         write (*,wpformat) ' Quartic Solver maximum relative accuracy (imag) = ',diMax 
     end if

     call rpoly (p,4,  zr,zi,fail)
!
!
!   ...Reorders Jenkins-Traub roots to correspond to roots from the quartic solver.
!
!  
     if (zr (2) > zr (1)) then
         A = zr (2)
         B = zi (2)
         zr (2) = zr (1)
         zi (2) = zi (1)
         zr (1) = A
         zi (1) = B
     end if

     if (zr (3) > zr (1)) then
         A = zr (3)
         B = zi (3)
         zr (3) = zr (2)
         zi (3) = zi (2)
         zr (2) = zr (1)
         zi (2) = zi (1)
         zr (1) = A
         zi (1) = B
     else if (zr (3) > zr (2)) then
         A = zr (3)
         B = zi (3)
         zr (3) = zr (2)
         zi (3) = zi (2)
         zr (2) = A
         zi (2) = B
     end if

     if (zr (4) > zr (1)) then
         A = zr (4)
         B = zi (4)
         zr (4) = zr (3)
         zi (4) = zi (3)
         zr (3) = zr (2)
         zi (3) = zi (2)
         zr (2) = zr (1)
         zi (2) = zi (1)
         zr (1) = A
         zi (1) = B
     else if (zr (4) > zr (2)) then
         A = zr (4)
         B = zi (4)
         zr (4) = zr (3)
         zi (4) = zi (3)
         zr (3) = zr (2)
         zi (3) = zi (2)
         zr (2) = A
         zi (2) = B
     else if (zr (4) > zr (3)) then
         A = zr (4)
         B = zi (4)
         zr (4) = zr (3)
         zi (4) = zi (3)
         zr (3) = A
         zi (3) = B
     end if

     if (zi (1) == 0.0_wp .and. zi (2) /= 0.0_wp) then
         A = zr (4)
         B = zi (4)
         zr (4) = zr (3)
         zi (4) = zi (3)
         zr (3) = zr (2)
         zi (3) = zi (2)
         zr (2) = A
         zi (2) = B
     end if

     write (*,wpformat) ' '
     write (*,wpformat) '  Jenkins Traub Root 1 (x + iy) = ',zr (1),' + ',zi (1),' i'
     write (*,wpformat) '  Jenkins Traub Root 2 (x + iy) = ',zr (2),' + ',zi (2),' i'
     write (*,wpformat) '  Jenkins Traub Root 3 (x + iy) = ',zr (3),' + ',zi (3),' i'
     write (*,wpformat) '  Jenkins Traub Root 4 (x + iy) = ',zr (4),' + ',zi (4),' i'
     write (*,wpformat) ' '
     
     d1r = abs ((exactQuarticRoot (1,1,i) - zr (1)) / exactQuarticRoot (1,1,i))
     d2r = abs ((exactQuarticRoot (2,1,i) - zr (2)) / exactQuarticRoot (2,1,i))
     d3r = abs ((exactQuarticRoot (3,1,i) - zr (3)) / exactQuarticRoot (3,1,i))
     d4r = abs ((exactQuarticRoot (4,1,i) - zr (4)) / exactQuarticRoot (4,1,i))

     drMax = max (d1r,d2r,d3r,d4r)
     write (*,wpformat) '  Jenkins Traub maximum relative accuracy (real) = ',drMax 

     d1i = -1.0_wp
     if (zi (1) /= 0.0_wp) then
         d1i = abs ((exactQuarticRoot (1,2,i) - zi (1)) / exactQuarticRoot (1,2,i))
     end if

     d3i = -1.0_wp
     if (zi (3) /= 0.0_wp) then
         d3i = abs ((exactQuarticRoot (3,2,i) - zi (3)) / exactQuarticRoot (3,2,i))
     end if

     if (d1i >= 0.0_wp .or. d3i >= 0.0_wp) then
         diMax = max (d1i,d3i)
         write (*,wpformat) '  Jenkins Traub maximum relative accuracy (imag) = ',diMax 
     end if

     write (*,wpformat) ' '
     write (*,wpformat) '    Mathematica Root 1 (x + iy) = ',NSolveQuarticRoot (1,1,i), &
                                                              ' + ',NSolveQuarticRoot (1,2,i),' i'
     write (*,wpformat) '    Mathematica Root 2 (x + iy) = ',NSolveQuarticRoot (2,1,i), &
                                                              ' + ',NSolveQuarticRoot (2,2,i),' i'
     write (*,wpformat) '    Mathematica Root 3 (x + iy) = ',NSolveQuarticRoot (3,1,i), &
                                                              ' + ',NSolveQuarticRoot (3,2,i),' i'
     write (*,wpformat) '    Mathematica Root 4 (x + iy) = ',NSolveQuarticRoot (4,1,i), &
                                                              ' + ',NSolveQuarticRoot (4,2,i),' i'
     write (*,wpformat) ' '

     d1r = abs ((exactQuarticRoot (1,1,i) - NSolveQuarticRoot (1,1,i)) / exactQuarticRoot (1,1,i))
     d2r = abs ((exactQuarticRoot (2,1,i) - NSolveQuarticRoot (2,1,i)) / exactQuarticRoot (2,1,i))
     d3r = abs ((exactQuarticRoot (3,1,i) - NSolveQuarticRoot (3,1,i)) / exactQuarticRoot (3,1,i))
     d4r = abs ((exactQuarticRoot (4,1,i) - NSolveQuarticRoot (4,1,i)) / exactQuarticRoot (4,1,i))

     drMax = max (d1r,d2r,d3r,d4r)
     write (*,wpformat) '    Mathematica maximum relative accuracy (real) = ',drMax 

     d1i = -1.0_wp
     if (NSolveQuarticRoot (1,2,i) /= 0.0_wp) then
         d1i = abs ((exactQuarticRoot (1,2,i) - NSolveQuarticRoot (1,2,i)) / exactQuarticRoot (1,2,i))
     end if

     d3i = -1.0_wp
     if (NSolveQuarticRoot (3,2,i) /= 0.0_wp) then
         d3i = abs ((exactQuarticRoot (3,2,i) - NSolveQuarticRoot (3,2,i)) / exactQuarticRoot (3,2,i))
     end if

     if (d1i >= 0.0_wp .or. d3i >= 0.0_wp) then
         diMax = max (d1i,d3i)
         write (*,wpformat) '    Mathematica maximum relative accuracy (imag) = ',diMax 
     end if

     write (*,wpformat) ' -----------------------------------------------------------------------'

  end do
!
!
!   ...Ready!
!
!  
end program PolynomialRootsCompare
