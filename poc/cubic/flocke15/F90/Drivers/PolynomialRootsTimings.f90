program PolynomialRootsTimings
!
!
!   ...This program produces a set of random coefficients for cubics
!      and quartics and records execution times for all solvers.
!
!  
  use Polynomial234RootSolvers, ONLY : cubicRoots, quarticRoots
  use JenkinsTraubSolver,       ONLY : rpoly
  use SetWorkingPrecision,      ONLY : wp, spKind

  implicit none

  character (len=*), parameter :: timeFormat = "(a,f10.3,a,f10.3)"

  logical :: Cubic = .true.         ! if true => do cubic timings
  logical :: Quartic = .true.       ! if true => do quartic timings
  logical :: fail

  integer :: n
  integer :: nReal
  integer :: nSet
  integer :: seedSize

! integer, parameter :: Re = 1
! integer, parameter :: Im = 2

  integer :: seedAstore (1:10)
  integer :: seedBstore (1:10)
  integer :: seedCstore (1:10)
  integer :: seedDstore (1:10)

  integer, allocatable :: seedA (:)
  integer, allocatable :: seedB (:)
  integer, allocatable :: seedC (:)
  integer, allocatable :: seedD (:)

  integer, parameter :: nPolynomials = 10000000

  real (kind = spKind) :: tbeg, tend, tmin, tmax, tavg, tpm
  real (kind = wp    ) :: c2,c1,c0
  real (kind = wp    ) :: q3,q2,q1,q0

  real (kind = spKind) :: timeJTcubic   (1:10)
  real (kind = spKind) :: timeJTquartic (1:10)
  real (kind = spKind) :: timeCS        (1:10)
  real (kind = spKind) :: timeQS        (1:10)

  real (kind = wp) :: root (1:4,1:2)

  real (kind = wp) :: Astore  (1:nPolynomials)
  real (kind = wp) :: Bstore  (1:nPolynomials)
  real (kind = wp) :: Cstore  (1:nPolynomials)
  real (kind = wp) :: Dstore  (1:nPolynomials)
  real (kind = wp) :: r       (1:nPolynomials)

  real (kind = wp) :: p (101), zr (100), zi (100)       ! Jenkins-Traub arrays
!
!
!   ...Set the seeds store.
!
!
  seedAstore (1)  = 23
  seedAstore (2)  = 54345
  seedAstore (3)  = 65734
  seedAstore (4)  = 776
  seedAstore (5)  = 12
  seedAstore (6)  = 324
  seedAstore (7)  = 8768786
  seedAstore (8)  = 43
  seedAstore (9)  = 999
  seedAstore (10) = 234

  seedBstore (1)  = 5366
  seedBstore (2)  = 7545
  seedBstore (3)  = 123
  seedBstore (4)  = 9
  seedBstore (5)  = 675
  seedBstore (6)  = 82553
  seedBstore (7)  = 44
  seedBstore (8)  = 224453
  seedBstore (9)  = 1234
  seedBstore (10) = 878

  seedCstore (1)  = 434342
  seedCstore (2)  = 54
  seedCstore (3)  = 7
  seedCstore (4)  = 3256
  seedCstore (5)  = 87678
  seedCstore (6)  = 3344
  seedCstore (7)  = 11234445
  seedCstore (8)  = 7676
  seedCstore (9)  = 44
  seedCstore (10) = 8132

  seedDstore (1)  = 74674
  seedDstore (2)  = 4543
  seedDstore (3)  = 7678
  seedDstore (4)  = 2222
  seedDstore (5)  = 543
  seedDstore (6)  = 8765
  seedDstore (7)  = 43
  seedDstore (8)  = 77
  seedDstore (9)  = 9000
  seedDstore (10) = 3222
!
!
!   ...Outer loop over all 10 sets. Set the seeds.
!
!
  call random_seed (size = seedSize)

  allocate (seedA (1:seedSize), seedB (1:seedSize), seedC (1:seedSize), seedD (1:seedSize))

  do nSet = 1,10

     seedA (1:seedSize) = seedAstore (nSet)
     seedB (1:seedSize) = seedBstore (nSet)
     seedC (1:seedSize) = seedCstore (nSet)
     seedD (1:seedSize) = seedDstore (nSet)
!
!
!   ...Calculate the random coefficients.
!
!  
     call random_seed   (put = seedA)
     call random_number (r)

     Astore (:) = 1.0_wp * (2 * r (:) - 1.0_wp)

     call random_seed   (put = seedB)
     call random_number (r)

     Bstore (:) = 1.0_wp * (2 * r (:) - 1.0_wp)

     call random_seed   (put = seedC)
     call random_number (r)

     Cstore (:) = 1.0_wp * (2 * r (:) - 1.0_wp)

     call random_seed   (put = seedD)
     call random_number (r)

     Dstore (:) = 1.0_wp * (2 * r (:) - 1.0_wp)
!
!
!   ...Do cubic timings (if requested).
!
!  
     if (Cubic) then

         call cpu_time (tbeg)

         do n = 1,nPolynomials

            p (1) = 1.0_wp
            p (2) = Astore (n)
            p (3) = Bstore (n)
            p (4) = Cstore (n)

            call rpoly (p,3,  zr,zi,fail)

         end do

         call cpu_time (tend)

         write (*,timeFormat) ' cpu time Cubic   Jenkins Traub = ',tend - tbeg

         timeJTcubic (nSet) = tend - tbeg

         call cpu_time (tbeg)

         do n = 1,nPolynomials

            c2 = Astore (n)
            c1 = Bstore (n)
            c0 = Cstore (n)

            call cubicRoots (c2,c1,c0, nReal, root (1:3,1:2))

         end do

         call cpu_time (tend)

         write (*,timeFormat) ' cpu time Cubic   Solver        = ',tend - tbeg

         timeCS (nSet) = tend - tbeg

     end if
!
!
!   ...Do quartic timings (if requested).
!
!  
     if (Quartic) then

         call cpu_time (tbeg)

         do n = 1,nPolynomials

            p (1) = 1.0_wp
            p (2) = Astore (n)
            p (3) = Bstore (n)
            p (4) = Cstore (n)
            p (5) = Dstore (n)

            call rpoly (p,4,  zr,zi,fail)

         end do

         call cpu_time (tend)

         write (*,timeFormat) ' cpu time Quartic Jenkins Traub = ',tend - tbeg

         timeJTquartic (nSet) = tend - tbeg

         call cpu_time (tbeg)

         do n = 1,nPolynomials

            q3 = Astore (n)
            q2 = Bstore (n)
            q1 = Cstore (n)
            q0 = Dstore (n)

            call quarticRoots (q3,q2,q1,q0, nReal, root (1:4,1:2))

         end do

         call cpu_time (tend)

         write (*,timeFormat) ' cpu time Quartic Solver        = ',tend - tbeg

         timeQS (nSet) = tend - tbeg

     end if

     write (*,*) ' ------------------------------------------------------ '

  end do
!
!
!   ...Collect minima and maxima.
!
!  
  if (Cubic) then

      tmin = minval (timeJTcubic (1:10))
      tmax = maxval (timeJTcubic (1:10))
      tavg = (tmin + tmax) / 2
      tpm  = (tmax - tmin) / 2

      write (*,timeFormat) ' minimum cpu time Cubic   Jenkins Traub = ',tmin
      write (*,timeFormat) ' maximum cpu time Cubic   Jenkins Traub = ',tmax
      write (*,timeFormat) ' average cpu time Cubic   Jenkins Traub  = ',tavg,' +/- ',tpm
      write (*,timeFormat) ' ------------------------------------------------------ '

      tmin = minval (timeCS (1:10))
      tmax = maxval (timeCS (1:10))
      tavg = (tmin + tmax) / 2
      tpm  = (tmax - tmin) / 2

      write (*,timeFormat) ' minimum cpu time Cubic   Solver        = ',tmin
      write (*,timeFormat) ' maximum cpu time Cubic   Solver        = ',tmax
      write (*,timeFormat) ' average cpu time Cubic   Solver        = ',tavg,' +/- ',tpm
      write (*,timeFormat) ' ------------------------------------------------------ '

  end if

  if (Quartic) then

      tmin = minval (timeJTquartic (1:10))
      tmax = maxval (timeJTquartic (1:10))
      tavg = (tmin + tmax) / 2
      tpm  = (tmax - tmin) / 2

      write (*,timeFormat) ' minimum cpu time Quartic Jenkins Traub = ',tmin
      write (*,timeFormat) ' maximum cpu time Quartic Jenkins Traub = ',tmax
      write (*,timeFormat) ' average cpu time Quartic Jenkins Traub  = ',tavg,' +/- ',tpm
      write (*,timeFormat) ' ------------------------------------------------------ '

      tmin = minval (timeQS (1:10))
      tmax = maxval (timeQS (1:10))
      tavg = (tmin + tmax) / 2
      tpm  = (tmax - tmin) / 2

      write (*,timeFormat) ' minimum cpu time Quartic Solver        = ',tmin
      write (*,timeFormat) ' maximum cpu time Quartic Solver        = ',tmax
      write (*,timeFormat) ' average cpu time Quartic Solver        = ',tavg,' +/- ',tpm
      write (*,timeFormat) ' ------------------------------------------------------ '

  end if
!
!
!   ...Ready!
!
!  
end program PolynomialRootsTimings
