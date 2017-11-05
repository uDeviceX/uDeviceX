program cubic
  use Polynomial234RootSolvers, ONLY : cubicRoots
  use SetWorkingPrecision,      ONLY : wp, spKind

  implicit none
  real (kind = wp    ) :: c2,c1,c0
  integer :: nReal
  real (kind = wp) :: root (1:3,1:2)

  c2 = 0.00002
  c1 = 6.7
  c0 = 8.9
  call cubicRoots (c2,c1,c0, nReal, root (1:3,1:2))

  write (*, *) nReal
  write (*, *) root(1, 1), root(2, 1), root(3, 1)
  write (*, *) root(1, 2), root(2, 2), root(3, 2)

end program cubic
