subroutine nxt_str(s)
  implicit none
  integer, save      :: i = 1
  character(len=256) :: s

  call getarg(i, s)
  i = i + 1
end subroutine nxt_str

subroutine nxt_real(x)
  use SetWorkingPrecision,      only : wp
  implicit none
  real (kind = wp  ) :: x
  character(len=256) :: s
  call nxt_str(s)
  read(s, *) x
end subroutine nxt_real

program cubic
  use Polynomial234RootSolvers, only : cubicRoots
  use SetWorkingPrecision,      only : wp, wpformat

  implicit none

  real (kind = wp    ) :: c3,c2,c1,c0
  integer :: nReal
  real (kind = wp) :: r (1:3,1:2)

  call nxt_real(c3)
  call nxt_real(c2)
  call nxt_real(c1)
  call nxt_real(c0)
  c2 = c2/c3; c1 = c1/c3; c0 = c0/c3
  call cubicRoots (c2,c1,c0, nReal, r (1:3,1:2))

  write (*, '(6e23.16)') r(1, 1), r(1, 2), r(2, 1), r(2, 2), r(3, 1), r(3, 2)

end program cubic
