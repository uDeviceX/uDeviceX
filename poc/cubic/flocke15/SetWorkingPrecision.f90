    MODULE SetWorkingPrecision
! ..
! .. Intrinsic Functions ..
      INTRINSIC KIND
! .. Parameters ..
! Define the standard precisions
! For IEEE standard arithmetic we could also use
!     INTEGER, PARAMETER :: skind = SELECTED_REAL_KIND(p=6, r=37)
!     INTEGER, PARAMETER :: dkind = SELECTED_REAL_KIND(p=15, r=307)
      INTEGER, PARAMETER :: spKind = KIND(0.0E0)
      INTEGER, PARAMETER :: dpKind = KIND(0.0D0)
      INTEGER, PARAMETER :: qpKind = selected_real_kind (32)
! Set the precision for the whole package
      INTEGER, PARAMETER :: wp = dpkind
! To change the default package precision to single precision
! change the parameter assignment to wp above to
!     INTEGER, PARAMETER :: wp = skind
! and recompile the complete package.
      CHARACTER(LEN=*), PARAMETER :: spformat = '(a, e12.6)'
      CHARACTER(LEN=*), PARAMETER :: dpformat = '(a, e23.16)'
      CHARACTER(LEN=*), PARAMETER :: wpformat = dpformat

    END MODULE SetWorkingPrecision

