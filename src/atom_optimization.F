!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Optimizer for the atomic code
! **************************************************************************************************
MODULE atom_optimization
   USE atom_types,                      ONLY: atom_optimization_type
   USE kinds,                           ONLY: dp
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'atom_optimization'

   TYPE hmat_type
      REAL(dp)                                  :: energy = 0.0_dp
      REAL(dp)                                  :: error = 0.0_dp
      REAL(dp), DIMENSION(:, :, :), POINTER     :: emat => NULL()
      REAL(dp), DIMENSION(:, :, :), POINTER     :: fmat => NULL()
      REAL(dp), DIMENSION(:, :, :), POINTER     :: pmat => NULL()
   END TYPE hmat_type

   TYPE atom_history_type
      INTEGER                                  :: max_history = 0
      INTEGER                                  :: hlen = 0
      INTEGER                                  :: hpos = 0
      REAL(dp)                                 :: damping = 0.0_dp
      REAL(dp)                                 :: eps_diis = 0.0_dp
      REAL(dp), DIMENSION(:, :), POINTER       :: dmat => NULL()
      TYPE(hmat_type), DIMENSION(:), POINTER   :: hmat => NULL()
   END TYPE atom_history_type

   PUBLIC :: atom_opt_fmat, &
             atom_history_type, atom_history_init, atom_history_update, atom_history_release

CONTAINS

! **************************************************************************************************
!> \brief Initialise a circular buffer to keep Kohn-Sham and density matrices from previous iteration.
!> \param history       object to initialise
!> \param optimization  optimisation parameters
!> \param matrix        reference matrix. Historic matrices will have the same size as
!>                      this reference matrix
!> \par History
!>    * 08.2016 new structure element: density matrix [Juerg Hutter]
!>    * 08.2008 created [Juerg Hutter]
! **************************************************************************************************
   PURE SUBROUTINE atom_history_init(history, optimization, matrix)
      TYPE(atom_history_type), INTENT(INOUT)             :: history
      TYPE(atom_optimization_type), INTENT(IN)           :: optimization
      REAL(dp), DIMENSION(:, :, :), INTENT(IN)           :: matrix

      INTEGER                                            :: i, n1, n2, n3, ndiis
      REAL(dp)                                           :: damp, eps

      ndiis = optimization%n_diis
      eps = optimization%eps_diis
      damp = optimization%damping

      CALL atom_history_release(history)

      history%max_history = ndiis
      history%hlen = 0
      history%hpos = 0
      history%damping = damp
      history%eps_diis = eps
      ALLOCATE (history%dmat(ndiis + 1, ndiis + 1))

      ALLOCATE (history%hmat(ndiis))
      n1 = SIZE(matrix, 1)
      n2 = SIZE(matrix, 2)
      n3 = SIZE(matrix, 3)
      DO i = 1, ndiis
         history%hmat(i)%energy = 0.0_dp
         history%hmat(i)%error = 0.0_dp
         ALLOCATE (history%hmat(i)%emat(n1, n2, n3))
         ALLOCATE (history%hmat(i)%fmat(n1, n2, n3))
         ALLOCATE (history%hmat(i)%pmat(n1, n2, n3))
      END DO

   END SUBROUTINE atom_history_init

! **************************************************************************************************
!> \brief Add matrices from the current iteration into the circular buffer.
!> \param history  object to keep historic matrices
!> \param pmat     density matrix
!> \param fmat     Kohn-Sham matrix
!> \param emat     error matrix
!> \param energy   total energy
!> \param error    convergence
!> \par History
!>    * 08.2016 new formal argument: density matrix [Juerg Hutter]
!>    * 08.2008 created [Juerg Hutter]
! **************************************************************************************************
   PURE SUBROUTINE atom_history_update(history, pmat, fmat, emat, energy, error)
      TYPE(atom_history_type), INTENT(INOUT)             :: history
      REAL(dp), DIMENSION(:, :, :), INTENT(IN)           :: pmat, fmat, emat
      REAL(dp), INTENT(IN)                               :: energy, error

      INTEGER                                            :: nlen, nmax, nnow

      nmax = history%max_history
      nlen = MIN(history%hlen + 1, nmax)
      nnow = history%hpos + 1
      IF (nnow > nmax) nnow = 1

      history%hmat(nnow)%energy = energy
      history%hmat(nnow)%error = error
      history%hmat(nnow)%pmat = pmat
      history%hmat(nnow)%fmat = fmat
      history%hmat(nnow)%emat = emat

      history%hlen = nlen
      history%hpos = nnow

   END SUBROUTINE atom_history_update

! **************************************************************************************************
!> \brief Release circular buffer to keep historic matrices.
!> \param history  object to release
!> \par History
!>    * 08.2008 created [Juerg Hutter]
! **************************************************************************************************
   PURE SUBROUTINE atom_history_release(history)
      TYPE(atom_history_type), INTENT(INOUT)             :: history

      INTEGER                                            :: i

      history%max_history = 0
      history%hlen = 0
      history%hpos = 0
      history%damping = 0._dp
      history%eps_diis = 0._dp
      IF (ASSOCIATED(history%dmat)) THEN
         DEALLOCATE (history%dmat)
      END IF
      IF (ASSOCIATED(history%hmat)) THEN
         DO i = 1, SIZE(history%hmat)
            IF (ASSOCIATED(history%hmat(i)%emat)) THEN
               DEALLOCATE (history%hmat(i)%emat)
            END IF
            IF (ASSOCIATED(history%hmat(i)%fmat)) THEN
               DEALLOCATE (history%hmat(i)%fmat)
            END IF
            IF (ASSOCIATED(history%hmat(i)%pmat)) THEN
               DEALLOCATE (history%hmat(i)%pmat)
            END IF
         END DO
         DEALLOCATE (history%hmat)
      END IF

   END SUBROUTINE atom_history_release

! **************************************************************************************************
!> \brief Construct a Kohn-Sham matrix for the next iteration based on the historic data.
!> \param fmat     new Kohn-Sham matrix
!> \param history  historic matrices
!> \param err      convergence
!> \par History
!>    * 08.2016 renamed to atom_opt_fmat() [Juerg Hutter]
!>    * 08.2008 created as atom_opt() [Juerg Hutter]
! **************************************************************************************************
   SUBROUTINE atom_opt_fmat(fmat, history, err)
      REAL(dp), DIMENSION(:, :, :), INTENT(INOUT)        :: fmat
      TYPE(atom_history_type), INTENT(INOUT)             :: history
      REAL(dp), INTENT(IN)                               :: err

      INTEGER                                            :: i, info, j, lwork, na, nb, nlen, nm, &
                                                            nmax, nnow, rank
      REAL(dp)                                           :: a, rcond, t
      REAL(dp), ALLOCATABLE, DIMENSION(:)                :: s, work
      REAL(dp), ALLOCATABLE, DIMENSION(:, :)             :: vec

      nmax = history%max_history
      nnow = history%hpos
      a = history%damping
      IF (history%hlen > 1) THEN
         IF (err < history%eps_diis) THEN
            ! DIIS
            rcond = 1.e-10_dp
            lwork = 25*nmax
            ALLOCATE (vec(nmax + 1, 2), s(nmax + 1), work(lwork))
            nlen = history%hlen
            vec = 0._dp
            vec(nlen + 1, 1) = 1._dp
            history%dmat(1:nlen, nlen + 1) = 1._dp
            history%dmat(nlen + 1, 1:nlen) = 1._dp
            history%dmat(nlen + 1, nlen + 1) = 0._dp
            DO i = 1, nlen
               na = nnow + 1 - i
               IF (na < 1) na = nmax + na
               DO j = i, nlen
                  nb = nnow + 1 - j
                  IF (nb < 1) nb = nmax + nb
                  t = SUM(history%hmat(na)%emat*history%hmat(nb)%emat)
                  history%dmat(i, j) = t
                  history%dmat(j, i) = t
               END DO
            END DO
            CALL dgelss(nlen + 1, nlen + 1, 1, history%dmat, nmax + 1, vec, nmax + 1, s, &
                        rcond, rank, work, lwork, info)
            CPASSERT(info == 0)
            fmat = 0._dp
            DO i = 1, nlen
               na = nnow + 1 - i
               IF (na < 1) na = nmax + na
               fmat = fmat + vec(i, 1)*history%hmat(na)%fmat
            END DO

            DEALLOCATE (vec, s, work)
         ELSE
            ! damping
            nm = nnow - 1
            IF (nm < 1) nm = history%max_history
            fmat = a*history%hmat(nnow)%fmat + (1._dp - a)*history%hmat(nm)%fmat
         END IF
      ELSEIF (history%hlen == 1) THEN
         fmat = history%hmat(nnow)%fmat
      ELSE
         CPABORT("")
      END IF

   END SUBROUTINE atom_opt_fmat

END MODULE atom_optimization
