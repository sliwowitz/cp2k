!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Provides types for the management of the xc-functionals and
!>      their derivatives.
! **************************************************************************************************
MODULE xc_derivative_types

   USE kinds,                           ONLY: dp
   USE pw_pool_types,                   ONLY: pw_pool_type
   USE xc_derivative_desc,              ONLY: create_split_desc
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'xc_derivative_types'

   PUBLIC :: xc_derivative_type, xc_derivative_p_type
   PUBLIC :: xc_derivative_create, xc_derivative_release, &
             xc_derivative_get

! **************************************************************************************************
!> \brief represent a derivative of a functional
! **************************************************************************************************
   TYPE xc_derivative_type
      INTEGER, DIMENSION(:), POINTER                :: split_desc => NULL()
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER, CONTIGUOUS    :: deriv_data => NULL()
   END TYPE xc_derivative_type

! **************************************************************************************************
!> \brief represent a pointer to a derivative (to have arrays of derivatives)
!> \param deriv the pointer to the derivative
!> \par History
!>      11.2003 created [fawzi]
!> \author fawzi
! **************************************************************************************************
   TYPE xc_derivative_p_type
      TYPE(xc_derivative_type), POINTER :: deriv => NULL()
   END TYPE xc_derivative_p_type

CONTAINS

! **************************************************************************************************
!> \brief allocates and initializes a derivative type
!> \param derivative the object to create
!> \param desc the derivative description
!> \param r3d_ptr the data array (the ownership of it passes to the
!>                      derivative type), the array is not zeroed
! **************************************************************************************************
   SUBROUTINE xc_derivative_create(derivative, desc, r3d_ptr)

      TYPE(xc_derivative_type)                           :: derivative
      INTEGER, DIMENSION(:), INTENT(in)                  :: desc
      REAL(kind=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         POINTER                                         :: r3d_ptr

      CALL create_split_desc(desc, derivative%split_desc)
      derivative%deriv_data => r3d_ptr

   END SUBROUTINE xc_derivative_create

! **************************************************************************************************
!> \brief allocates and initializes a derivative type
!> \param derivative the object to create
!> \param pw_pool if given gives back the cr3d array %deriv_data back to it
!>                      instead of deallocating it
! **************************************************************************************************
   SUBROUTINE xc_derivative_release(derivative, pw_pool)

      TYPE(xc_derivative_type)                           :: derivative
      TYPE(pw_pool_type), OPTIONAL, POINTER              :: pw_pool

      IF (PRESENT(pw_pool)) THEN
         IF (ASSOCIATED(pw_pool)) THEN
            CALL pw_pool%give_back_cr3d(derivative%deriv_data)
         END IF
      END IF
      IF (ASSOCIATED(derivative%deriv_data)) THEN
         DEALLOCATE (derivative%deriv_data)
      END IF
      IF (ASSOCIATED(derivative%split_desc)) DEALLOCATE (derivative%split_desc)

   END SUBROUTINE xc_derivative_release

! **************************************************************************************************
!> \brief returns various information on the given derivative
!> \param deriv the derivative you want information about
!> \param split_desc an array that describes the derivative (each position represents a variable,
!>        see xc_derivative_desc.F)
!> \param order the order of the derivative
!> \param deriv_data the 3d real array with the derivative
!> \param accept_null_data if deriv_data can be unassociated (defaults to no)
! **************************************************************************************************
   SUBROUTINE xc_derivative_get(deriv, split_desc, &
                                order, deriv_data, accept_null_data)
      TYPE(xc_derivative_type), INTENT(IN)               :: deriv
      INTEGER, DIMENSION(:), OPTIONAL, POINTER           :: split_desc
      INTEGER, INTENT(out), OPTIONAL                     :: order
      REAL(kind=dp), DIMENSION(:, :, :), OPTIONAL, &
         POINTER                                         :: deriv_data
      LOGICAL, INTENT(in), OPTIONAL                      :: accept_null_data

      LOGICAL                                            :: my_accept_null_data

      my_accept_null_data = .FALSE.
      IF (PRESENT(accept_null_data)) my_accept_null_data = accept_null_data

      IF (PRESENT(split_desc)) split_desc => deriv%split_desc
      IF (PRESENT(deriv_data)) THEN
         deriv_data => deriv%deriv_data
         IF (.NOT. my_accept_null_data) THEN
            CPASSERT(ASSOCIATED(deriv_data))
         END IF
      END IF
      IF (PRESENT(order)) order = SIZE(deriv%split_desc)
   END SUBROUTINE xc_derivative_get

END MODULE xc_derivative_types

