!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief a module to allow simple buffering of read lines of a parser
!> \author Teodoro Laino [tlaino] - University of Zurich
!> \date 08.2008
! **************************************************************************************************
MODULE cp_parser_buffer_types

   USE kinds,                           ONLY: max_line_length
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

! ****************************************************************************
!> \brief  Buffer type for speeding-up the parsing in parallel
!> \author Teodoro Laino [tlaino] - University of Zurich
!> \date   08.2008
! **************************************************************************************************
   TYPE buffer_type
      INTEGER                              :: size = 0, buffer_id = 0
      INTEGER                              :: present_line_number = 0, &
                                              last_line_number = 0, &
                                              istat = 0
      INTEGER, DIMENSION(:), POINTER       :: input_line_numbers => NULL()
      CHARACTER(LEN=max_line_length), &
         DIMENSION(:), POINTER           :: input_lines => NULL()
      TYPE(buffer_type), POINTER           :: sub_buffer => NULL()
   END TYPE buffer_type

   PUBLIC :: buffer_type, create_buffer_type, release_buffer_type, copy_buffer_type, &
             initialize_sub_buffer, finalize_sub_buffer
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cp_parser_buffer_types'
   INTEGER, PARAMETER, PRIVATE          :: buffer_size = 1000

CONTAINS

! ****************************************************************************
!> \brief  Creates the parser buffer type
!> \param buffer ...
!> \date   08.2008
!> \author Teodoro Laino [tlaino] - University of Zurich
! **************************************************************************************************
   SUBROUTINE create_buffer_type(buffer)
      TYPE(buffer_type), POINTER                         :: buffer

      CPASSERT(.NOT. ASSOCIATED(buffer))
      ALLOCATE (buffer)
      buffer%size = buffer_size
      ALLOCATE (buffer%input_lines(buffer%size))
      ALLOCATE (buffer%input_line_numbers(buffer%size))
      buffer%present_line_number = buffer%size
      buffer%last_line_number = buffer%size
   END SUBROUTINE create_buffer_type

! ****************************************************************************
!> \brief  Releases the parser buffer type
!> \param buffer ...
!> \date   08.2008
!> \author Teodoro Laino [tlaino] - University of Zurich
! **************************************************************************************************
   RECURSIVE SUBROUTINE release_buffer_type(buffer)
      TYPE(buffer_type), POINTER                         :: buffer

      CPASSERT(ASSOCIATED(buffer))
      DEALLOCATE (buffer%input_lines)
      DEALLOCATE (buffer%input_line_numbers)
      IF (ASSOCIATED(buffer%sub_buffer)) THEN
         CALL release_buffer_type(buffer%sub_buffer)
      END IF
      DEALLOCATE (buffer)
   END SUBROUTINE release_buffer_type

! ****************************************************************************
!> \brief  Copies  buffer types
!> \param buffer_in ...
!> \param buffer_out ...
!> \param force ...
!> \date   08.2008
!> \author Teodoro Laino [tlaino] - University of Zurich
! **************************************************************************************************
   RECURSIVE SUBROUTINE copy_buffer_type(buffer_in, buffer_out, force)
      TYPE(buffer_type), POINTER                         :: buffer_in, buffer_out
      LOGICAL, INTENT(IN), OPTIONAL                      :: force

      INTEGER                                            :: i
      LOGICAL                                            :: my_force

      CPASSERT(ASSOCIATED(buffer_in))
      CPASSERT(ASSOCIATED(buffer_out))
      CPASSERT(buffer_in%size == buffer_out%size)
      my_force = .FALSE.
      IF (PRESENT(force)) my_force = force
      ! Copy buffer structure
      buffer_out%present_line_number = buffer_in%present_line_number
      buffer_out%last_line_number = buffer_in%last_line_number
      buffer_out%istat = buffer_in%istat
      ! This part can be quite expensive.. we do it only when strictly necessary..
      IF ((buffer_out%buffer_id /= buffer_in%buffer_id) .OR. (my_force)) THEN
         buffer_out%buffer_id = buffer_in%buffer_id
         buffer_out%input_line_numbers = buffer_in%input_line_numbers
         ! Explicit loop: bypass a NAG bug..
         DO i = 1, SIZE(buffer_in%input_lines)
            buffer_out%input_lines(i) = buffer_in%input_lines(i)
         END DO
      END IF
      IF (ASSOCIATED(buffer_in%sub_buffer) .AND. ASSOCIATED(buffer_out%sub_buffer)) THEN
         CALL copy_buffer_type(buffer_in%sub_buffer, buffer_out%sub_buffer, force)
      END IF
   END SUBROUTINE copy_buffer_type

! ****************************************************************************
!> \brief  Initializes sub buffer structure
!> \param sub_buffer ...
!> \param buffer ...
!> \date   08.2008
!> \author Teodoro Laino [tlaino] - University of Zurich
! **************************************************************************************************
   SUBROUTINE initialize_sub_buffer(sub_buffer, buffer)
      TYPE(buffer_type), POINTER                         :: sub_buffer, buffer

      CPASSERT(ASSOCIATED(buffer))
      CPASSERT(.NOT. ASSOCIATED(sub_buffer))
      CALL create_buffer_type(sub_buffer)
      CALL copy_buffer_type(buffer, sub_buffer)
      sub_buffer%present_line_number = 0
   END SUBROUTINE initialize_sub_buffer

! ****************************************************************************
!> \brief  Finalizes sub buffer structure
!> \param sub_buffer ...
!> \param buffer ...
!> \date   08.2008
!> \author Teodoro Laino [tlaino] - University of Zurich
! **************************************************************************************************
   SUBROUTINE finalize_sub_buffer(sub_buffer, buffer)
      TYPE(buffer_type), POINTER                         :: sub_buffer, buffer

      CPASSERT(ASSOCIATED(buffer))
      CPASSERT(ASSOCIATED(sub_buffer))
      CALL copy_buffer_type(sub_buffer, buffer)
      CALL release_buffer_type(sub_buffer)
   END SUBROUTINE finalize_sub_buffer

END MODULE cp_parser_buffer_types
