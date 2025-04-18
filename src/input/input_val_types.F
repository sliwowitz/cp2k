!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief a wrapper for basic fortran types.
!> \par History
!>      06.2004 created
!> \author fawzi
! **************************************************************************************************
MODULE input_val_types

   USE cp_parser_types,                 ONLY: default_continuation_character
   USE cp_units,                        ONLY: cp_unit_create,&
                                              cp_unit_desc,&
                                              cp_unit_from_cp2k,&
                                              cp_unit_from_cp2k1,&
                                              cp_unit_release,&
                                              cp_unit_type
   USE input_enumeration_types,         ONLY: enum_i2c,&
                                              enum_release,&
                                              enum_retain,&
                                              enumeration_type
   USE kinds,                           ONLY: default_string_length,&
                                              dp
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'input_val_types'

   PUBLIC :: val_p_type, val_type
   PUBLIC :: val_create, val_retain, val_release, val_get, val_write, &
             val_write_internal, val_duplicate

   INTEGER, PARAMETER, PUBLIC :: no_t = 0, logical_t = 1, &
                                 integer_t = 2, real_t = 3, char_t = 4, enum_t = 5, lchar_t = 6

! **************************************************************************************************
!> \brief pointer to a val, to create arrays of pointers
!> \param val to pointer to the val
!> \author fawzi
! **************************************************************************************************
   TYPE val_p_type
      TYPE(val_type), POINTER :: val => NULL()
   END TYPE val_p_type

! **************************************************************************************************
!> \brief a type to  have a wrapper that stores any basic fortran type
!> \param type_of_var type stored in the val (should be one of no_t,
!>        integer_t, logical_t, real_t, char_t)
!> \param l_val , i_val, c_val, r_val: arrays with logical,integer,character
!>        or real values. Only one should be associated (and namely the one
!>        specified in type_of_var).
!> \param enum an enumaration to map char to integers
!> \author fawzi
! **************************************************************************************************
   TYPE val_type
      INTEGER :: ref_count = 0, type_of_var = no_t
      LOGICAL, DIMENSION(:), POINTER :: l_val => NULL()
      INTEGER, DIMENSION(:), POINTER :: i_val => NULL()
      CHARACTER(len=default_string_length), DIMENSION(:), POINTER :: &
         c_val => NULL()
      REAL(kind=dp), DIMENSION(:), POINTER :: r_val => NULL()
      TYPE(enumeration_type), POINTER :: enum => NULL()
   END TYPE val_type
CONTAINS

! **************************************************************************************************
!> \brief creates a keyword value
!> \param val the object to be created
!> \param l_val ,i_val,r_val,c_val,lc_val: a logical,integer,real,string, long
!>        string to be stored in the val
!> \param l_vals , i_vals, r_vals, c_vals: an array of logicals,
!>        integers, reals, characters, long strings to be stored in val
!> \param l_vals_ptr , i_vals_ptr, r_vals_ptr, c_vals_ptr: an array of logicals,
!>        ... to be stored in val, val will get the ownership of the pointer
!> \param i_val ...
!> \param i_vals ...
!> \param i_vals_ptr ...
!> \param r_val ...
!> \param r_vals ...
!> \param r_vals_ptr ...
!> \param c_val ...
!> \param c_vals ...
!> \param c_vals_ptr ...
!> \param lc_val ...
!> \param lc_vals ...
!> \param lc_vals_ptr ...
!> \param enum the enumaration type this value is using
!> \author fawzi
!> \note
!>      using an enumeration only i_val/i_vals/i_vals_ptr are accepted
! **************************************************************************************************
   SUBROUTINE val_create(val, l_val, l_vals, l_vals_ptr, i_val, i_vals, i_vals_ptr, &
                         r_val, r_vals, r_vals_ptr, c_val, c_vals, c_vals_ptr, lc_val, lc_vals, &
                         lc_vals_ptr, enum)

      TYPE(val_type), POINTER                            :: val
      LOGICAL, INTENT(in), OPTIONAL                      :: l_val
      LOGICAL, DIMENSION(:), INTENT(in), OPTIONAL        :: l_vals
      LOGICAL, DIMENSION(:), OPTIONAL, POINTER           :: l_vals_ptr
      INTEGER, INTENT(in), OPTIONAL                      :: i_val
      INTEGER, DIMENSION(:), INTENT(in), OPTIONAL        :: i_vals
      INTEGER, DIMENSION(:), OPTIONAL, POINTER           :: i_vals_ptr
      REAL(KIND=DP), INTENT(in), OPTIONAL                :: r_val
      REAL(KIND=DP), DIMENSION(:), INTENT(in), OPTIONAL  :: r_vals
      REAL(KIND=DP), DIMENSION(:), OPTIONAL, POINTER     :: r_vals_ptr
      CHARACTER(LEN=*), INTENT(in), OPTIONAL             :: c_val
      CHARACTER(LEN=*), DIMENSION(:), INTENT(in), &
         OPTIONAL                                        :: c_vals
      CHARACTER(LEN=default_string_length), &
         DIMENSION(:), OPTIONAL, POINTER                 :: c_vals_ptr
      CHARACTER(LEN=*), INTENT(in), OPTIONAL             :: lc_val
      CHARACTER(LEN=*), DIMENSION(:), INTENT(in), &
         OPTIONAL                                        :: lc_vals
      CHARACTER(LEN=default_string_length), &
         DIMENSION(:), OPTIONAL, POINTER                 :: lc_vals_ptr
      TYPE(enumeration_type), OPTIONAL, POINTER          :: enum

      INTEGER                                            :: i, len_c, narg, nVal

      CPASSERT(.NOT. ASSOCIATED(val))
      ALLOCATE (val)
      NULLIFY (val%l_val, val%i_val, val%r_val, val%c_val, val%enum)
      val%type_of_var = no_t
      val%ref_count = 1

      narg = 0
      val%type_of_var = no_t
      IF (PRESENT(l_val)) THEN
         narg = narg + 1
         ALLOCATE (val%l_val(1))
         val%l_val(1) = l_val
         val%type_of_var = logical_t
      END IF
      IF (PRESENT(l_vals)) THEN
         narg = narg + 1
         ALLOCATE (val%l_val(SIZE(l_vals)))
         val%l_val = l_vals
         val%type_of_var = logical_t
      END IF
      IF (PRESENT(l_vals_ptr)) THEN
         narg = narg + 1
         val%l_val => l_vals_ptr
         val%type_of_var = logical_t
      END IF

      IF (PRESENT(r_val)) THEN
         narg = narg + 1
         ALLOCATE (val%r_val(1))
         val%r_val(1) = r_val
         val%type_of_var = real_t
      END IF
      IF (PRESENT(r_vals)) THEN
         narg = narg + 1
         ALLOCATE (val%r_val(SIZE(r_vals)))
         val%r_val = r_vals
         val%type_of_var = real_t
      END IF
      IF (PRESENT(r_vals_ptr)) THEN
         narg = narg + 1
         val%r_val => r_vals_ptr
         val%type_of_var = real_t
      END IF

      IF (PRESENT(i_val)) THEN
         narg = narg + 1
         ALLOCATE (val%i_val(1))
         val%i_val(1) = i_val
         val%type_of_var = integer_t
      END IF
      IF (PRESENT(i_vals)) THEN
         narg = narg + 1
         ALLOCATE (val%i_val(SIZE(i_vals)))
         val%i_val = i_vals
         val%type_of_var = integer_t
      END IF
      IF (PRESENT(i_vals_ptr)) THEN
         narg = narg + 1
         val%i_val => i_vals_ptr
         val%type_of_var = integer_t
      END IF

      IF (PRESENT(c_val)) THEN
         CPASSERT(LEN_TRIM(c_val) <= default_string_length)
         narg = narg + 1
         ALLOCATE (val%c_val(1))
         val%c_val(1) = c_val
         val%type_of_var = char_t
      END IF
      IF (PRESENT(c_vals)) THEN
         CPASSERT(ALL(LEN_TRIM(c_vals) <= default_string_length))
         narg = narg + 1
         ALLOCATE (val%c_val(SIZE(c_vals)))
         val%c_val = c_vals
         val%type_of_var = char_t
      END IF
      IF (PRESENT(c_vals_ptr)) THEN
         narg = narg + 1
         val%c_val => c_vals_ptr
         val%type_of_var = char_t
      END IF
      IF (PRESENT(lc_val)) THEN
         narg = narg + 1
         len_c = LEN_TRIM(lc_val)
         nVal = MAX(1, CEILING(REAL(len_c, dp)/80._dp))
         ALLOCATE (val%c_val(nVal))

         IF (len_c == 0) THEN
            val%c_val(1) = ""
         ELSE
            DO i = 1, nVal
               val%c_val(i) = lc_val((i - 1)*default_string_length + 1: &
                                     MIN(len_c, i*default_string_length))
            END DO
         END IF
         val%type_of_var = lchar_t
      END IF
      IF (PRESENT(lc_vals)) THEN
         CPASSERT(ALL(LEN_TRIM(lc_vals) <= default_string_length))
         narg = narg + 1
         ALLOCATE (val%c_val(SIZE(lc_vals)))
         val%c_val = lc_vals
         val%type_of_var = lchar_t
      END IF
      IF (PRESENT(lc_vals_ptr)) THEN
         narg = narg + 1
         val%c_val => lc_vals_ptr
         val%type_of_var = lchar_t
      END IF
      CPASSERT(narg <= 1)
      IF (PRESENT(enum)) THEN
         IF (ASSOCIATED(enum)) THEN
            IF (val%type_of_var /= no_t .AND. val%type_of_var /= integer_t .AND. &
                val%type_of_var /= enum_t) THEN
               CPABORT("")
            END IF
            IF (ASSOCIATED(val%i_val)) THEN
               val%type_of_var = enum_t
               val%enum => enum
               CALL enum_retain(enum)
            END IF
         END IF
      END IF

      CPASSERT(ASSOCIATED(val%enum) .EQV. val%type_of_var == enum_t)

   END SUBROUTINE val_create

! **************************************************************************************************
!> \brief releases the given val
!> \param val the val to release
!> \author fawzi
! **************************************************************************************************
   SUBROUTINE val_release(val)

      TYPE(val_type), POINTER                            :: val

      IF (ASSOCIATED(val)) THEN
         CPASSERT(val%ref_count > 0)
         val%ref_count = val%ref_count - 1
         IF (val%ref_count == 0) THEN
            IF (ASSOCIATED(val%l_val)) THEN
               DEALLOCATE (val%l_val)
            END IF
            IF (ASSOCIATED(val%i_val)) THEN
               DEALLOCATE (val%i_val)
            END IF
            IF (ASSOCIATED(val%r_val)) THEN
               DEALLOCATE (val%r_val)
            END IF
            IF (ASSOCIATED(val%c_val)) THEN
               DEALLOCATE (val%c_val)
            END IF
            CALL enum_release(val%enum)
            val%type_of_var = no_t
            DEALLOCATE (val)
         END IF
      END IF

      NULLIFY (val)

   END SUBROUTINE val_release

! **************************************************************************************************
!> \brief retains the given val
!> \param val the val to retain
!> \author fawzi
! **************************************************************************************************
   SUBROUTINE val_retain(val)

      TYPE(val_type), POINTER                            :: val

      CPASSERT(ASSOCIATED(val))
      CPASSERT(val%ref_count > 0)
      val%ref_count = val%ref_count + 1

   END SUBROUTINE val_retain

! **************************************************************************************************
!> \brief returns the stored values
!> \param val the object from which you want to extract the values
!> \param has_l ...
!> \param has_i ...
!> \param has_r ...
!> \param has_lc ...
!> \param has_c ...
!> \param l_val gets a logical from the val
!> \param l_vals gets an array of logicals from the val
!> \param i_val gets an integer from the val
!> \param i_vals gets an array of integers from the val
!> \param r_val gets a real from the val
!> \param r_vals gets an array of reals from the val
!> \param c_val gets a char from the val
!> \param c_vals gets an array of chars from the val
!> \param len_c len_trim of c_val (if it was a lc_val, of type lchar_t
!>        it might be longet than default_string_length)
!> \param type_of_var ...
!> \param enum ...
!> \author fawzi
!> \note
!>      using an enumeration only i_val/i_vals/i_vals_ptr are accepted
!>      add something like ignore_string_cut that if true does not warn if
!>      the c_val is too short to contain the string
! **************************************************************************************************
   SUBROUTINE val_get(val, has_l, has_i, has_r, has_lc, has_c, l_val, l_vals, i_val, &
                      i_vals, r_val, r_vals, c_val, c_vals, len_c, type_of_var, enum)

      TYPE(val_type), POINTER                            :: val
      LOGICAL, INTENT(out), OPTIONAL                     :: has_l, has_i, has_r, has_lc, has_c, l_val
      LOGICAL, DIMENSION(:), OPTIONAL, POINTER           :: l_vals
      INTEGER, INTENT(out), OPTIONAL                     :: i_val
      INTEGER, DIMENSION(:), OPTIONAL, POINTER           :: i_vals
      REAL(KIND=DP), INTENT(out), OPTIONAL               :: r_val
      REAL(KIND=DP), DIMENSION(:), OPTIONAL, POINTER     :: r_vals
      CHARACTER(LEN=*), INTENT(out), OPTIONAL            :: c_val
      CHARACTER(LEN=default_string_length), &
         DIMENSION(:), OPTIONAL, POINTER                 :: c_vals
      INTEGER, INTENT(out), OPTIONAL                     :: len_c, type_of_var
      TYPE(enumeration_type), OPTIONAL, POINTER          :: enum

      INTEGER                                            :: i, l_in, l_out

      IF (PRESENT(has_l)) has_l = ASSOCIATED(val%l_val)
      IF (PRESENT(has_i)) has_i = ASSOCIATED(val%i_val)
      IF (PRESENT(has_r)) has_r = ASSOCIATED(val%r_val)
      IF (PRESENT(has_c)) has_c = ASSOCIATED(val%c_val) ! use type_of_var?
      IF (PRESENT(has_lc)) has_lc = (val%type_of_var == lchar_t)
      IF (PRESENT(l_vals)) l_vals => val%l_val
      IF (PRESENT(l_val)) THEN
         IF (ASSOCIATED(val%l_val)) THEN
            IF (SIZE(val%l_val) > 0) THEN
               l_val = val%l_val(1)
            ELSE
               CPABORT("")
            END IF
         ELSE
            CPABORT("")
         END IF
      END IF

      IF (PRESENT(i_vals)) i_vals => val%i_val
      IF (PRESENT(i_val)) THEN
         IF (ASSOCIATED(val%i_val)) THEN
            IF (SIZE(val%i_val) > 0) THEN
               i_val = val%i_val(1)
            ELSE
               CPABORT("")
            END IF
         ELSE
            CPABORT("")
         END IF
      END IF

      IF (PRESENT(r_vals)) r_vals => val%r_val
      IF (PRESENT(r_val)) THEN
         IF (ASSOCIATED(val%r_val)) THEN
            IF (SIZE(val%r_val) > 0) THEN
               r_val = val%r_val(1)
            ELSE
               CPABORT("")
            END IF
         ELSE
            CPABORT("")
         END IF
      END IF

      IF (PRESENT(c_vals)) c_vals => val%c_val
      IF (PRESENT(c_val)) THEN
         l_out = LEN(c_val)
         IF (ASSOCIATED(val%c_val)) THEN
            IF (SIZE(val%c_val) > 0) THEN
               IF (val%type_of_var == lchar_t) THEN
                  l_in = default_string_length*(SIZE(val%c_val) - 1) + &
                         LEN_TRIM(val%c_val(SIZE(val%c_val)))
                  IF (l_out < l_in) &
                     CALL cp_warn(__LOCATION__, &
                                  "val_get will truncate value, value beginning with '"// &
                                  TRIM(val%c_val(1))//"' is too long for variable")
                  DO i = 1, SIZE(val%c_val)
                     c_val((i - 1)*default_string_length + 1:MIN(l_out, i*default_string_length)) = &
                        val%c_val(i) (1:MIN(80, l_out - (i - 1)*default_string_length))
                     IF (l_out <= i*default_string_length) EXIT
                  END DO
                  IF (l_out > SIZE(val%c_val)*default_string_length) &
                     c_val(SIZE(val%c_val)*default_string_length + 1:l_out) = ""
               ELSE
                  l_in = LEN_TRIM(val%c_val(1))
                  IF (l_out < l_in) &
                     CALL cp_warn(__LOCATION__, &
                                  "val_get will truncate value, value '"// &
                                  TRIM(val%c_val(1))//"' is too long for variable")
                  c_val = val%c_val(1)
               END IF
            ELSE
               CPABORT("")
            END IF
         ELSE IF (ASSOCIATED(val%i_val) .AND. ASSOCIATED(val%enum)) THEN
            IF (SIZE(val%i_val) > 0) THEN
               c_val = enum_i2c(val%enum, val%i_val(1))
            ELSE
               CPABORT("")
            END IF
         ELSE
            CPABORT("")
         END IF
      END IF

      IF (PRESENT(len_c)) THEN
         IF (ASSOCIATED(val%c_val)) THEN
            IF (SIZE(val%c_val) > 0) THEN
               IF (val%type_of_var == lchar_t) THEN
                  len_c = default_string_length*(SIZE(val%c_val) - 1) + &
                          LEN_TRIM(val%c_val(SIZE(val%c_val)))
               ELSE
                  len_c = LEN_TRIM(val%c_val(1))
               END IF
            ELSE
               len_c = -HUGE(0)
            END IF
         ELSE IF (ASSOCIATED(val%i_val) .AND. ASSOCIATED(val%enum)) THEN
            IF (SIZE(val%i_val) > 0) THEN
               len_c = LEN_TRIM(enum_i2c(val%enum, val%i_val(1)))
            ELSE
               len_c = -HUGE(0)
            END IF
         ELSE
            len_c = -HUGE(0)
         END IF
      END IF

      IF (PRESENT(type_of_var)) type_of_var = val%type_of_var

      IF (PRESENT(enum)) enum => val%enum

   END SUBROUTINE val_get

! **************************************************************************************************
!> \brief writes out the values stored in the val
!> \param val the val to write
!> \param unit_nr the number of the unit to write to
!> \param unit the unit of mesure in which the output should be written
!>        (overrides unit_str)
!> \param unit_str the unit of mesure in which the output should be written
!> \param fmt ...
!> \author fawzi
!> \note
!>      unit of mesure used only for reals
! **************************************************************************************************
   SUBROUTINE val_write(val, unit_nr, unit, unit_str, fmt)

      TYPE(val_type), POINTER                            :: val
      INTEGER, INTENT(in)                                :: unit_nr
      TYPE(cp_unit_type), OPTIONAL, POINTER              :: unit
      CHARACTER(len=*), INTENT(in), OPTIONAL             :: unit_str, fmt

      CHARACTER(len=default_string_length)               :: c_string, myfmt, rcval
      INTEGER                                            :: i, iend, item, j, l
      LOGICAL                                            :: owns_unit
      TYPE(cp_unit_type), POINTER                        :: my_unit

      NULLIFY (my_unit)
      myfmt = ""
      owns_unit = .FALSE.

      IF (PRESENT(fmt)) myfmt = fmt
      IF (PRESENT(unit)) my_unit => unit
      IF (.NOT. ASSOCIATED(my_unit) .AND. PRESENT(unit_str)) THEN
         ALLOCATE (my_unit)
         CALL cp_unit_create(my_unit, unit_str)
         owns_unit = .TRUE.
      END IF

      IF (ASSOCIATED(val)) THEN
         SELECT CASE (val%type_of_var)
         CASE (logical_t)
            IF (ASSOCIATED(val%l_val)) THEN
               DO i = 1, SIZE(val%l_val)
                  IF (MODULO(i, 20) == 0) THEN
                     WRITE (UNIT=unit_nr, FMT="(1X,A1)") default_continuation_character
                     WRITE (UNIT=unit_nr, FMT="("//TRIM(myfmt)//")", ADVANCE="NO")
                  END IF
                  WRITE (UNIT=unit_nr, FMT="(1X,L1)", ADVANCE="NO") &
                     val%l_val(i)
               END DO
            ELSE
               CPABORT("Input value of type <logical_t> not associated")
            END IF
         CASE (integer_t)
            IF (ASSOCIATED(val%i_val)) THEN
               item = 0
               i = 1
               loop_i: DO WHILE (i <= SIZE(val%i_val))
                  item = item + 1
                  IF (MODULO(item, 10) == 0) THEN
                     WRITE (UNIT=unit_nr, FMT="(1X,A)") default_continuation_character
                     WRITE (UNIT=unit_nr, FMT="("//TRIM(myfmt)//")", ADVANCE="NO")
                  END IF
                  iend = i
                  loop_j: DO j = i + 1, SIZE(val%i_val)
                     IF (val%i_val(j - 1) + 1 == val%i_val(j)) THEN
                        iend = iend + 1
                     ELSE
                        EXIT loop_j
                     END IF
                  END DO loop_j
                  IF ((iend - i) > 1) THEN
                     WRITE (UNIT=unit_nr, FMT="(1X,I0,A2,I0)", ADVANCE="NO") &
                        val%i_val(i), "..", val%i_val(iend)
                     i = iend
                  ELSE
                     WRITE (UNIT=unit_nr, FMT="(1X,I0)", ADVANCE="NO") &
                        val%i_val(i)
                  END IF
                  i = i + 1
               END DO loop_i
            ELSE
               CPABORT("Input value of type <integer_t> not associated")
            END IF
         CASE (real_t)
            IF (ASSOCIATED(val%r_val)) THEN
               DO i = 1, SIZE(val%r_val)
                  IF (MODULO(i, 5) == 0) THEN
                     WRITE (UNIT=unit_nr, FMT="(1X,A)") default_continuation_character
                     WRITE (UNIT=unit_nr, FMT="("//TRIM(myfmt)//")", ADVANCE="NO")
                  END IF
                  IF (ASSOCIATED(my_unit)) THEN
                     WRITE (UNIT=rcval, FMT="(ES25.16E3)") &
                        cp_unit_from_cp2k1(val%r_val(i), my_unit)
                  ELSE
                     WRITE (UNIT=rcval, FMT="(ES25.16E3)") val%r_val(i)
                  END IF
                  WRITE (UNIT=unit_nr, FMT="(A)", ADVANCE="NO") TRIM(rcval)
               END DO
            ELSE
               CPABORT("Input value of type <real_t> not associated")
            END IF
         CASE (char_t)
            IF (ASSOCIATED(val%c_val)) THEN
               l = 0
               DO i = 1, SIZE(val%c_val)
                  l = l + 1
                  IF (l > 10 .AND. l + LEN_TRIM(val%c_val(i)) > 76) THEN
                     WRITE (UNIT=unit_nr, FMT="(A1)") default_continuation_character
                     WRITE (UNIT=unit_nr, FMT="("//TRIM(myfmt)//")", ADVANCE="NO")
                     l = 0
                     WRITE (UNIT=unit_nr, FMT="(1X,A)", ADVANCE="NO") """"//TRIM(val%c_val(i))//""""
                     l = l + LEN_TRIM(val%c_val(i)) + 3
                  ELSE IF (LEN_TRIM(val%c_val(i)) > 0) THEN
                     l = l + LEN_TRIM(val%c_val(i))
                     WRITE (UNIT=unit_nr, FMT="(1X,A)", ADVANCE="NO") """"//TRIM(val%c_val(i))//""""
                  ELSE
                     l = l + 3
                     WRITE (UNIT=unit_nr, FMT="(1X,A)", ADVANCE="NO") '""'
                  END IF
               END DO
            ELSE
               CPABORT("Input value of type <char_t> not associated")
            END IF
         CASE (lchar_t)
            IF (ASSOCIATED(val%c_val)) THEN
               SELECT CASE (SIZE(val%c_val))
               CASE (1)
                  WRITE (UNIT=unit_nr, FMT='(1X,A)', ADVANCE="NO") TRIM(val%c_val(1))
               CASE (2)
                  WRITE (UNIT=unit_nr, FMT='(1X,A)', ADVANCE="NO") val%c_val(1)
                  WRITE (UNIT=unit_nr, FMT='(A)', ADVANCE="NO") TRIM(val%c_val(2))
               CASE (3:)
                  WRITE (UNIT=unit_nr, FMT='(1X,A)', ADVANCE="NO") val%c_val(1)
                  DO i = 2, SIZE(val%c_val) - 1
                     WRITE (UNIT=unit_nr, FMT="(A)", ADVANCE="NO") val%c_val(i)
                  END DO
                  WRITE (UNIT=unit_nr, FMT='(A)', ADVANCE="NO") TRIM(val%c_val(SIZE(val%c_val)))
               END SELECT
            ELSE
               CPABORT("Input value of type <lchar_t> not associated")
            END IF
         CASE (enum_t)
            IF (ASSOCIATED(val%i_val)) THEN
               l = 0
               DO i = 1, SIZE(val%i_val)
                  c_string = enum_i2c(val%enum, val%i_val(i))
                  IF (l > 10 .AND. l + LEN_TRIM(c_string) > 76) THEN
                     WRITE (UNIT=unit_nr, FMT="(1X,A)") default_continuation_character
                     WRITE (UNIT=unit_nr, FMT="("//TRIM(myfmt)//")", ADVANCE="NO")
                     l = 0
                  ELSE
                     l = l + LEN_TRIM(c_string) + 3
                  END IF
                  WRITE (UNIT=unit_nr, FMT="(1X,A)", ADVANCE="NO") TRIM(c_string)
               END DO
            ELSE
               CPABORT("Input value of type <enum_t> not associated")
            END IF
         CASE (no_t)
            WRITE (UNIT=unit_nr, FMT="(' *empty*')", ADVANCE="NO")
         CASE default
            CPABORT("Unexpected type_of_var for val")
         END SELECT
      ELSE
         WRITE (UNIT=unit_nr, FMT="(1X,A)", ADVANCE="NO") "NULL()"
      END IF

      IF (owns_unit) THEN
         CALL cp_unit_release(my_unit)
         DEALLOCATE (my_unit)
      END IF

      WRITE (UNIT=unit_nr, FMT="()")

   END SUBROUTINE val_write

! **************************************************************************************************
!> \brief   Write values to an internal file, i.e. string variable.
!> \param val ...
!> \param string ...
!> \param unit ...
!> \date    10.03.2005
!> \par History
!>          17.01.2006, MK, Optional argument unit for the conversion to the external unit added
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE val_write_internal(val, string, unit)

      TYPE(val_type), POINTER                            :: val
      CHARACTER(LEN=*), INTENT(OUT)                      :: string
      TYPE(cp_unit_type), OPTIONAL, POINTER              :: unit

      CHARACTER(LEN=default_string_length)               :: enum_string
      INTEGER                                            :: i, ipos
      REAL(KIND=dp)                                      :: value

      string = ""

      IF (ASSOCIATED(val)) THEN

         SELECT CASE (val%type_of_var)
         CASE (logical_t)
            IF (ASSOCIATED(val%l_val)) THEN
               DO i = 1, SIZE(val%l_val)
                  WRITE (UNIT=string(2*i - 1:), FMT="(1X,L1)") val%l_val(i)
               END DO
            ELSE
               CPABORT("")
            END IF
         CASE (integer_t)
            IF (ASSOCIATED(val%i_val)) THEN
               DO i = 1, SIZE(val%i_val)
                  WRITE (UNIT=string(12*i - 11:), FMT="(I12)") val%i_val(i)
               END DO
            ELSE
               CPABORT("")
            END IF
         CASE (real_t)
            IF (ASSOCIATED(val%r_val)) THEN
               IF (PRESENT(unit)) THEN
                  DO i = 1, SIZE(val%r_val)
                     value = cp_unit_from_cp2k(value=val%r_val(i), &
                                               unit_str=cp_unit_desc(unit=unit))
                     WRITE (UNIT=string(17*i - 16:), FMT="(ES17.8E3)") value
                  END DO
               ELSE
                  DO i = 1, SIZE(val%r_val)
                     WRITE (UNIT=string(17*i - 16:), FMT="(ES17.8E3)") val%r_val(i)
                  END DO
               END IF
            ELSE
               CPABORT("")
            END IF
         CASE (char_t)
            IF (ASSOCIATED(val%c_val)) THEN
               ipos = 1
               DO i = 1, SIZE(val%c_val)
                  WRITE (UNIT=string(ipos:), FMT="(A)") TRIM(ADJUSTL(val%c_val(i)))
                  ipos = ipos + LEN_TRIM(ADJUSTL(val%c_val(i))) + 1
               END DO
            ELSE
               CPABORT("")
            END IF
         CASE (lchar_t)
            IF (ASSOCIATED(val%c_val)) THEN
               CALL val_get(val, c_val=string)
            ELSE
               CPABORT("")
            END IF
         CASE (enum_t)
            IF (ASSOCIATED(val%i_val)) THEN
               DO i = 1, SIZE(val%i_val)
                  enum_string = enum_i2c(val%enum, val%i_val(i))
                  WRITE (UNIT=string, FMT="(A)") TRIM(ADJUSTL(enum_string))
               END DO
            ELSE
               CPABORT("")
            END IF
         CASE default
            CPABORT("unexpected type_of_var for val ")
         END SELECT

      END IF

   END SUBROUTINE val_write_internal

! **************************************************************************************************
!> \brief creates a copy of the given value
!> \param val_in the value to copy
!> \param val_out the value tha will be created
!> \author fawzi
! **************************************************************************************************
   SUBROUTINE val_duplicate(val_in, val_out)

      TYPE(val_type), POINTER                            :: val_in, val_out

      CPASSERT(ASSOCIATED(val_in))
      CPASSERT(.NOT. ASSOCIATED(val_out))
      ALLOCATE (val_out)
      val_out%type_of_var = val_in%type_of_var
      val_out%ref_count = 1
      val_out%enum => val_in%enum
      IF (ASSOCIATED(val_out%enum)) CALL enum_retain(val_out%enum)

      NULLIFY (val_out%l_val, val_out%i_val, val_out%c_val, val_out%r_val)
      IF (ASSOCIATED(val_in%l_val)) THEN
         ALLOCATE (val_out%l_val(SIZE(val_in%l_val)))
         val_out%l_val = val_in%l_val
      END IF
      IF (ASSOCIATED(val_in%i_val)) THEN
         ALLOCATE (val_out%i_val(SIZE(val_in%i_val)))
         val_out%i_val = val_in%i_val
      END IF
      IF (ASSOCIATED(val_in%r_val)) THEN
         ALLOCATE (val_out%r_val(SIZE(val_in%r_val)))
         val_out%r_val = val_in%r_val
      END IF
      IF (ASSOCIATED(val_in%c_val)) THEN
         ALLOCATE (val_out%c_val(SIZE(val_in%c_val)))
         val_out%c_val = val_in%c_val
      END IF

   END SUBROUTINE val_duplicate

END MODULE input_val_types
