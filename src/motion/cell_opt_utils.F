!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief contains a functional that calculates the energy and its derivatives
!>      for the geometry optimizer
!> \par History
!>      03.2008 - Teodoro Laino [tlaino] - University of Zurich - Cell Optimization
! **************************************************************************************************
MODULE cell_opt_utils
   USE cell_types,                      ONLY: &
        cell_sym_cubic, cell_sym_hexagonal_gamma_120, cell_sym_hexagonal_gamma_60, &
        cell_sym_monoclinic, cell_sym_monoclinic_gamma_ab, cell_sym_orthorhombic, &
        cell_sym_rhombohedral, cell_sym_tetragonal_ab, cell_sym_tetragonal_ac, &
        cell_sym_tetragonal_bc, cell_sym_triclinic, cell_type
   USE cp_files,                        ONLY: close_file,&
                                              open_file
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_create,&
                                              cp_logger_get_default_unit_nr,&
                                              cp_logger_release,&
                                              cp_logger_set,&
                                              cp_logger_type,&
                                              cp_to_string
   USE input_constants,                 ONLY: fix_none,&
                                              fix_x,&
                                              fix_xy,&
                                              fix_xz,&
                                              fix_y,&
                                              fix_yz,&
                                              fix_z
   USE input_cp2k_global,               ONLY: create_global_section
   USE input_enumeration_types,         ONLY: enum_i2c,&
                                              enumeration_type
   USE input_keyword_types,             ONLY: keyword_get,&
                                              keyword_type
   USE input_section_types,             ONLY: section_get_keyword,&
                                              section_release,&
                                              section_type,&
                                              section_vals_type,&
                                              section_vals_val_get,&
                                              section_vals_val_set
   USE kinds,                           ONLY: default_path_length,&
                                              default_string_length,&
                                              dp
   USE mathconstants,                   ONLY: sqrt3
   USE mathlib,                         ONLY: angle
   USE message_passing,                 ONLY: mp_para_env_type
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cell_opt_utils'

   PUBLIC :: get_dg_dh, gopt_new_logger_create, &
             gopt_new_logger_release, read_external_press_tensor, &
             apply_cell_constraints

CONTAINS

! **************************************************************************************************
!> \brief creates a new logger used for cell optimization algorithm
!> \param new_logger ...
!> \param root_section ...
!> \param para_env ...
!> \param project_name ...
!> \param id_run ...
!> \author Teodoro Laino [tlaino] - University of Zurich - 03.2008
! **************************************************************************************************
   SUBROUTINE gopt_new_logger_create(new_logger, root_section, para_env, project_name, &
                                     id_run)
      TYPE(cp_logger_type), POINTER                      :: new_logger
      TYPE(section_vals_type), POINTER                   :: root_section
      TYPE(mp_para_env_type), POINTER                    :: para_env
      CHARACTER(len=default_string_length), INTENT(OUT)  :: project_name
      INTEGER, INTENT(IN)                                :: id_run

      CHARACTER(len=default_path_length)                 :: c_val, input_file_path, output_file_path
      CHARACTER(len=default_string_length)               :: label
      INTEGER                                            :: i, lp, unit_nr
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(enumeration_type), POINTER                    :: enum
      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: section

      NULLIFY (new_logger, logger, enum, keyword, section)
      logger => cp_get_default_logger()

      CALL create_global_section(section)
      keyword => section_get_keyword(section, "RUN_TYPE")
      CALL keyword_get(keyword, enum=enum)
      label = TRIM(enum_i2c(enum, id_run))
      CALL section_release(section)

      ! Redirecting output of the sub_calculation to a different file
      CALL section_vals_val_get(root_section, "GLOBAL%PROJECT_NAME", c_val=project_name)
      input_file_path = TRIM(project_name)
      lp = LEN_TRIM(input_file_path)
      i = logger%iter_info%iteration(logger%iter_info%n_rlevel)
      input_file_path(lp + 1:LEN(input_file_path)) = "-"//TRIM(label)//"-"//ADJUSTL(cp_to_string(i))
      lp = LEN_TRIM(input_file_path)
      CALL section_vals_val_set(root_section, "GLOBAL%PROJECT_NAME", c_val=input_file_path(1:lp))
      CALL section_vals_val_set(root_section, "GLOBAL%RUN_TYPE", i_val=id_run)

      ! Redirecting output into a new file
      output_file_path = input_file_path(1:lp)//".out"
      IF (para_env%is_source()) THEN
         CALL open_file(file_name=output_file_path, file_status="UNKNOWN", &
                        file_action="WRITE", file_position="APPEND", unit_number=unit_nr)
      ELSE
         unit_nr = -1
      END IF
      CALL cp_logger_create(new_logger, para_env=para_env, default_global_unit_nr=unit_nr, &
                            close_global_unit_on_dealloc=.FALSE.)
      CALL section_vals_val_get(root_section, "GLOBAL%PROJECT", c_val=c_val)
      IF (c_val /= "") THEN
         CALL cp_logger_set(new_logger, local_filename=TRIM(c_val)//"_localLog")
      END IF
      new_logger%iter_info%project_name = c_val
      CALL section_vals_val_get(root_section, "GLOBAL%PRINT_LEVEL", &
                                i_val=new_logger%iter_info%print_level)

   END SUBROUTINE gopt_new_logger_create

! **************************************************************************************************
!> \brief releases a new logger used for cell optimization algorithm
!> \param new_logger ...
!> \param root_section ...
!> \param para_env ...
!> \param project_name ...
!> \param id_run ...
!> \author Teodoro Laino [tlaino] - University of Zurich - 03.2008
! **************************************************************************************************
   SUBROUTINE gopt_new_logger_release(new_logger, root_section, para_env, project_name, id_run)
      TYPE(cp_logger_type), POINTER                      :: new_logger
      TYPE(section_vals_type), POINTER                   :: root_section
      TYPE(mp_para_env_type), POINTER                    :: para_env
      CHARACTER(len=default_string_length), INTENT(IN)   :: project_name
      INTEGER, INTENT(IN)                                :: id_run

      INTEGER                                            :: unit_nr

      IF (para_env%is_source()) THEN
         unit_nr = cp_logger_get_default_unit_nr(new_logger)
         CALL close_file(unit_number=unit_nr)
      END IF
      CALL cp_logger_release(new_logger)
      CALL section_vals_val_set(root_section, "GLOBAL%RUN_TYPE", i_val=id_run)
      CALL section_vals_val_set(root_section, "GLOBAL%PROJECT_NAME", c_val=project_name)

   END SUBROUTINE gopt_new_logger_release

! **************************************************************************************************
!> \brief Reads the external pressure tensor
!> \param geo_section ...
!> \param cell ...
!> \param pres_ext ...
!> \param mtrx ...
!> \param rot ...
!> \author Teodoro Laino [tlaino] - University of Zurich - 03.2008
! **************************************************************************************************
   SUBROUTINE read_external_press_tensor(geo_section, cell, pres_ext, mtrx, rot)
      TYPE(section_vals_type), POINTER                   :: geo_section
      TYPE(cell_type), POINTER                           :: cell
      REAL(KIND=dp), INTENT(OUT)                         :: pres_ext
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(OUT)        :: mtrx
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: rot

      INTEGER                                            :: i, ind, j
      LOGICAL                                            :: check
      REAL(KIND=dp), DIMENSION(3, 3)                     :: pres_ext_tens
      REAL(KIND=dp), DIMENSION(:), POINTER               :: pvals

      NULLIFY (pvals)
      mtrx = 0.0_dp
      pres_ext_tens = 0.0_dp
      pres_ext = 0.0_dp
      CALL section_vals_val_get(geo_section, "EXTERNAL_PRESSURE", r_vals=pvals)
      check = (SIZE(pvals) == 1) .OR. (SIZE(pvals) == 9)
      IF (.NOT. check) &
         CPABORT("EXTERNAL_PRESSURE can have 1 or 9 components only!")

      IF (SIZE(pvals) == 9) THEN
         ind = 0
         DO i = 1, 3
            DO j = 1, 3
               ind = ind + 1
               pres_ext_tens(j, i) = pvals(ind)
            END DO
         END DO
         ! Also the pressure tensor must be oriented in the same canonical directions
         ! of the simulation cell
         pres_ext_tens = MATMUL(TRANSPOSE(rot), pres_ext_tens)
         DO i = 1, 3
            pres_ext = pres_ext + pres_ext_tens(i, i)
         END DO
         pres_ext = pres_ext/3.0_dp
         DO i = 1, 3
            pres_ext_tens(i, i) = pres_ext_tens(i, i) - pres_ext
         END DO
      ELSE
         pres_ext = pvals(1)
      END IF

      IF (ANY(pres_ext_tens > 1.0E-5_dp)) THEN
         mtrx = cell%deth*MATMUL(cell%h_inv, MATMUL(pres_ext_tens, TRANSPOSE(cell%h_inv)))
      END IF

   END SUBROUTINE read_external_press_tensor

! **************************************************************************************************
!> \brief Computes the derivatives for the cell
!> \param gradient ...
!> \param av_ptens ...
!> \param pres_ext ...
!> \param cell ...
!> \param mtrx ...
!> \param keep_angles ...
!> \param keep_symmetry ...
!> \param pres_int ...
!> \param pres_constr ...
!> \param constraint_id ...
!> \author Teodoro Laino [tlaino] - University of Zurich - 03.2008
! **************************************************************************************************
   SUBROUTINE get_dg_dh(gradient, av_ptens, pres_ext, cell, mtrx, keep_angles, &
                        keep_symmetry, pres_int, pres_constr, constraint_id)

      REAL(KIND=dp), DIMENSION(:), POINTER               :: gradient
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: av_ptens
      REAL(KIND=dp), INTENT(IN)                          :: pres_ext
      TYPE(cell_type), POINTER                           :: cell
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: mtrx
      LOGICAL, INTENT(IN), OPTIONAL                      :: keep_angles, keep_symmetry
      REAL(KIND=dp), INTENT(OUT)                         :: pres_int, pres_constr
      INTEGER, INTENT(IN), OPTIONAL                      :: constraint_id

      INTEGER                                            :: i, my_constraint_id
      LOGICAL                                            :: my_keep_angles, my_keep_symmetry
      REAL(KIND=dp), DIMENSION(3, 3)                     :: correction, pten_hinv_old, ptens

      my_keep_angles = .FALSE.
      IF (PRESENT(keep_angles)) my_keep_angles = keep_angles
      my_keep_symmetry = .FALSE.
      IF (PRESENT(keep_symmetry)) my_keep_symmetry = keep_symmetry
      gradient = 0.0_dp
      IF (PRESENT(constraint_id)) THEN
         my_constraint_id = constraint_id
      ELSE
         my_constraint_id = fix_none
      END IF

      gradient = 0.0_dp

      ptens = av_ptens

      ! Evaluating the internal pressure
      pres_int = 0.0_dp
      DO i = 1, 3
         pres_int = pres_int + ptens(i, i)
      END DO
      pres_int = pres_int/3.0_dp

      SELECT CASE (my_constraint_id)
      CASE (fix_x)
         pres_constr = ptens(2, 2) + ptens(3, 3)
      CASE (fix_y)
         pres_constr = ptens(1, 1) + ptens(3, 3)
      CASE (fix_z)
         pres_constr = ptens(1, 1) + ptens(2, 2)
      CASE (fix_xy)
         pres_constr = ptens(3, 3)
      CASE (fix_xz)
         pres_constr = ptens(2, 2)
      CASE (fix_yz)
         pres_constr = ptens(1, 1)
      CASE (fix_none)
         pres_constr = ptens(1, 1) + ptens(2, 2) + ptens(3, 3)
      END SELECT
      pres_constr = pres_constr/3.0_dp

      ptens(1, 1) = av_ptens(1, 1) - pres_ext
      ptens(2, 2) = av_ptens(2, 2) - pres_ext
      ptens(3, 3) = av_ptens(3, 3) - pres_ext

      pten_hinv_old = cell%deth*MATMUL(cell%h_inv, ptens)
      correction = MATMUL(mtrx, cell%hmat)

      gradient(1) = pten_hinv_old(1, 1) - correction(1, 1)
      gradient(2) = pten_hinv_old(2, 1) - correction(2, 1)
      gradient(3) = pten_hinv_old(2, 2) - correction(2, 2)
      gradient(4) = pten_hinv_old(3, 1) - correction(3, 1)
      gradient(5) = pten_hinv_old(3, 2) - correction(3, 2)
      gradient(6) = pten_hinv_old(3, 3) - correction(3, 3)

      CALL apply_cell_constraints(gradient, cell, my_keep_angles, my_keep_symmetry, my_constraint_id)

      gradient = -gradient

   END SUBROUTINE get_dg_dh

! **************************************************************************************************
!> \brief Apply cell constraints
!> \param gradient ...
!> \param cell ...
!> \param keep_angles ...
!> \param keep_symmetry ...
!> \param constraint_id ...
!> \author Matthias Krack (October 26, 2017, MK)
! **************************************************************************************************
   SUBROUTINE apply_cell_constraints(gradient, cell, keep_angles, keep_symmetry, constraint_id)

      REAL(KIND=dp), DIMENSION(:), POINTER               :: gradient
      TYPE(cell_type), POINTER                           :: cell
      LOGICAL, INTENT(IN)                                :: keep_angles, keep_symmetry
      INTEGER, INTENT(IN)                                :: constraint_id

      REAL(KIND=dp)                                      :: a, a_length, ab_length, b_length, cosa, &
                                                            cosah, cosg, deriv_gamma, g, gamma, &
                                                            norm, norm_b, norm_c, sina, sinah, sing

      IF (keep_angles) THEN
         ! If we want to keep the angles constant we have to project out the
         ! components of the cell angles
         norm_b = DOT_PRODUCT(cell%hmat(:, 2), cell%hmat(:, 2))
         norm = DOT_PRODUCT(cell%hmat(1:2, 2), gradient(2:3))
         gradient(2:3) = cell%hmat(1:2, 2)/norm_b*norm
         norm_c = DOT_PRODUCT(cell%hmat(:, 3), cell%hmat(:, 3))
         norm = DOT_PRODUCT(cell%hmat(1:3, 3), gradient(4:6))
         gradient(4:6) = cell%hmat(1:3, 3)/norm_c*norm
         ! Retain an exact orthorhombic cell
         ! (off-diagonal elements must remain zero identically to keep QS fast)
         IF (cell%orthorhombic) THEN
            gradient(2) = 0.0_dp
            gradient(4) = 0.0_dp
            gradient(5) = 0.0_dp
         END IF
      END IF

      IF (keep_symmetry) THEN
         SELECT CASE (cell%symmetry_id)
         CASE (cell_sym_cubic, &
               cell_sym_tetragonal_ab, &
               cell_sym_tetragonal_ac, &
               cell_sym_tetragonal_bc, &
               cell_sym_orthorhombic)
            SELECT CASE (cell%symmetry_id)
            CASE (cell_sym_cubic)
               g = (gradient(1) + gradient(3) + gradient(6))/3.0_dp
               gradient(1) = g
               gradient(3) = g
               gradient(6) = g
            CASE (cell_sym_tetragonal_ab, &
                  cell_sym_tetragonal_ac, &
                  cell_sym_tetragonal_bc)
               SELECT CASE (cell%symmetry_id)
               CASE (cell_sym_tetragonal_ab)
                  g = 0.5_dp*(gradient(1) + gradient(3))
                  gradient(1) = g
                  gradient(3) = g
               CASE (cell_sym_tetragonal_ac)
                  g = 0.5_dp*(gradient(1) + gradient(6))
                  gradient(1) = g
                  gradient(6) = g
               CASE (cell_sym_tetragonal_bc)
                  g = 0.5_dp*(gradient(3) + gradient(6))
                  gradient(3) = g
                  gradient(6) = g
               END SELECT
            CASE (cell_sym_orthorhombic)
               ! Nothing else to do
            END SELECT
            gradient(2) = 0.0_dp
            gradient(4) = 0.0_dp
            gradient(5) = 0.0_dp
         CASE (cell_sym_hexagonal_gamma_60)
            g = 0.5_dp*(gradient(1) + 0.5_dp*(gradient(2) + sqrt3*gradient(3)))
            gradient(1) = g
            gradient(2) = 0.5_dp*g
            gradient(3) = sqrt3*gradient(2)
            gradient(4) = 0.0_dp
            gradient(5) = 0.0_dp
         CASE (cell_sym_hexagonal_gamma_120)
            g = 0.5_dp*(gradient(1) - 0.5_dp*(gradient(2) - sqrt3*gradient(3)))
            gradient(1) = g
            gradient(2) = -0.5_dp*g
            gradient(3) = -sqrt3*gradient(2)
            gradient(4) = 0.0_dp
            gradient(5) = 0.0_dp
         CASE (cell_sym_rhombohedral)
            a = (angle(cell%hmat(:, 3), cell%hmat(:, 2)) + &
                 angle(cell%hmat(:, 1), cell%hmat(:, 3)) + &
                 angle(cell%hmat(:, 1), cell%hmat(:, 2)))/3.0_dp
            cosa = COS(a)
            sina = SIN(a)
            cosah = COS(0.5_dp*a)
            sinah = SIN(0.5_dp*a)
            norm = cosa/cosah
            norm_c = SQRT(1.0_dp - norm*norm)
            g = (gradient(1) + gradient(2)*cosa + gradient(3)*sina + &
                 gradient(4)*cosah*norm + gradient(5)*sinah*norm + gradient(6)*norm_c)/3.0_dp
            gradient(1) = g
            gradient(2) = g*cosa
            gradient(3) = g*sina
            gradient(4) = g*cosah*norm
            gradient(5) = g*sinah*norm
            gradient(6) = g*norm_c
         CASE (cell_sym_monoclinic)
            gradient(2) = 0.0_dp
            gradient(5) = 0.0_dp
         CASE (cell_sym_monoclinic_gamma_ab)
            ! Cell symmetry with a = b, alpha = beta = 90 degree and gammma not equal 90 degree
            a_length = SQRT(cell%hmat(1, 1)*cell%hmat(1, 1) + &
                            cell%hmat(2, 1)*cell%hmat(2, 1) + &
                            cell%hmat(3, 1)*cell%hmat(3, 1))
            b_length = SQRT(cell%hmat(1, 2)*cell%hmat(1, 2) + &
                            cell%hmat(2, 2)*cell%hmat(2, 2) + &
                            cell%hmat(3, 2)*cell%hmat(3, 2))
            ab_length = 0.5_dp*(a_length + b_length)
            gamma = angle(cell%hmat(:, 1), cell%hmat(:, 2))
            cosg = COS(gamma)
            sing = SIN(gamma)
            ! Here, g is the average derivative of the cell vector length ab_length, and deriv_gamma is the derivative of the angle gamma
            g = 0.5_dp*(gradient(1) + cosg*gradient(2) + sing*gradient(3))
            deriv_gamma = (gradient(3)*cosg - gradient(2)*sing)/b_length
            gradient(1) = g
            gradient(2) = g*cosg - ab_length*sing*deriv_gamma
            gradient(3) = g*sing + ab_length*cosg*deriv_gamma
            gradient(4) = 0.0_dp
            gradient(5) = 0.0_dp
         CASE (cell_sym_triclinic)
            ! Nothing to do
         END SELECT
      END IF

      SELECT CASE (constraint_id)
      CASE (fix_x)
         gradient(1:2) = 0.0_dp
         gradient(4) = 0.0_dp
      CASE (fix_y)
         gradient(2:3) = 0.0_dp
         gradient(5) = 0.0_dp
      CASE (fix_z)
         gradient(4:6) = 0.0_dp
      CASE (fix_xy)
         gradient(1:5) = 0.0_dp
      CASE (fix_xz)
         gradient(1:2) = 0.0_dp
         gradient(4:6) = 0.0_dp
      CASE (fix_yz)
         gradient(2:6) = 0.0_dp
      CASE (fix_none)
         ! Nothing to do
      END SELECT

   END SUBROUTINE apply_cell_constraints

END MODULE cell_opt_utils
