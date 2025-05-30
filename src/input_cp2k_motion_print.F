!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \par History
!>      - taken out of input_cp2k_motion
!> \author Ole Schuett
! **************************************************************************************************
MODULE input_cp2k_motion_print
   USE cp_output_handling,              ONLY: add_last_numeric,&
                                              cp_print_key_section_create,&
                                              high_print_level,&
                                              low_print_level,&
                                              silent_print_level
   USE input_constants,                 ONLY: dump_atomic,&
                                              dump_dcd,&
                                              dump_dcd_aligned_cell,&
                                              dump_pdb,&
                                              dump_xmol
   USE input_cp2k_subsys,               ONLY: create_structure_data_section
   USE input_keyword_types,             ONLY: keyword_create,&
                                              keyword_release,&
                                              keyword_type
   USE input_section_types,             ONLY: section_add_keyword,&
                                              section_add_subsection,&
                                              section_create,&
                                              section_release,&
                                              section_type
   USE string_utilities,                ONLY: s2a
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'input_cp2k_motion_print'

   PUBLIC :: create_motion_print_section, add_format_keyword

CONTAINS

! **************************************************************************************************
!> \brief creates the motion%print section
!> \param section the section to be created
!> \author teo
! **************************************************************************************************
   SUBROUTINE create_motion_print_section(section)
      TYPE(section_type), POINTER                        :: section

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: print_key

      NULLIFY (keyword, section, print_key)

      CALL section_create(section, __LOCATION__, name="print", &
                          description="Controls the printing properties during an MD/Optimization run", &
                          n_keywords=1, n_subsections=1, repeats=.TRUE.)

      CALL keyword_create(keyword, __LOCATION__, name="MEMORY_INFO", &
                          variants=(/"MEMORY"/), &
                          description="Whether overall memory usage should be sampled and printed "// &
                          "at each MD/Optimization step.", &
                          usage="MEMORY_INFO LOGICAL", &
                          default_l_val=.TRUE., lone_keyword_l_val=.TRUE.)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "TRAJECTORY", &
                                       description="Controls the output of the trajectory", &
                                       print_level=low_print_level, common_iter_levels=1, &
                                       filename="", unit_str="angstrom")
      CALL add_format_keyword(keyword, print_key, pos=.TRUE., &
                              description="Specifies the format of the output file for the trajectory.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create( &
         print_key, __LOCATION__, "SHELL_TRAJECTORY", &
         description="Controls the output of the trajectory of shells when the shell-model is used ", &
         print_level=high_print_level, common_iter_levels=1, &
         filename="", unit_str="angstrom")
      CALL add_format_keyword(keyword, print_key, pos=.TRUE., &
                              description="Specifies the format of the output file for the trajectory of shells.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "CORE_TRAJECTORY", &
                                       description="Controls the output of the trajectory of cores when the shell-model is used ", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="", unit_str="angstrom")
      CALL add_format_keyword(keyword, print_key, pos=.TRUE., &
                              description="Specifies the format of the output file for the trajectory of cores.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "CELL", &
                                       description="Controls the output of the simulation cell. "// &
                                       "For later analysis of the trajectory it is recommendable that the "// &
                                       "frequency of printing is the same as the one used for the trajectory file.", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "VELOCITIES", &
                                       description="Controls the output of the atomic velocities. "// &
                                       "The default unit for the atomic velocities $v$ is bohr/au_time. "// &
                                       "The kinetic energy $K_i$ in [hartree] of an atom $i$ with mass $m_i$ in "// &
                                       "atomic mass units [amu] (or just [u]) and velocity $v_i$ can be obtained by "// &
                                       "$K_i = \frac{1}{2}f_u\,m_i\,v_i^2$ with $f_u = 1822.888$ as conversion factor "// &
                                       "from [u] to [a.u.]. $f_u$ is printed in full precision in the top of the CP2K "// &
                                       "output for print level MEDIUM or higher.", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="", unit_str="bohr*au_t^-1")
      CALL add_format_keyword(keyword, print_key, pos=.FALSE., &
                              description="Specifies the format of the output file for the velocities.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "SHELL_VELOCITIES", &
                                       description="Controls the output of the velocities of shells when the shell model is used", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="", unit_str="bohr*au_t^-1")
      CALL add_format_keyword(keyword, print_key, pos=.FALSE., &
                              description="Specifies the format of the output file for the velocities of shells.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "CORE_VELOCITIES", &
                                       description="controls the output of the velocities of cores when the shell model is used", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="", unit_str="bohr*au_t^-1")
      CALL add_format_keyword(keyword, print_key, pos=.FALSE., &
                              description="Specifies the format of the output file for the velocities of cores.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL create_structure_data_section(print_key)
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create( &
         print_key, __LOCATION__, "FORCE_MIXING_LABELS", &
         description="Controls the output of the force mixing (FORCE_EVAL&QMMM&FORCE_MIXING) labels", &
         print_level=high_print_level, common_iter_levels=1, &
         filename="")
      CALL add_format_keyword(keyword, print_key, pos=.FALSE., &
                              description="Specifies the format of the output file for the force mixing labels.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "FORCES", &
                                       description="Controls the output of the forces", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="", unit_str="hartree*bohr^-1")
      CALL add_format_keyword(keyword, print_key, pos=.FALSE., &
                              description="Specifies the format of the output file for the forces.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "SHELL_FORCES", &
                                       description="Controls the output of the forces on shells when shell-model is used", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="", unit_str="hartree*bohr^-1")
      CALL add_format_keyword(keyword, print_key, pos=.FALSE., &
                              description="Specifies the format of the output file for the forces on shells.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "CORE_FORCES", &
                                       description="Controls the output of the forces on cores when shell-model is used", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="", unit_str="hartree*bohr^-1")
      CALL add_format_keyword(keyword, print_key, pos=.FALSE., &
                              description="Specifies the format of the output file for the forces on cores.")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "MIXED_ENERGIES", &
                                       description="Controls the output of the energies of the two "// &
                                       "regular FORCE_EVALS in the MIXED method "// &
                                       "printed is step,time,Etot,E_F1,E_F2,CONS_QNT", &
                                       print_level=low_print_level, common_iter_levels=1, &
                                       filename="")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "STRESS", &
                                       description="Controls the output of the stress tensor", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "POLAR_MATRIX", &
                                       description="Controls the output of the polarisability tensor during an MD run", &
                                       print_level=low_print_level, common_iter_levels=1, &
                                       filename="")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "RESTART", &
                                       description="Controls the dumping of the restart file during runs. "// &
                                       "By default keeps a short history of three restarts. See also RESTART_HISTORY", &
                                       each_iter_names=s2a("MD"), each_iter_values=(/20/), &
                                       print_level=silent_print_level, common_iter_levels=1, &
                                       add_last=add_last_numeric, filename="")

      CALL keyword_create(keyword, __LOCATION__, name="BACKUP_COPIES", &
                          description="Specifies the maximum number of backup copies.", &
                          usage="BACKUP_COPIES {int}", &
                          default_i_val=1)
      CALL section_add_keyword(print_key, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="SPLIT_RESTART_FILE", &
                          description="If specified selected input sections, which are growing with the "// &
                          "number of atoms in the system, are written to another restart file "// &
                          "in binary format instead of the default restart file in human "// &
                          "readable ASCII format. This split of the restart file may "// &
                          "provide significant memory savings and an accelerated I/O for "// &
                          "systems with a very large number of atoms", &
                          usage="SPLIT_RESTART_FILE yes", &
                          default_l_val=.FALSE., &
                          lone_keyword_l_val=.TRUE.)
      CALL section_add_keyword(print_key, keyword)
      CALL keyword_release(keyword)

      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "RESTART_HISTORY", &
                                       description="Dumps unique restart files during the run keeping all of them. "// &
                                       "Most useful if recovery is needed at a later point.", &
                                       print_level=low_print_level, common_iter_levels=0, &
                                       each_iter_names=s2a("MD", "GEO_OPT", "ROT_OPT"), each_iter_values=(/500, 500, 500/), &
                                       filename="")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "TRANSLATION_VECTOR", &
                                       description="Dumps the translation vector applied along an MD (if any). Useful"// &
                                       " for postprocessing of QMMM trajectories in which the QM fragment is continuously"// &
                                       " centered in the QM box", &
                                       print_level=high_print_level, common_iter_levels=1, &
                                       filename="")
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

   END SUBROUTINE create_motion_print_section

! **************************************************************************************************
!> \brief creates the FORMAT keyword
!> \param keyword ...
!> \param section will contain the pint section
!> \param pos ...
!> \param description ...
!> \author Teodoro Laino 10.2008 [tlaino]
! **************************************************************************************************
   SUBROUTINE add_format_keyword(keyword, section, pos, description)
      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: section
      LOGICAL, INTENT(IN)                                :: pos
      CHARACTER(LEN=*), INTENT(IN)                       :: description

      CPASSERT(ASSOCIATED(section))
      CPASSERT(.NOT. ASSOCIATED(keyword))

      IF (pos) THEN

         CALL keyword_create( &
            keyword, __LOCATION__, name="FORMAT", &
            description=description, usage="FORMAT (ATOMIC|DCD|PDB|XMOL|XYZ)", &
            default_i_val=dump_xmol, &
            enum_c_vals=s2a("ATOMIC", "DCD", "DCD_ALIGNED_CELL", "PDB", "XMOL", "XYZ"), &
            enum_i_vals=(/dump_atomic, dump_dcd, dump_dcd_aligned_cell, dump_pdb, dump_xmol, dump_xmol/), &
            enum_desc=s2a("Write only the coordinates X,Y,Z without element symbols to a formatted file", &
                          "Write the coordinates (no element labels) and the cell information to a binary file", &
                          "Like DCD, but the dumped coordinates refer to an aligned cell following the common convention: "// &
                          "the cell vector **a** is aligned with the *x* axis and the cell vector **b** lies in "// &
                          "the *xy* plane. This allows the reconstruction of scaled coordinates from the DCD data only.", &
                          "Write the atomic information in PDB format to a formatted file", &
                          "Mostly known as XYZ format, provides in a formatted file: element_symbol X Y Z", &
                          "Alias name for XMOL"))
         CALL section_add_keyword(section, keyword)
         CALL keyword_release(keyword)

         CALL keyword_create(keyword, __LOCATION__, name="CHARGE_OCCUP", &
                             variants=(/"CHARGE_O"/), &
                             description="Write the MM charges to the OCCUP field of the PDB file", &
                             usage="CHARGE_OCCUP logical", &
                             default_l_val=.FALSE., lone_keyword_l_val=.TRUE.)
         CALL section_add_keyword(section, keyword)
         CALL keyword_release(keyword)

         CALL keyword_create(keyword, __LOCATION__, name="CHARGE_BETA", &
                             variants=(/"CHARGE_B"/), &
                             description="Write the MM charges to the BETA field of the PDB file", &
                             usage="CHARGE_BETA logical", &
                             default_l_val=.FALSE., lone_keyword_l_val=.TRUE.)
         CALL section_add_keyword(section, keyword)
         CALL keyword_release(keyword)

         CALL keyword_create(keyword, __LOCATION__, name="CHARGE_EXTENDED", &
                             description="Write the MM charges to the very last field of the PDB file (starting from column 81)", &
                             usage="CHARGE_EXTENDED logical", &
                             default_l_val=.FALSE., lone_keyword_l_val=.TRUE.)
         CALL section_add_keyword(section, keyword)
         CALL keyword_release(keyword)

      ELSE

         CALL keyword_create(keyword, __LOCATION__, name="FORMAT", &
                             description=description, usage="FORMAT (ATOMIC|DCD|XMOL|XYZ)", &
                             default_i_val=dump_xmol, &
                             enum_c_vals=s2a("ATOMIC", "DCD", "XMOL", "XYZ"), &
                             enum_i_vals=(/dump_atomic, dump_dcd, dump_xmol, dump_xmol/), &
                             enum_desc=s2a("Write only the coordinates X,Y,Z without element symbols to a formatted file", &
                                           "Write the coordinates (no element labels) and the cell information to a binary file", &
                                           "Mostly known as XYZ format, provides in a formatted file: element_symbol X Y Z", &
                                           "Alias name for XMOL"))
         CALL section_add_keyword(section, keyword)
         CALL keyword_release(keyword)

      END IF

      CALL keyword_create(keyword, __LOCATION__, name="PRINT_ATOM_KIND", &
                          description="Write the atom kind given in the subsys section instead of the element symbol. "// &
                          "Only valid for the XMOL format.", &
                          usage="PRINT_ATOM_KIND logical", &
                          default_l_val=.FALSE., lone_keyword_l_val=.TRUE.)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

   END SUBROUTINE add_format_keyword

END MODULE input_cp2k_motion_print
