!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Declares the input for global optimization
!> \author Ole Schuett
! **************************************************************************************************
MODULE glbopt_input
   USE bibliography,                    ONLY: Goedecker2004
   USE cp_output_handling,              ONLY: cp_print_key_section_create,&
                                              low_print_level
   USE input_constants,                 ONLY: glbopt_do_mincrawl,&
                                              glbopt_do_minhop
   USE input_keyword_types,             ONLY: keyword_create,&
                                              keyword_release,&
                                              keyword_type
   USE input_section_types,             ONLY: section_add_keyword,&
                                              section_add_subsection,&
                                              section_create,&
                                              section_release,&
                                              section_type
   USE input_val_types,                 ONLY: integer_t,&
                                              real_t
   USE kinds,                           ONLY: dp
   USE string_utilities,                ONLY: s2a
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'glbopt_input'

   PUBLIC :: glbopt_declare_input

CONTAINS

! **************************************************************************************************
!> \brief Declares the SWARM%GLOBAL_OPT input section
!> \param swarm_section ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE glbopt_declare_input(swarm_section)
      TYPE(section_type), POINTER                        :: swarm_section

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: glbopt_section, printkey

      NULLIFY (glbopt_section, keyword, printkey)

      CALL section_create(glbopt_section, __LOCATION__, name="GLOBAL_OPT", &
                          description="Section to control global geometry optimizations.", &
                          repeats=.FALSE.)

      CALL keyword_create(keyword, __LOCATION__, name="METHOD", &
                          description="Methods to use for optimization.", &
                          default_i_val=glbopt_do_minhop, &
                          enum_c_vals=s2a("MINIMA_HOPPING", "MINIMA_CRAWLING"), &
                          enum_desc=s2a("Runs Minima-Hopping algorithm.", &
                                        "Runs Minima-Crawling algorithm."), &
                          enum_i_vals=(/glbopt_do_minhop, glbopt_do_mincrawl/))
      CALL section_add_keyword(glbopt_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="E_TARGET", &
                          description="Target Energy, the optimization will quit once a lower potential energy is reached.", &
                          default_r_val=-1*HUGE(1.0_dp), type_of_var=real_t, unit_str="hartree")
      CALL section_add_keyword(glbopt_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="MD_BUMPS_MAX", &
                          description="Number of bumps in potential energy after which MD runs ends.", &
                          type_of_var=integer_t, default_i_val=3)
      CALL section_add_keyword(glbopt_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="BUMP_STEPS_UPWARDS", &
                          description="Number of MD steps with potential energy increases required for a bump.", &
                          type_of_var=integer_t, default_i_val=2)
      CALL section_add_keyword(glbopt_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="BUMP_STEPS_DOWNWARDS", &
                          description="Number of MD steps with potential energy decreases required for a bump.", &
                          type_of_var=integer_t, default_i_val=2)
      CALL section_add_keyword(glbopt_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="FRAGMENTATION_THRESHOLD", &
                          description="Threshold for atom distance used for detecting fragmentation of clusters.", &
                          default_r_val=2.0_dp, unit_str="angstrom", type_of_var=real_t)
      CALL section_add_keyword(glbopt_section, keyword)
      CALL keyword_release(keyword)

      !CALL keyword_create(keyword, __LOCATION__, name="MD_ADAPTIVE_TIMESTEP",&
      !     description="Make MD timesteps longer for lower temperatures.", &
      !     default_r_val=0.0_dp, type_of_var=real_t)
      !CALL section_add_keyword(glbopt_section, keyword)
      !CALL keyword_release(keyword)

      CALL cp_print_key_section_create( &
         printkey, __LOCATION__, "PROGRESS_TRAJECTORY", &
         description="Printkey to control the writting of the progress trajectory. "// &
         "This trajectory contains the minima, which are lower in energy than the by then lowerest.", &
         print_level=low_print_level, common_iter_levels=1, &
         filename="", unit_str="angstrom")
      CALL section_add_subsection(glbopt_section, printkey)
      CALL section_release(printkey)

      CALL history_declare_input(glbopt_section)
      CALL minhop_declare_input(glbopt_section)
      CALL mincrawl_declare_input(glbopt_section)

      CALL section_add_subsection(swarm_section, glbopt_section)
      CALL section_release(glbopt_section)
   END SUBROUTINE glbopt_declare_input

! **************************************************************************************************
!> \brief Declares the SWARM%GLOBAL_OPT%HISTORY input section
!> \param glbopt_section ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE history_declare_input(glbopt_section)
      TYPE(section_type), POINTER                        :: glbopt_section

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: history_section

      NULLIFY (history_section, keyword)

      CALL section_create(history_section, __LOCATION__, name="HISTORY", &
                          description="Section controlling the history of visited minima and "// &
                          "how minima are recognized at a later point.", &
                          repeats=.FALSE.)

      CALL keyword_create(keyword, __LOCATION__, name="ENERGY_PRECISION", &
                          description="If the difference of two energies is below this threshold "// &
                          "they are considert equal.", &
                          default_r_val=1.0e-5_dp, type_of_var=real_t)
      CALL section_add_keyword(history_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="FINGERPRINT_PRECISION", &
                          description="If the euclidean distance of two fingerprints is below "// &
                          "this threshold, they are considert equal.", &
                          default_r_val=1.0e-2_dp, type_of_var=real_t)
      CALL section_add_keyword(history_section, keyword)
      CALL keyword_release(keyword)

      CALL section_add_subsection(glbopt_section, history_section)
      CALL section_release(history_section)
   END SUBROUTINE history_declare_input

! **************************************************************************************************
!> \brief Declares the SWARM%GLOBAL_OPT%MINIMA_HOPPING input section
!> \param glbopt_section ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE minhop_declare_input(glbopt_section)
      TYPE(section_type), POINTER                        :: glbopt_section

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: minhop_section

      NULLIFY (minhop_section, keyword)

      CALL section_create(minhop_section, __LOCATION__, name="MINIMA_HOPPING", &
                          description="Section controlling the Minima Hopping method.", &
                          citations=(/Goedecker2004/), &
                          repeats=.FALSE.)

      CALL keyword_create(keyword, __LOCATION__, name="BETA_1", &
                          description="Factor used to increase temperature when escape failed, "// &
                          "should be larger than 1.", &
                          default_r_val=1.1_dp, type_of_var=real_t)
      CALL section_add_keyword(minhop_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="BETA_2", &
                          description="Factor used to increase temperature when escape found "// &
                          "known minima, should be larger than 1.", &
                          default_r_val=1.1_dp, type_of_var=real_t)
      CALL section_add_keyword(minhop_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="BETA_3", &
                          description="Factor used to decrease temperature when escape succeeded, "// &
                          "should be smaller than 1.", &
                          default_r_val=1.0/1.1_dp, type_of_var=real_t)
      CALL section_add_keyword(minhop_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="ALPHA_1", &
                          description="Factor used to decrease acceptance energy, when minima was accepted, "// &
                          "should be smaller than 1.", &
                          default_r_val=0.98_dp, type_of_var=real_t)
      CALL section_add_keyword(minhop_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="ALPHA_2", &
                          description="Factor used to increase acceptance energy, when minima was rejected, "// &
                          "should be larger than 1.", &
                          default_r_val=1.0/0.98_dp, type_of_var=real_t)
      CALL section_add_keyword(minhop_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="E_ACCEPT_INIT", &
                          description="Initial value of acceptance energy", &
                          default_r_val=0.005_dp, type_of_var=real_t, unit_str="hartree")
      CALL section_add_keyword(minhop_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="TEMPERATURE_INIT", &
                          description="Initially temperature in Kelvin", &
                          default_r_val=100.0_dp, type_of_var=real_t)
      CALL section_add_keyword(minhop_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="SHARE_HISTORY", &
                          description="If set all worker will use a single share history of visited minima.", &
                          default_l_val=.FALSE., lone_keyword_l_val=.TRUE.)
      CALL section_add_keyword(minhop_section, keyword)
      CALL keyword_release(keyword)

      CALL section_add_subsection(glbopt_section, minhop_section)
      CALL section_release(minhop_section)
   END SUBROUTINE minhop_declare_input

! **************************************************************************************************
!> \brief Declares the SWARM%GLOBAL_OPT%MINIMA_CRAWLING input section
!> \param glbopt_section ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE mincrawl_declare_input(glbopt_section)
      TYPE(section_type), POINTER                        :: glbopt_section

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: mincrawl_section, printkey

      NULLIFY (mincrawl_section, keyword, printkey)

      CALL section_create(mincrawl_section, __LOCATION__, name="MINIMA_CRAWLING", &
                          description="Section controls Minima Crawling run.", &
                          repeats=.FALSE.)

      CALL keyword_create(keyword, __LOCATION__, name="TEMPSTEP_BASE", &
                          description="Base used to calculate temperature steps base**n", &
                          default_r_val=1.1_dp, type_of_var=real_t)
      CALL section_add_keyword(mincrawl_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="TEMPSTEP_MAX", &
                          description="Maximum number of temperature steps.", &
                          default_i_val=100, type_of_var=integer_t)
      CALL section_add_keyword(mincrawl_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="TEMPDIST_UPDATE_WIDTH", &
                          description="Width of gaussian used to update temperature distribution.", &
                          default_r_val=2.0_dp, type_of_var=real_t)
      CALL section_add_keyword(mincrawl_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="TEMPDIST_UPDATE_HEIGHT", &
                          description="Height of gaussian used to update temperature distribution.", &
                          default_r_val=0.1_dp, type_of_var=real_t)
      CALL section_add_keyword(mincrawl_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="TEMPERATURE_INIT", &
                          description="Initial temperature in Kelvin", &
                          default_r_val=100.0_dp, type_of_var=real_t)
      CALL section_add_keyword(mincrawl_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="TEMPDIST_INIT_WIDTH", &
                          description="Initial width of temperature distribution.", &
                          default_r_val=5.0_dp, type_of_var=real_t)
      CALL section_add_keyword(mincrawl_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="WORKER_PER_MINIMA", &
                          description="Maximum number of active workers per Minima.", &
                          default_i_val=3, type_of_var=integer_t)
      CALL section_add_keyword(mincrawl_section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="ESCAPE_HISTORY_LENGTH", &
                          description="Number of escapes averaged for scoring of minima.", &
                          default_i_val=10, type_of_var=integer_t)
      CALL section_add_keyword(mincrawl_section, keyword)
      CALL keyword_release(keyword)

      CALL cp_print_key_section_create(printkey, __LOCATION__, "MINIMA_TRAJECTORY", &
                                       description="Printkey to control the writting of the minima trajectory. "// &
                                       "This trajectory contains all encountered local minima.", &
                                       print_level=low_print_level, common_iter_levels=1, &
                                       filename="", unit_str="angstrom")
      CALL section_add_subsection(mincrawl_section, printkey)
      CALL section_release(printkey)

      CALL section_add_subsection(glbopt_section, mincrawl_section)
      CALL section_release(mincrawl_section)
   END SUBROUTINE mincrawl_declare_input

END MODULE glbopt_input

