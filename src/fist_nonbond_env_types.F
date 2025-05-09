!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \par History
!>      none
!> \author HAF
! **************************************************************************************************
MODULE fist_nonbond_env_types
   USE ace_wrapper,                     ONLY: ace_model_release,&
                                              ace_model_type
   USE atomic_kind_types,               ONLY: atomic_kind_type
   USE cell_types,                      ONLY: cell_release,&
                                              cell_type
   USE deepmd_wrapper,                  ONLY: deepmd_model_release,&
                                              deepmd_model_type
   USE fist_neighbor_list_types,        ONLY: fist_neighbor_deallocate,&
                                              fist_neighbor_type
   USE kinds,                           ONLY: default_string_length,&
                                              dp
   USE pair_potential_types,            ONLY: ace_type,&
                                              gal21_type,&
                                              gal_type,&
                                              nequip_type,&
                                              pair_potential_pp_release,&
                                              pair_potential_pp_type,&
                                              siepmann_type,&
                                              tersoff_type
   USE torch_api,                       ONLY: torch_model_release,&
                                              torch_model_type
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fist_nonbond_env_types'
   PUBLIC :: fist_nonbond_env_type, fist_nonbond_env_set, &
             fist_nonbond_env_get, fist_nonbond_env_create, &
             fist_nonbond_env_release, pos_type, eam_type, &
             quip_data_type, nequip_data_type, allegro_data_type, &
             deepmd_data_type, ace_data_type

! **************************************************************************************************
   TYPE pos_type
      REAL(KIND=dp) :: r(3) = 0.0_dp
   END TYPE

   TYPE eam_type
      REAL(KIND=dp) :: f_embed = 0.0_dp
      REAL(KIND=dp) :: rho = 0.0_dp
   END TYPE

   TYPE quip_data_type
      INTEGER, POINTER        :: use_indices(:) => NULL()
      REAL(KIND=dp), POINTER  :: force(:, :) => NULL()
      REAL(KIND=dp)           :: virial(3, 3) = 0.0_dp
   END TYPE

   TYPE nequip_data_type
      INTEGER, POINTER        :: use_indices(:) => NULL()
      REAL(KIND=dp), POINTER  :: force(:, :) => NULL()
      REAL(KIND=dp)           :: virial(3, 3) = 0.0_dp
      TYPE(torch_model_type)  :: model
   END TYPE

   TYPE allegro_data_type
      INTEGER, POINTER        :: use_indices(:) => NULL()
      REAL(KIND=dp), POINTER  :: force(:, :) => NULL()
      REAL(KIND=dp)           :: virial(3, 3) = 0.0_dp
      TYPE(torch_model_type)  :: model
   END TYPE

   TYPE deepmd_data_type
      INTEGER, POINTER        :: use_indices(:) => NULL()
      REAL(KIND=dp), POINTER  :: force(:, :) => NULL()
      REAL(KIND=dp)           :: virial(3, 3) = 0.0_dp
      TYPE(deepmd_model_type) :: model
   END TYPE

   TYPE ace_data_type
      INTEGER, ALLOCATABLE    :: use_indices(:)
      INTEGER, ALLOCATABLE    :: inverse_index_map(:)
      INTEGER                 :: natom = 0
      INTEGER                 :: nghost = 0
      INTEGER                 :: refupdate = 0
      INTEGER                 :: nei = 0
      INTEGER, ALLOCATABLE    :: uctype(:)
      INTEGER, ALLOCATABLE    :: attype(:)
      INTEGER, ALLOCATABLE    :: origin(:)
      INTEGER, ALLOCATABLE    :: shift(:, :)
      INTEGER, ALLOCATABLE    :: neiat(:)
      INTEGER, ALLOCATABLE    :: nlist(:)
      REAL(KIND=dp), ALLOCATABLE  :: force(:, :)
      REAL(KIND=dp), ALLOCATABLE  :: atpos(:, :)
      REAL(KIND=dp)           :: virial(3, 3) = 0.0_dp
      TYPE(ace_model_type)    :: model
   END TYPE

! **************************************************************************************************
   TYPE fist_nonbond_env_type
      INTEGER                                    :: natom_types = -1
      INTEGER                                    :: counter = -1
      INTEGER                                    :: last_update = -1
      INTEGER                                    :: num_update = -1
      LOGICAL                                    :: do_nonbonded = .FALSE.
      LOGICAL                                    :: do_electrostatics = .FALSE.
      LOGICAL                                    :: shift_cutoff = .FALSE.
      CHARACTER(len=default_string_length)       :: unit_type = ""
      REAL(KIND=dp)                              :: lup = 0.0_dp
      REAL(KIND=dp)                              :: aup = 0.0_dp
      REAL(KIND=dp)                              :: ei_scale14 = 0.0_dp
      REAL(KIND=dp)                              :: vdw_scale14 = 0.0_dp
      REAL(KIND=dp)                              :: long_range_correction = 0.0_dp
      REAL(KIND=dp), DIMENSION(:, :), POINTER    :: rlist_cut => NULL()
      REAL(KIND=dp), DIMENSION(:, :), POINTER    :: rlist_lowsq => NULL()
      REAL(KIND=dp), DIMENSION(:, :), POINTER    :: ij_kind_full_fac => NULL()
      REAL(KIND=dp), DIMENSION(:), POINTER       :: charges => NULL()
      TYPE(fist_neighbor_type), POINTER          :: nonbonded => NULL()
      TYPE(pair_potential_pp_type), POINTER      :: potparm14 => NULL()
      TYPE(pair_potential_pp_type), POINTER      :: potparm => NULL()
      TYPE(cell_type), POINTER                   :: cell_last_update => NULL()
      TYPE(pos_type), DIMENSION(:), POINTER      :: r_last_update => NULL()
      TYPE(pos_type), DIMENSION(:), POINTER      :: r_last_update_pbc => NULL()
      TYPE(pos_type), DIMENSION(:), POINTER      :: rshell_last_update_pbc => NULL()
      TYPE(pos_type), DIMENSION(:), POINTER      :: rcore_last_update_pbc => NULL()
      TYPE(eam_type), DIMENSION(:), POINTER      :: eam_data => NULL()
      TYPE(quip_data_type), POINTER              :: quip_data => NULL()
      TYPE(deepmd_data_type), POINTER            :: deepmd_data => NULL()
      TYPE(ace_data_type), POINTER               :: ace_data => NULL()
      TYPE(nequip_data_type), POINTER            :: nequip_data => NULL()
      TYPE(allegro_data_type), POINTER           :: allegro_data => NULL()
   END TYPE fist_nonbond_env_type

CONTAINS

! **************************************************************************************************
!> \brief sets a fist_nonbond_env
!> \param fist_nonbond_env the object to create
!> \param potparm14 ...
!> \param potparm ...
!> \param nonbonded ...
!> \param rlist_cut ...
!> \param rlist_lowsq ...
!> \param aup ...
!> \param lup ...
!> \param ei_scale14 ...
!> \param vdw_scale14 ...
!> \param shift_cutoff ...
!> \param do_electrostatics ...
!> \param r_last_update ...
!> \param r_last_update_pbc ...
!> \param rshell_last_update_pbc ...
!> \param rcore_last_update_pbc ...
!> \param cell_last_update ...
!> \param num_update ...
!> \param last_update ...
!> \param counter ...
!> \param natom_types ...
!> \param long_range_correction ...
!> \param ij_kind_full_fac ...
!> \param eam_data ...
!> \param quip_data ...
!> \param nequip_data ...
!> \param allegro_data ...
!> \param deepmd_data ...
!> \param ace_data ...
!> \param charges ...
!> \par History
!>      12.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE fist_nonbond_env_get(fist_nonbond_env, potparm14, potparm, &
                                   nonbonded, rlist_cut, rlist_lowsq, aup, lup, ei_scale14, vdw_scale14, &
                                   shift_cutoff, do_electrostatics, r_last_update, r_last_update_pbc, rshell_last_update_pbc, &
                                   rcore_last_update_pbc, cell_last_update, num_update, last_update, &
                                   counter, natom_types, long_range_correction, ij_kind_full_fac, eam_data, &
                                   quip_data, nequip_data, allegro_data, deepmd_data, ace_data, charges)

      TYPE(fist_nonbond_env_type), INTENT(IN)            :: fist_nonbond_env
      TYPE(pair_potential_pp_type), OPTIONAL, POINTER    :: potparm14, potparm
      TYPE(fist_neighbor_type), OPTIONAL, POINTER        :: nonbonded
      REAL(KIND=dp), DIMENSION(:, :), OPTIONAL, POINTER  :: rlist_cut, rlist_lowsq
      REAL(KIND=dp), OPTIONAL                            :: aup, lup, ei_scale14, vdw_scale14
      LOGICAL, INTENT(OUT), OPTIONAL                     :: shift_cutoff, do_electrostatics
      TYPE(pos_type), DIMENSION(:), OPTIONAL, POINTER    :: r_last_update, r_last_update_pbc, &
                                                            rshell_last_update_pbc, &
                                                            rcore_last_update_pbc
      TYPE(cell_type), OPTIONAL, POINTER                 :: cell_last_update
      INTEGER, OPTIONAL                                  :: num_update, last_update, counter, &
                                                            natom_types
      REAL(KIND=dp), OPTIONAL                            :: long_range_correction
      REAL(KIND=dp), DIMENSION(:, :), OPTIONAL, POINTER  :: ij_kind_full_fac
      TYPE(eam_type), DIMENSION(:), OPTIONAL, POINTER    :: eam_data
      TYPE(quip_data_type), OPTIONAL, POINTER            :: quip_data
      TYPE(nequip_data_type), OPTIONAL, POINTER          :: nequip_data
      TYPE(allegro_data_type), OPTIONAL, POINTER         :: allegro_data
      TYPE(deepmd_data_type), OPTIONAL, POINTER          :: deepmd_data
      TYPE(ace_data_type), OPTIONAL, POINTER             :: ace_data
      REAL(KIND=dp), DIMENSION(:), OPTIONAL, POINTER     :: charges

      IF (PRESENT(charges)) charges => fist_nonbond_env%charges
      IF (PRESENT(potparm14)) potparm14 => fist_nonbond_env%potparm14
      IF (PRESENT(eam_data)) eam_data => fist_nonbond_env%eam_data
      IF (PRESENT(quip_data)) quip_data => fist_nonbond_env%quip_data
      IF (PRESENT(nequip_data)) nequip_data => fist_nonbond_env%nequip_data
      IF (PRESENT(allegro_data)) allegro_data => fist_nonbond_env%allegro_data
      IF (PRESENT(deepmd_data)) deepmd_data => fist_nonbond_env%deepmd_data
      IF (PRESENT(ace_data)) ace_data => fist_nonbond_env%ace_data
      IF (PRESENT(potparm)) potparm => fist_nonbond_env%potparm
      IF (PRESENT(rlist_cut)) rlist_cut => fist_nonbond_env%rlist_cut
      IF (PRESENT(rlist_lowsq)) rlist_lowsq => fist_nonbond_env%rlist_lowsq
      IF (PRESENT(ij_kind_full_fac)) ij_kind_full_fac => fist_nonbond_env%ij_kind_full_fac
      IF (PRESENT(nonbonded)) nonbonded => fist_nonbond_env%nonbonded
      IF (PRESENT(r_last_update)) &
         r_last_update => fist_nonbond_env%r_last_update
      IF (PRESENT(r_last_update_pbc)) &
         r_last_update_pbc => fist_nonbond_env%r_last_update_pbc
      IF (PRESENT(rshell_last_update_pbc)) &
         rshell_last_update_pbc => fist_nonbond_env%rshell_last_update_pbc
      IF (PRESENT(rcore_last_update_pbc)) &
         rcore_last_update_pbc => fist_nonbond_env%rcore_last_update_pbc
      IF (PRESENT(cell_last_update)) &
         cell_last_update => fist_nonbond_env%cell_last_update
      IF (PRESENT(lup)) lup = fist_nonbond_env%lup
      IF (PRESENT(aup)) aup = fist_nonbond_env%aup
      IF (PRESENT(ei_scale14)) ei_scale14 = fist_nonbond_env%ei_scale14
      IF (PRESENT(vdw_scale14)) vdw_scale14 = fist_nonbond_env%vdw_scale14
      IF (PRESENT(shift_cutoff)) &
         shift_cutoff = fist_nonbond_env%shift_cutoff
      IF (PRESENT(do_electrostatics)) do_electrostatics = fist_nonbond_env%do_electrostatics
      IF (PRESENT(natom_types)) natom_types = fist_nonbond_env%natom_types
      IF (PRESENT(counter)) counter = fist_nonbond_env%counter
      IF (PRESENT(last_update)) last_update = fist_nonbond_env%last_update
      IF (PRESENT(num_update)) num_update = fist_nonbond_env%num_update
      IF (PRESENT(long_range_correction)) &
         long_range_correction = fist_nonbond_env%long_range_correction
   END SUBROUTINE fist_nonbond_env_get

! **************************************************************************************************
!> \brief sets a fist_nonbond_env
!> \param fist_nonbond_env the object to create
!> \param potparm14 ...
!> \param potparm ...
!> \param rlist_cut ...
!> \param rlist_lowsq ...
!> \param nonbonded ...
!> \param aup ...
!> \param lup ...
!> \param ei_scale14 ...
!> \param vdw_scale14 ...
!> \param shift_cutoff ...
!> \param do_electrostatics ...
!> \param r_last_update ...
!> \param r_last_update_pbc ...
!> \param rshell_last_update_pbc ...
!> \param rcore_last_update_pbc ...
!> \param cell_last_update ...
!> \param num_update ...
!> \param last_update ...
!> \param counter ...
!> \param natom_types ...
!> \param long_range_correction ...
!> \param eam_data ...
!> \param quip_data ...
!> \param nequip_data ...
!> \param allegro_data ...
!> \param deepmd_data ...
!> \param ace_data ...
!> \param charges ...
!> \par History
!>      12.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE fist_nonbond_env_set(fist_nonbond_env, potparm14, potparm, &
                                   rlist_cut, rlist_lowsq, nonbonded, aup, lup, ei_scale14, vdw_scale14, &
                                   shift_cutoff, do_electrostatics, r_last_update, r_last_update_pbc, rshell_last_update_pbc, &
                                   rcore_last_update_pbc, cell_last_update, num_update, last_update, &
                                   counter, natom_types, long_range_correction, eam_data, quip_data, &
                                   nequip_data, allegro_data, deepmd_data, ace_data, charges)

      TYPE(fist_nonbond_env_type), INTENT(INOUT)         :: fist_nonbond_env
      TYPE(pair_potential_pp_type), OPTIONAL, POINTER    :: potparm14, potparm
      REAL(KIND=dp), DIMENSION(:, :), OPTIONAL, POINTER  :: rlist_cut, rlist_lowsq
      TYPE(fist_neighbor_type), OPTIONAL, POINTER        :: nonbonded
      REAL(KIND=dp), OPTIONAL                            :: aup, lup, ei_scale14, vdw_scale14
      LOGICAL, INTENT(IN), OPTIONAL                      :: shift_cutoff, do_electrostatics
      TYPE(pos_type), DIMENSION(:), OPTIONAL, POINTER    :: r_last_update, r_last_update_pbc, &
                                                            rshell_last_update_pbc, &
                                                            rcore_last_update_pbc
      TYPE(cell_type), OPTIONAL, POINTER                 :: cell_last_update
      INTEGER, OPTIONAL                                  :: num_update, last_update, counter, &
                                                            natom_types
      REAL(KIND=dp), OPTIONAL                            :: long_range_correction
      TYPE(eam_type), DIMENSION(:), OPTIONAL, POINTER    :: eam_data
      TYPE(quip_data_type), OPTIONAL, POINTER            :: quip_data
      TYPE(nequip_data_type), OPTIONAL, POINTER          :: nequip_data
      TYPE(allegro_data_type), OPTIONAL, POINTER         :: allegro_data
      TYPE(deepmd_data_type), OPTIONAL, POINTER          :: deepmd_data
      TYPE(ace_data_type), OPTIONAL, POINTER             :: ace_data
      REAL(KIND=dp), DIMENSION(:), OPTIONAL, POINTER     :: charges

      IF (PRESENT(potparm14)) fist_nonbond_env%potparm14 => potparm14
      IF (PRESENT(eam_data)) fist_nonbond_env%eam_data => eam_data
      IF (PRESENT(quip_data)) fist_nonbond_env%quip_data => quip_data
      IF (PRESENT(nequip_data)) fist_nonbond_env%nequip_data => nequip_data
      IF (PRESENT(allegro_data)) fist_nonbond_env%allegro_data => allegro_data
      IF (PRESENT(deepmd_data)) fist_nonbond_env%deepmd_data => deepmd_data
      IF (PRESENT(ace_data)) fist_nonbond_env%ace_data => ace_data
      IF (PRESENT(potparm)) fist_nonbond_env%potparm => potparm
      IF (PRESENT(rlist_cut)) fist_nonbond_env%rlist_cut => rlist_cut
      IF (PRESENT(charges)) fist_nonbond_env%charges => charges
      IF (PRESENT(rlist_lowsq)) fist_nonbond_env%rlist_lowsq => rlist_lowsq
      IF (PRESENT(nonbonded)) fist_nonbond_env%nonbonded => nonbonded
      IF (PRESENT(r_last_update)) &
         fist_nonbond_env%r_last_update => r_last_update
      IF (PRESENT(r_last_update_pbc)) &
         fist_nonbond_env%r_last_update_pbc => r_last_update_pbc
      IF (PRESENT(rshell_last_update_pbc)) &
         fist_nonbond_env%rshell_last_update_pbc => rshell_last_update_pbc
      IF (PRESENT(rcore_last_update_pbc)) &
         fist_nonbond_env%rcore_last_update_pbc => rcore_last_update_pbc
      IF (PRESENT(cell_last_update)) &
         fist_nonbond_env%cell_last_update => cell_last_update
      IF (PRESENT(lup)) fist_nonbond_env%lup = lup
      IF (PRESENT(aup)) fist_nonbond_env%aup = aup
      IF (PRESENT(ei_scale14)) fist_nonbond_env%ei_scale14 = ei_scale14
      IF (PRESENT(vdw_scale14)) fist_nonbond_env%vdw_scale14 = vdw_scale14
      IF (PRESENT(shift_cutoff)) &
         fist_nonbond_env%shift_cutoff = shift_cutoff
      IF (PRESENT(do_electrostatics)) fist_nonbond_env%do_electrostatics = do_electrostatics
      IF (PRESENT(natom_types)) fist_nonbond_env%natom_types = natom_types
      IF (PRESENT(counter)) fist_nonbond_env%counter = counter
      IF (PRESENT(last_update)) fist_nonbond_env%last_update = last_update
      IF (PRESENT(num_update)) fist_nonbond_env%num_update = num_update
      IF (PRESENT(long_range_correction)) &
         fist_nonbond_env%long_range_correction = long_range_correction
   END SUBROUTINE fist_nonbond_env_set

! **************************************************************************************************
!> \brief allocates and intitializes a fist_nonbond_env
!> \param fist_nonbond_env the object to create
!> \param atomic_kind_set ...
!> \param potparm14 ...
!> \param potparm ...
!> \param do_nonbonded ...
!> \param do_electrostatics ...
!> \param verlet_skin ...
!> \param ewald_rcut ...
!> \param ei_scale14 ...
!> \param vdw_scale14 ...
!> \param shift_cutoff ...
!> \par History
!>      12.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE fist_nonbond_env_create(fist_nonbond_env, atomic_kind_set, &
                                      potparm14, potparm, do_nonbonded, do_electrostatics, verlet_skin, ewald_rcut, &
                                      ei_scale14, vdw_scale14, shift_cutoff)
      TYPE(fist_nonbond_env_type), INTENT(OUT)           :: fist_nonbond_env
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(pair_potential_pp_type), OPTIONAL, POINTER    :: potparm14, potparm
      LOGICAL, INTENT(IN)                                :: do_nonbonded, do_electrostatics
      REAL(KIND=dp), INTENT(IN)                          :: verlet_skin, ewald_rcut, ei_scale14, &
                                                            vdw_scale14
      LOGICAL, INTENT(IN)                                :: shift_cutoff

      NULLIFY (fist_nonbond_env%potparm14)
      NULLIFY (fist_nonbond_env%potparm)
      NULLIFY (fist_nonbond_env%rlist_cut)
      NULLIFY (fist_nonbond_env%rlist_lowsq)
      NULLIFY (fist_nonbond_env%ij_kind_full_fac)
      NULLIFY (fist_nonbond_env%nonbonded)
      NULLIFY (fist_nonbond_env%cell_last_update)
      NULLIFY (fist_nonbond_env%r_last_update)
      NULLIFY (fist_nonbond_env%r_last_update_pbc)
      NULLIFY (fist_nonbond_env%rshell_last_update_pbc)
      NULLIFY (fist_nonbond_env%rcore_last_update_pbc)
      NULLIFY (fist_nonbond_env%eam_data)
      NULLIFY (fist_nonbond_env%quip_data)
      NULLIFY (fist_nonbond_env%nequip_data)
      NULLIFY (fist_nonbond_env%allegro_data)
      NULLIFY (fist_nonbond_env%deepmd_data)
      NULLIFY (fist_nonbond_env%ace_data)
      NULLIFY (fist_nonbond_env%charges)
      CALL init_fist_nonbond_env(fist_nonbond_env, atomic_kind_set, potparm14, &
                                 potparm, do_nonbonded, do_electrostatics, verlet_skin, ewald_rcut, ei_scale14, &
                                 vdw_scale14, shift_cutoff)
   END SUBROUTINE fist_nonbond_env_create

! **************************************************************************************************
!> \brief Purpose: Initialise the FIST nonbond environment.
!> \param fist_nonbond_env the object to create
!> \param atomic_kind_set ...
!> \param potparm14 ...
!> \param potparm ...
!> \param do_nonbonded ...
!> \param do_electrostatics ...
!> \param verlet_skin ...
!> \param ewald_rcut ...
!> \param ei_scale14 ...
!> \param vdw_scale14 ...
!> \param shift_cutoff ...
! **************************************************************************************************
   SUBROUTINE init_fist_nonbond_env(fist_nonbond_env, atomic_kind_set, &
                                    potparm14, potparm, do_nonbonded, do_electrostatics, verlet_skin, ewald_rcut, ei_scale14, &
                                    vdw_scale14, shift_cutoff)

      TYPE(fist_nonbond_env_type), INTENT(INOUT)         :: fist_nonbond_env
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(pair_potential_pp_type), OPTIONAL, POINTER    :: potparm14, potparm
      LOGICAL, INTENT(IN)                                :: do_nonbonded, do_electrostatics
      REAL(KIND=dp), INTENT(IN)                          :: verlet_skin, ewald_rcut, ei_scale14, &
                                                            vdw_scale14
      LOGICAL, INTENT(IN)                                :: shift_cutoff

      INTEGER                                            :: idim, jdim, natom_types
      LOGICAL                                            :: check, use_potparm, use_potparm14
      REAL(KIND=dp)                                      :: fac, rcut, rlow

      use_potparm14 = PRESENT(potparm14)
      IF (use_potparm14) use_potparm14 = use_potparm14 .OR. ASSOCIATED(potparm14)
      use_potparm = PRESENT(potparm)
      IF (use_potparm) use_potparm = use_potparm .OR. ASSOCIATED(potparm)
      NULLIFY (fist_nonbond_env%nonbonded)
      NULLIFY (fist_nonbond_env%r_last_update)
      NULLIFY (fist_nonbond_env%r_last_update_pbc)
      NULLIFY (fist_nonbond_env%rshell_last_update_pbc)
      NULLIFY (fist_nonbond_env%rcore_last_update_pbc)
      NULLIFY (fist_nonbond_env%cell_last_update)
      NULLIFY (fist_nonbond_env%rlist_cut)
      NULLIFY (fist_nonbond_env%rlist_lowsq)
      NULLIFY (fist_nonbond_env%ij_kind_full_fac)
      fist_nonbond_env%unit_type = "ANGSTROM"
      fist_nonbond_env%do_nonbonded = do_nonbonded
      fist_nonbond_env%do_electrostatics = do_electrostatics
      fist_nonbond_env%lup = 0
      fist_nonbond_env%aup = 0
      fist_nonbond_env%ei_scale14 = ei_scale14
      fist_nonbond_env%vdw_scale14 = vdw_scale14
      fist_nonbond_env%shift_cutoff = shift_cutoff
      fist_nonbond_env%counter = 0
      fist_nonbond_env%last_update = 0
      fist_nonbond_env%num_update = 0
      fist_nonbond_env%long_range_correction = 0
      IF (do_nonbonded) THEN
         natom_types = 1
         ! Determine size of kind arrays
         natom_types = SIZE(atomic_kind_set)
         IF (use_potparm14) THEN
            check = (SIZE(potparm14%pot, 1) == natom_types)
            CPASSERT(check)
         END IF
         IF (use_potparm) THEN
            check = (SIZE(potparm%pot, 1) == natom_types)
            CPASSERT(check)
         END IF
         ALLOCATE (fist_nonbond_env%rlist_cut(natom_types, natom_types))
         ALLOCATE (fist_nonbond_env%rlist_lowsq(natom_types, natom_types))
         ALLOCATE (fist_nonbond_env%ij_kind_full_fac(natom_types, natom_types))
         fist_nonbond_env%ij_kind_full_fac = 1.0_dp
         DO idim = 1, natom_types
            DO jdim = idim, natom_types
               IF ((use_potparm) .OR. (use_potparm14)) THEN
                  IF (use_potparm) THEN
                     rcut = SQRT(potparm%pot(idim, jdim)%pot%rcutsq)
                     fac = potparm%pot(idim, jdim)%pot%spl_f%rscale(1)
                     rlow = fac/(potparm%pot(idim, jdim)%pot%pair_spline_data(1)%spline_data%xn)
                  ELSE
                     rcut = SQRT(potparm14%pot(idim, jdim)%pot%rcutsq)
                     fac = potparm14%pot(idim, jdim)%pot%spl_f%rscale(1)
                     rlow = fac/(potparm14%pot(idim, jdim)%pot%pair_spline_data(1)%spline_data%xn)
                  END IF
                  ! Warning: rlist_rcut should only be used by the neighbor list
                  ! algorithm. It is not the cutoff for the evaluation of the
                  ! interactions because rlist_rcut includes the Verlet skin.
                  rcut = MAX(rcut, ewald_rcut) + verlet_skin
                  fist_nonbond_env%rlist_cut(idim, jdim) = rcut
                  fist_nonbond_env%rlist_cut(jdim, idim) = rcut
                  rlow = rlow*(1.06_dp)**2 ! 1.06_dp in order to have 1/2 Emax_spline
                  fist_nonbond_env%rlist_lowsq(idim, jdim) = rlow
                  fist_nonbond_env%rlist_lowsq(jdim, idim) = rlow
                  ! In case of manybody potential the neighbor list will be full.
                  ! This means that for each atom pair (a,b) of the current types,
                  ! atom a is in the neighbor list of b and b is in the neighbor
                  ! list of a. ij_kind_full_fac is used to correct for the double
                  ! counting in the conventional pair potentials cause by this
                  ! situation.
                  IF (ANY(potparm%pot(idim, jdim)%pot%type == tersoff_type)) THEN
                     ! TODO: what if 14 is not of tersoff type while the normal
                     ! nonbond is? (or the reverse). We'd better impose
                     ! consistency.
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                  END IF
                  IF (ANY(potparm%pot(idim, jdim)%pot%type == siepmann_type)) THEN
                     ! TODO:see tersoff_type
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                  END IF
                  IF (ANY(potparm%pot(idim, jdim)%pot%type == ace_type)) THEN
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                     fist_nonbond_env%ij_kind_full_fac(jdim, idim) = 0.5_dp
                  END IF
                  IF (ANY(potparm%pot(idim, jdim)%pot%type == gal_type)) THEN
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                  END IF
                  IF (ANY(potparm%pot(idim, jdim)%pot%type == gal21_type)) THEN
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                  END IF
                  IF (ANY(potparm%pot(idim, jdim)%pot%type == nequip_type)) THEN
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                     fist_nonbond_env%ij_kind_full_fac(idim, jdim) = 0.5_dp
                  END IF
               ELSE
                  ! In case we don't use potparm for initialization let's account
                  ! only for the real-space part of the Ewald sum.
                  fist_nonbond_env%rlist_cut(idim, jdim) = ewald_rcut
                  fist_nonbond_env%rlist_cut(jdim, idim) = ewald_rcut
                  fist_nonbond_env%rlist_lowsq(idim, jdim) = 0.0_dp
                  fist_nonbond_env%rlist_lowsq(jdim, idim) = 0.0_dp
               END IF
            END DO
         END DO
         IF (use_potparm14) fist_nonbond_env%potparm14 => potparm14
         IF (use_potparm) fist_nonbond_env%potparm => potparm
         fist_nonbond_env%natom_types = natom_types
      ELSE
         NULLIFY (fist_nonbond_env%potparm)
         NULLIFY (fist_nonbond_env%potparm14)
      END IF
   END SUBROUTINE init_fist_nonbond_env

! **************************************************************************************************
!> \brief releases the given fist_nonbond_env (see doc/ReferenceCounting.html)
!> \param fist_nonbond_env the object to release
!> \par History
!>      12.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE fist_nonbond_env_release(fist_nonbond_env)
      TYPE(fist_nonbond_env_type), INTENT(INOUT)         :: fist_nonbond_env

      IF (ASSOCIATED(fist_nonbond_env%nonbonded)) THEN
         CALL fist_neighbor_deallocate(fist_nonbond_env%nonbonded)
      END IF
      ! Release potparm
      CALL pair_potential_pp_release(fist_nonbond_env%potparm)
      ! Release potparm14
      CALL pair_potential_pp_release(fist_nonbond_env%potparm14)
      IF (ASSOCIATED(fist_nonbond_env%r_last_update)) THEN
         DEALLOCATE (fist_nonbond_env%r_last_update)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%r_last_update_pbc)) THEN
         DEALLOCATE (fist_nonbond_env%r_last_update_pbc)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%charges)) THEN
         DEALLOCATE (fist_nonbond_env%charges)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%eam_data)) THEN
         DEALLOCATE (fist_nonbond_env%eam_data)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%quip_data)) THEN
         IF (ASSOCIATED(fist_nonbond_env%quip_data%force)) THEN
            DEALLOCATE (fist_nonbond_env%quip_data%force)
         END IF
         IF (ASSOCIATED(fist_nonbond_env%quip_data%use_indices)) THEN
            DEALLOCATE (fist_nonbond_env%quip_data%use_indices)
         END IF
         DEALLOCATE (fist_nonbond_env%quip_data)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%nequip_data)) THEN
         IF (ASSOCIATED(fist_nonbond_env%nequip_data%force)) THEN
            DEALLOCATE (fist_nonbond_env%nequip_data%force)
         END IF
         IF (ASSOCIATED(fist_nonbond_env%nequip_data%use_indices)) THEN
            DEALLOCATE (fist_nonbond_env%nequip_data%use_indices)
         END IF
         CALL torch_model_release(fist_nonbond_env%nequip_data%model)
         DEALLOCATE (fist_nonbond_env%nequip_data)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%allegro_data)) THEN
         IF (ASSOCIATED(fist_nonbond_env%allegro_data%force)) THEN
            DEALLOCATE (fist_nonbond_env%allegro_data%force)
         END IF
         IF (ASSOCIATED(fist_nonbond_env%allegro_data%use_indices)) THEN
            DEALLOCATE (fist_nonbond_env%allegro_data%use_indices)
         END IF
         CALL torch_model_release(fist_nonbond_env%allegro_data%model)
         DEALLOCATE (fist_nonbond_env%allegro_data)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%deepmd_data)) THEN
         IF (ASSOCIATED(fist_nonbond_env%deepmd_data%force)) THEN
            DEALLOCATE (fist_nonbond_env%deepmd_data%force)
         END IF
         IF (ASSOCIATED(fist_nonbond_env%deepmd_data%use_indices)) THEN
            DEALLOCATE (fist_nonbond_env%deepmd_data%use_indices)
         END IF
         CALL deepmd_model_release(fist_nonbond_env%deepmd_data%model)
         DEALLOCATE (fist_nonbond_env%deepmd_data)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%ace_data)) THEN
         IF (ALLOCATED(fist_nonbond_env%ace_data%use_indices)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%use_indices)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%inverse_index_map)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%inverse_index_map)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%force)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%force)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%atpos)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%atpos)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%uctype)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%uctype)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%attype)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%attype)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%origin)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%origin)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%shift)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%shift)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%neiat)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%neiat)
         END IF
         IF (ALLOCATED(fist_nonbond_env%ace_data%nlist)) THEN
            DEALLOCATE (fist_nonbond_env%ace_data%nlist)
         END IF
         CALL ace_model_release(fist_nonbond_env%ace_data%model)
         DEALLOCATE (fist_nonbond_env%ace_data)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%rshell_last_update_pbc)) THEN
         DEALLOCATE (fist_nonbond_env%rshell_last_update_pbc)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%rcore_last_update_pbc)) THEN
         DEALLOCATE (fist_nonbond_env%rcore_last_update_pbc)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%cell_last_update)) THEN
         CALL cell_release(fist_nonbond_env%cell_last_update)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%ij_kind_full_fac)) THEN
         DEALLOCATE (fist_nonbond_env%ij_kind_full_fac)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%rlist_cut)) THEN
         DEALLOCATE (fist_nonbond_env%rlist_cut)
      END IF
      IF (ASSOCIATED(fist_nonbond_env%rlist_lowsq)) THEN
         DEALLOCATE (fist_nonbond_env%rlist_lowsq)
      END IF
   END SUBROUTINE fist_nonbond_env_release

END MODULE fist_nonbond_env_types
