!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief computes the conserved quantities for a given md ensemble
!>      and also kinetic energies, thermo/barostat stuff
!> \author gtb, 05.02.2003
! **************************************************************************************************
MODULE md_conserved_quantities
   USE atomic_kind_list_types,          ONLY: atomic_kind_list_type
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind
   USE barostat_utils,                  ONLY: get_baro_energies
   USE cell_types,                      ONLY: cell_type
   USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                              cp_subsys_type
   USE distribution_1d_types,           ONLY: distribution_1d_type
   USE extended_system_types,           ONLY: npt_info_type
   USE force_env_types,                 ONLY: force_env_get,&
                                              force_env_type
   USE input_constants,                 ONLY: &
        isokin_ensemble, langevin_ensemble, npe_f_ensemble, npe_i_ensemble, &
        nph_uniaxial_damped_ensemble, nph_uniaxial_ensemble, npt_f_ensemble, npt_i_ensemble, &
        npt_ia_ensemble, nve_ensemble, nvt_adiabatic_ensemble, nvt_ensemble, reftraj_ensemble
   USE input_section_types,             ONLY: section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: dp
   USE mathconstants,                   ONLY: zero
   USE md_ener_types,                   ONLY: md_ener_type,&
                                              zero_md_ener
   USE md_environment_types,            ONLY: get_md_env,&
                                              md_environment_type,&
                                              set_md_env
   USE message_passing,                 ONLY: mp_comm_type,&
                                              mp_para_env_type
   USE particle_list_types,             ONLY: particle_list_type
   USE particle_types,                  ONLY: particle_type
   USE physcon,                         ONLY: kelvin
   USE qmmm_types,                      ONLY: qmmm_env_type
   USE qmmm_types_low,                  ONLY: force_mixing_label_QM_dynamics
   USE qmmmx_types,                     ONLY: qmmmx_env_type
   USE shell_potential_types,           ONLY: shell_kind_type
   USE simpar_types,                    ONLY: simpar_type
   USE thermostat_types,                ONLY: thermostat_type
   USE thermostat_utils,                ONLY: get_thermostat_energies
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: compute_conserved_quantity, calc_nfree_qm
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'md_conserved_quantities'

CONTAINS

! **************************************************************************************************
!> \brief calculates conserved quantity.
!> \param md_env ...
!> \param md_ener ...
!> \param tkind ...
!> \param tshell ...
!> \param natom ...
!> \par Input Arguments
!>     md_env is the md_environment
!>     epot is the total potential energy
!> \par Output Arguments
!>     cons is the conserved quantity
!> \par Output Optional Arguments
!>     cons_rel : relative cons. quantity (to the first md step)
!>     ekin : kinetic energy of particles
!>     temp : temperature
!>     temp_qm : temperature of the QM system in a QM/MM calculation
!> \par History
!>      none
!> \author gloria
! **************************************************************************************************
   SUBROUTINE compute_conserved_quantity(md_env, md_ener, tkind, tshell, &
                                         natom)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), POINTER                        :: md_ener
      LOGICAL, INTENT(IN)                                :: tkind, tshell
      INTEGER, INTENT(IN)                                :: natom

      INTEGER                                            :: ikind, nfree_qm, nkind
      INTEGER, POINTER                                   :: itimes
      LOGICAL                                            :: init
      REAL(KIND=dp), POINTER                             :: constant
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(simpar_type), POINTER                         :: simpar

      NULLIFY (itimes, para_env, simpar)

      CALL zero_md_ener(md_ener, tkind, tshell)

      CALL get_md_env(md_env=md_env, &
                      constant=constant, &
                      itimes=itimes, &
                      init=init, &
                      simpar=simpar, &
                      para_env=para_env)

      CALL get_part_ke(md_env, md_ener, tkind, tshell, para_env)

      IF (md_ener%nfree /= 0) THEN
         md_ener%temp_part = 2.0_dp*md_ener%ekin/REAL(simpar%nfree, KIND=dp)
         md_ener%temp_part = md_ener%temp_part*kelvin
      END IF

      nfree_qm = calc_nfree_qm(md_env, md_ener)
      IF (nfree_qm > 0) THEN
         md_ener%temp_qm = 2.0_dp*md_ener%ekin_qm/REAL(nfree_qm, KIND=dp)
         md_ener%temp_qm = md_ener%temp_qm*kelvin
      END IF

      IF (md_ener%nfree_shell > 0) THEN
         md_ener%temp_shell = 2.0_dp*md_ener%ekin_shell/REAL(md_ener%nfree_shell, KIND=dp)
         md_ener%temp_shell = md_ener%temp_shell*kelvin
      END IF

      IF (tkind) THEN
         nkind = SIZE(md_ener%temp_kind)
         DO ikind = 1, nkind
            md_ener%temp_kind(ikind) = 2.0_dp* &
                                       md_ener%ekin_kind(ikind)/REAL(md_ener%nfree_kind(ikind), KIND=dp)
            md_ener%temp_kind(ikind) = md_ener%temp_kind(ikind)*kelvin
         END DO
         IF (tshell) THEN
            DO ikind = 1, nkind
               md_ener%temp_shell_kind(ikind) = 2.0_dp* &
                                                md_ener%ekin_shell_kind(ikind)/REAL(md_ener%nfree_shell_kind(ikind), KIND=dp)
               md_ener%temp_shell_kind(ikind) = md_ener%temp_shell_kind(ikind)*kelvin
            END DO
         END IF
      END IF

      SELECT CASE (simpar%ensemble)
      CASE DEFAULT
         CPABORT('Unknown ensemble')
      CASE (isokin_ensemble)
         md_ener%constant = md_ener%ekin
      CASE (reftraj_ensemble) ! no constant of motion available
         md_ener%constant = md_ener%epot
      CASE (nve_ensemble)
         CALL get_econs_nve(md_env, md_ener, para_env)
      CASE (nvt_ensemble)
         CALL get_econs_nvt(md_env, md_ener, para_env)
      CASE (npt_i_ensemble, npt_f_ensemble, npt_ia_ensemble)
         CALL get_econs_npt(md_env, md_ener, para_env)
         md_ener%temp_baro = md_ener%temp_baro*kelvin
      CASE (nph_uniaxial_ensemble)
         CALL get_econs_nph_uniaxial(md_env, md_ener)
         md_ener%temp_baro = md_ener%temp_baro*kelvin
      CASE (nph_uniaxial_damped_ensemble)
         CALL get_econs_nph_uniaxial(md_env, md_ener)
         md_ener%temp_baro = md_ener%temp_baro*kelvin
      CASE (langevin_ensemble)
         md_ener%constant = md_ener%ekin + md_ener%epot
      CASE (npe_f_ensemble, npe_i_ensemble)
         CALL get_econs_npe(md_env, md_ener, para_env)
         md_ener%temp_baro = md_ener%temp_baro*kelvin
      CASE (nvt_adiabatic_ensemble)
         CALL get_econs_nvt_adiabatic(md_env, md_ener, para_env)
      END SELECT

      IF (init) THEN
         ! If the value was not read from input let's set it at the begin of the MD
         IF (constant == 0.0_dp) THEN
            constant = md_ener%constant
            CALL set_md_env(md_env=md_env, constant=constant)
         END IF
      ELSE
         CALL get_md_env(md_env=md_env, constant=constant)
         md_ener%delta_cons = (md_ener%constant - constant)/REAL(natom, KIND=dp)*kelvin
      END IF

   END SUBROUTINE compute_conserved_quantity

! **************************************************************************************************
!> \brief Calculates the number of QM degress of freedom
!> \param md_env ...
!> \param md_ener ...
!> \return ...
! **************************************************************************************************
   FUNCTION calc_nfree_qm(md_env, md_ener) RESULT(nfree_qm)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), POINTER                        :: md_ener
      INTEGER                                            :: nfree_qm

      INTEGER                                            :: ip
      INTEGER, POINTER                                   :: cur_indices(:), cur_labels(:)
      TYPE(cp_subsys_type), POINTER                      :: subsys
      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(particle_list_type), POINTER                  :: particles
      TYPE(qmmm_env_type), POINTER                       :: qmmm_env
      TYPE(qmmmx_env_type), POINTER                      :: qmmmx_env
      TYPE(section_vals_type), POINTER                   :: force_env_section

      NULLIFY (qmmm_env, qmmmx_env, subsys, particles, force_env, force_env_section)
      nfree_qm = 0

      CALL get_md_env(md_env, force_env=force_env)
      CALL force_env_get(force_env, &
                         subsys=subsys, &
                         qmmm_env=qmmm_env, &
                         qmmmx_env=qmmmx_env, &
                         force_env_section=force_env_section)

      IF (ASSOCIATED(qmmm_env)) THEN ! conventional QM/MM
         CALL cp_subsys_get(subsys, particles=particles)
         ! The degrees of freedom for the quantum part of the system
         ! are set to 3*Number of QM atoms and to simpar%nfree in case all the MM
         ! system is treated at QM level (not really QM/MM, just for consistency).
         ! The degree of freedom will not be correct if 1-3 atoms are treated only
         ! MM. In this case we should take care of rotations
         nfree_qm = 3*SIZE(qmmm_env%qm%qm_atom_index)
         IF (nfree_qm == 3*(particles%n_els)) nfree_qm = md_ener%nfree
      END IF

      IF (ASSOCIATED(qmmmx_env)) THEN ! doing force mixing
         CALL section_vals_val_get(force_env_section, "QMMM%FORCE_MIXING%RESTART_INFO%INDICES", i_vals=cur_indices)
         CALL section_vals_val_get(force_env_section, "QMMM%FORCE_MIXING%RESTART_INFO%LABELS", i_vals=cur_labels)
         nfree_qm = 0
         DO ip = 1, SIZE(cur_indices)
            IF (cur_labels(ip) >= force_mixing_label_QM_dynamics) THEN ! this is a QM atom
               nfree_qm = nfree_qm + 3
            END IF
         END DO
      END IF

      CPASSERT(.NOT. (ASSOCIATED(qmmm_env) .AND. ASSOCIATED(qmmmx_env)))
   END FUNCTION calc_nfree_qm

! **************************************************************************************************
!> \brief calculates conserved quantity for nvt ensemble
!> \param md_env ...
!> \param md_ener ...
!> \param para_env ...
!> \par History
!>      none
!> \author gloria
! **************************************************************************************************
   SUBROUTINE get_econs_nve(md_env, md_ener, para_env)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), INTENT(inout)                  :: md_ener
      TYPE(mp_para_env_type), POINTER                    :: para_env

      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(thermostat_type), POINTER                     :: thermostat_coeff, thermostat_shell

      NULLIFY (force_env, thermostat_coeff, thermostat_shell)

      CALL get_md_env(md_env, force_env=force_env, thermostat_coeff=thermostat_coeff, &
                      thermostat_shell=thermostat_shell)
      md_ener%constant = md_ener%ekin + md_ener%epot + md_ener%ekin_shell

      CALL get_thermostat_energies(thermostat_shell, md_ener%thermostat_shell_pot, &
                                   md_ener%thermostat_shell_kin, para_env)
      md_ener%constant = md_ener%constant + md_ener%thermostat_shell_kin + md_ener%thermostat_shell_pot

   END SUBROUTINE get_econs_nve

! **************************************************************************************************
!> \brief calculates conserved quantity for nvt ensemble
!> \param md_env ...
!> \param md_ener ...
!> \param para_env ...
!> \par History
!>      none
!> \author gloria
! **************************************************************************************************
   SUBROUTINE get_econs_nvt_adiabatic(md_env, md_ener, para_env)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), INTENT(inout)                  :: md_ener
      TYPE(mp_para_env_type), POINTER                    :: para_env

      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(thermostat_type), POINTER                     :: thermostat_fast, thermostat_slow

      NULLIFY (force_env, thermostat_fast, thermostat_slow)
      CALL get_md_env(md_env, force_env=force_env, thermostat_fast=thermostat_fast, &
                      thermostat_slow=thermostat_slow)
      CALL get_thermostat_energies(thermostat_fast, md_ener%thermostat_fast_pot, &
                                   md_ener%thermostat_fast_kin, para_env)
      md_ener%constant = md_ener%ekin + md_ener%epot + &
                         md_ener%thermostat_fast_kin + md_ener%thermostat_fast_pot
      CALL get_thermostat_energies(thermostat_slow, md_ener%thermostat_slow_pot, &
                                   md_ener%thermostat_slow_kin, para_env)
      md_ener%constant = md_ener%constant + &
                         md_ener%thermostat_slow_kin + md_ener%thermostat_slow_pot

   END SUBROUTINE get_econs_nvt_adiabatic

! **************************************************************************************************
!> \brief calculates conserved quantity for nvt ensemble
!> \param md_env ...
!> \param md_ener ...
!> \param para_env ...
!> \par History
!>      none
!> \author gloria
! **************************************************************************************************
   SUBROUTINE get_econs_nvt(md_env, md_ener, para_env)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), INTENT(inout)                  :: md_ener
      TYPE(mp_para_env_type), POINTER                    :: para_env

      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(thermostat_type), POINTER                     :: thermostat_coeff, thermostat_part, &
                                                            thermostat_shell

      NULLIFY (force_env, thermostat_part, thermostat_coeff, thermostat_shell)
      CALL get_md_env(md_env, force_env=force_env, thermostat_part=thermostat_part, &
                      thermostat_coeff=thermostat_coeff, thermostat_shell=thermostat_shell)
      CALL get_thermostat_energies(thermostat_part, md_ener%thermostat_part_pot, &
                                   md_ener%thermostat_part_kin, para_env)
      md_ener%constant = md_ener%ekin + md_ener%epot + md_ener%ekin_shell + &
                         md_ener%thermostat_part_kin + md_ener%thermostat_part_pot

      CALL get_thermostat_energies(thermostat_shell, md_ener%thermostat_shell_pot, &
                                   md_ener%thermostat_shell_kin, para_env)
      md_ener%constant = md_ener%constant + md_ener%thermostat_shell_kin + md_ener%thermostat_shell_pot

   END SUBROUTINE get_econs_nvt

! **************************************************************************************************
!> \brief calculates conserved quantity for npe ensemble
!> \param md_env ...
!> \param md_ener ...
!> \param para_env ...
!> \par History
!>      none
!> \author  marcella (02-2008)
! **************************************************************************************************
   SUBROUTINE get_econs_npe(md_env, md_ener, para_env)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), INTENT(inout)                  :: md_ener
      TYPE(mp_para_env_type), POINTER                    :: para_env

      INTEGER                                            :: nfree
      TYPE(cell_type), POINTER                           :: box
      TYPE(npt_info_type), POINTER                       :: npt(:, :)
      TYPE(simpar_type), POINTER                         :: simpar
      TYPE(thermostat_type), POINTER                     :: thermostat_baro, thermostat_shell

      NULLIFY (thermostat_baro, thermostat_shell, npt)
      CALL get_md_env(md_env, thermostat_baro=thermostat_baro, &
                      simpar=simpar, npt=npt, cell=box, &
                      thermostat_shell=thermostat_shell)
      CALL get_baro_energies(box, simpar, npt, md_ener%baro_kin, &
                             md_ener%baro_pot)
      nfree = SIZE(npt, 1)*SIZE(npt, 2)
      md_ener%temp_baro = 2.0_dp*md_ener%baro_kin/nfree

      md_ener%constant = md_ener%ekin + md_ener%epot + md_ener%ekin_shell &
                         + md_ener%baro_kin + md_ener%baro_pot

      CALL get_thermostat_energies(thermostat_shell, md_ener%thermostat_shell_pot, &
                                   md_ener%thermostat_shell_kin, para_env)
      md_ener%constant = md_ener%constant + md_ener%thermostat_shell_kin + &
                         md_ener%thermostat_shell_pot

   END SUBROUTINE get_econs_npe

! **************************************************************************************************
!> \brief calculates conserved quantity for npt ensemble
!> \param md_env ...
!> \param md_ener ...
!> \param para_env ...
!> \par History
!>      none
!> \author gloria
! **************************************************************************************************
   SUBROUTINE get_econs_npt(md_env, md_ener, para_env)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), INTENT(inout)                  :: md_ener
      TYPE(mp_para_env_type), POINTER                    :: para_env

      INTEGER                                            :: nfree
      TYPE(cell_type), POINTER                           :: box
      TYPE(npt_info_type), POINTER                       :: npt(:, :)
      TYPE(simpar_type), POINTER                         :: simpar
      TYPE(thermostat_type), POINTER                     :: thermostat_baro, thermostat_part, &
                                                            thermostat_shell

      NULLIFY (thermostat_baro, thermostat_part, thermostat_shell, npt, simpar, box)
      CALL get_md_env(md_env, thermostat_part=thermostat_part, thermostat_baro=thermostat_baro, &
                      simpar=simpar, npt=npt, cell=box, thermostat_shell=thermostat_shell)
      CALL get_thermostat_energies(thermostat_part, md_ener%thermostat_part_pot, &
                                   md_ener%thermostat_part_kin, para_env)
      CALL get_thermostat_energies(thermostat_baro, md_ener%thermostat_baro_pot, &
                                   md_ener%thermostat_baro_kin, para_env)
      CALL get_baro_energies(box, simpar, npt, md_ener%baro_kin, md_ener%baro_pot)
      nfree = SIZE(npt, 1)*SIZE(npt, 2)
      md_ener%temp_baro = 2.0_dp*md_ener%baro_kin/nfree

      md_ener%constant = md_ener%ekin + md_ener%epot + md_ener%ekin_shell &
                         + md_ener%thermostat_part_kin + md_ener%thermostat_part_pot &
                         + md_ener%thermostat_baro_kin + md_ener%thermostat_baro_pot &
                         + md_ener%baro_kin + md_ener%baro_pot

      CALL get_thermostat_energies(thermostat_shell, md_ener%thermostat_shell_pot, &
                                   md_ener%thermostat_shell_kin, para_env)
      md_ener%constant = md_ener%constant + md_ener%thermostat_shell_kin + md_ener%thermostat_shell_pot

   END SUBROUTINE get_econs_npt

! **************************************************************************************************
!> \brief calculates conserved quantity for nph_uniaxial
!> \param md_env ...
!> \param md_ener ...
!> \par History
!>      none
!> \author cjm
! **************************************************************************************************
   SUBROUTINE get_econs_nph_uniaxial(md_env, md_ener)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), INTENT(inout)                  :: md_ener

      TYPE(cell_type), POINTER                           :: box
      TYPE(npt_info_type), POINTER                       :: npt(:, :)
      TYPE(simpar_type), POINTER                         :: simpar

      CALL get_md_env(md_env, simpar=simpar, npt=npt, cell=box)

      CALL get_baro_energies(box, simpar, npt, md_ener%baro_kin, md_ener%baro_pot)
      md_ener%temp_baro = 2.0_dp*md_ener%baro_kin
      md_ener%constant = md_ener%ekin + md_ener%epot + md_ener%baro_kin + md_ener%baro_pot
   END SUBROUTINE get_econs_nph_uniaxial

! **************************************************************************************************
!> \brief Calculates kinetic energy of particles
!> \param md_env ...
!> \param md_ener ...
!> \param tkind ...
!> \param tshell ...
!> \param group ...
!> \par History
!>      none
!> \author CJM
! **************************************************************************************************
   SUBROUTINE get_part_ke(md_env, md_ener, tkind, tshell, group)
      TYPE(md_environment_type), POINTER                 :: md_env
      TYPE(md_ener_type), POINTER                        :: md_ener
      LOGICAL, INTENT(IN)                                :: tkind, tshell

      CLASS(mp_comm_type), INTENT(IN)                     :: group

      INTEGER                                            :: i, iparticle, iparticle_kind, &
                                                            iparticle_local, nparticle_kind, &
                                                            nparticle_local, shell_index
      INTEGER, POINTER                                   :: cur_indices(:), cur_labels(:)
      LOGICAL                                            :: is_shell
      REAL(KIND=dp)                                      :: ekin_c, ekin_com, ekin_s, mass
      TYPE(atomic_kind_list_type), POINTER               :: atomic_kinds
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(atomic_kind_type), POINTER                    :: atomic_kind
      TYPE(cp_subsys_type), POINTER                      :: subsys
      TYPE(distribution_1d_type), POINTER                :: local_particles
      TYPE(force_env_type), POINTER                      :: force_env
      TYPE(particle_list_type), POINTER                  :: core_particles, particles, &
                                                            shell_particles
      TYPE(particle_type), DIMENSION(:), POINTER         :: core_particle_set, particle_set, &
                                                            shell_particle_set
      TYPE(qmmm_env_type), POINTER                       :: qmmm_env
      TYPE(qmmmx_env_type), POINTER                      :: qmmmx_env
      TYPE(section_vals_type), POINTER                   :: force_env_section
      TYPE(shell_kind_type), POINTER                     :: shell

      NULLIFY (force_env, qmmm_env, qmmmx_env, subsys, force_env_section)
      CALL get_md_env(md_env, force_env=force_env)
      CALL force_env_get(force_env, &
                         subsys=subsys, &
                         qmmm_env=qmmm_env, &
                         qmmmx_env=qmmmx_env, &
                         force_env_section=force_env_section)

      CALL cp_subsys_get(subsys=subsys, &
                         atomic_kinds=atomic_kinds, &
                         local_particles=local_particles, &
                         particles=particles, shell_particles=shell_particles, &
                         core_particles=core_particles)

      nparticle_kind = atomic_kinds%n_els
      atomic_kind_set => atomic_kinds%els

      ekin_s = zero
      ekin_c = zero
      ekin_com = zero
      IF (tkind) THEN
         md_ener%nfree_kind = 0
         IF (tshell) THEN
            md_ener%nfree_shell_kind = 0
         END IF
      END IF

      particle_set => particles%els
      IF (tshell) THEN
         shell_particle_set => shell_particles%els
         core_particle_set => core_particles%els
         DO iparticle_kind = 1, nparticle_kind
            atomic_kind => atomic_kind_set(iparticle_kind)
            CALL get_atomic_kind(atomic_kind=atomic_kind, mass=mass, &
                                 shell_active=is_shell, shell=shell)
            nparticle_local = local_particles%n_el(iparticle_kind)
            IF (is_shell) THEN
               DO iparticle_local = 1, nparticle_local
                  iparticle = local_particles%list(iparticle_kind)%array(iparticle_local)
                  shell_index = particle_set(iparticle)%shell_index
                  !ekin
                  ekin_com = 0.5_dp*mass* &
                             (particle_set(iparticle)%v(1)*particle_set(iparticle)%v(1) &
                              + particle_set(iparticle)%v(2)*particle_set(iparticle)%v(2) &
                              + particle_set(iparticle)%v(3)*particle_set(iparticle)%v(3))
                  !vcom
                  md_ener%vcom(1) = md_ener%vcom(1) + particle_set(iparticle)%v(1)*mass
                  md_ener%vcom(2) = md_ener%vcom(2) + particle_set(iparticle)%v(2)*mass
                  md_ener%vcom(3) = md_ener%vcom(3) + particle_set(iparticle)%v(3)*mass
                  md_ener%total_mass = md_ener%total_mass + mass

                  md_ener%ekin = md_ener%ekin + ekin_com
                  ekin_c = 0.5_dp*shell%mass_core* &
                           (core_particle_set(shell_index)%v(1)*core_particle_set(shell_index)%v(1) &
                            + core_particle_set(shell_index)%v(2)*core_particle_set(shell_index)%v(2) &
                            + core_particle_set(shell_index)%v(3)*core_particle_set(shell_index)%v(3))
                  ekin_s = 0.5_dp*shell%mass_shell* &
                           (shell_particle_set(shell_index)%v(1)*shell_particle_set(shell_index)%v(1) &
                            + shell_particle_set(shell_index)%v(2)*shell_particle_set(shell_index)%v(2) &
                            + shell_particle_set(shell_index)%v(3)*shell_particle_set(shell_index)%v(3))
                  md_ener%ekin_shell = md_ener%ekin_shell + ekin_c + ekin_s - ekin_com

                  IF (tkind) THEN
                     md_ener%ekin_kind(iparticle_kind) = md_ener%ekin_kind(iparticle_kind) + ekin_com
                     md_ener%nfree_kind(iparticle_kind) = md_ener%nfree_kind(iparticle_kind) + 3
                     md_ener%ekin_shell_kind(iparticle_kind) = md_ener%ekin_shell_kind(iparticle_kind) + &
                                                               ekin_c + ekin_s - ekin_com
                     md_ener%nfree_shell_kind(iparticle_kind) = md_ener%nfree_shell_kind(iparticle_kind) + 3
                  END IF

               END DO ! iparticle_local
            ELSE
               DO iparticle_local = 1, nparticle_local
                  iparticle = local_particles%list(iparticle_kind)%array(iparticle_local)
                  ekin_com = 0.5_dp*mass* &
                             (particle_set(iparticle)%v(1)*particle_set(iparticle)%v(1) &
                              + particle_set(iparticle)%v(2)*particle_set(iparticle)%v(2) &
                              + particle_set(iparticle)%v(3)*particle_set(iparticle)%v(3))
                  !vcom
                  md_ener%vcom(1) = md_ener%vcom(1) + particle_set(iparticle)%v(1)*mass
                  md_ener%vcom(2) = md_ener%vcom(2) + particle_set(iparticle)%v(2)*mass
                  md_ener%vcom(3) = md_ener%vcom(3) + particle_set(iparticle)%v(3)*mass
                  md_ener%total_mass = md_ener%total_mass + mass

                  md_ener%ekin = md_ener%ekin + ekin_com
                  IF (tkind) THEN
                     md_ener%ekin_kind(iparticle_kind) = md_ener%ekin_kind(iparticle_kind) + ekin_com
                     md_ener%nfree_kind(iparticle_kind) = md_ener%nfree_kind(iparticle_kind) + 3
                  END IF
               END DO ! iparticle_local
            END IF
         END DO ! iparticle_kind
         IF (tkind) THEN
            CALL group%sum(md_ener%ekin_kind)
            CALL group%sum(md_ener%nfree_kind)
            CALL group%sum(md_ener%ekin_shell_kind)
            CALL group%sum(md_ener%nfree_shell_kind)
         END IF
         ! sum all contributions to energy over calculated parts on all processors
         CALL group%sum(md_ener%ekin_shell)
      ELSE
         DO iparticle_kind = 1, nparticle_kind
            atomic_kind => atomic_kind_set(iparticle_kind)
            CALL get_atomic_kind(atomic_kind=atomic_kind, mass=mass)
            nparticle_local = local_particles%n_el(iparticle_kind)
            DO iparticle_local = 1, nparticle_local
               iparticle = local_particles%list(iparticle_kind)%array(iparticle_local)
               ! ekin
               ekin_com = 0.5_dp*mass* &
                          (particle_set(iparticle)%v(1)*particle_set(iparticle)%v(1) &
                           + particle_set(iparticle)%v(2)*particle_set(iparticle)%v(2) &
                           + particle_set(iparticle)%v(3)*particle_set(iparticle)%v(3))

               !vcom
               md_ener%vcom(1) = md_ener%vcom(1) + particle_set(iparticle)%v(1)*mass
               md_ener%vcom(2) = md_ener%vcom(2) + particle_set(iparticle)%v(2)*mass
               md_ener%vcom(3) = md_ener%vcom(3) + particle_set(iparticle)%v(3)*mass
               md_ener%total_mass = md_ener%total_mass + mass

               md_ener%ekin = md_ener%ekin + ekin_com
               IF (tkind) THEN
                  md_ener%ekin_kind(iparticle_kind) = md_ener%ekin_kind(iparticle_kind) + ekin_com
                  md_ener%nfree_kind(iparticle_kind) = md_ener%nfree_kind(iparticle_kind) + 3
               END IF
            END DO
         END DO ! iparticle_kind
         IF (tkind) THEN
            CALL group%sum(md_ener%ekin_kind)
            CALL group%sum(md_ener%nfree_kind)
         END IF
      END IF

      ! sum all contributions to energy over calculated parts on all processors
      CALL group%sum(md_ener%ekin)
      CALL group%sum(md_ener%vcom)
      CALL group%sum(md_ener%total_mass)
      md_ener%vcom = md_ener%vcom/md_ener%total_mass
      !
      ! Compute the QM/MM kinetic energy

      IF (ASSOCIATED(qmmm_env)) THEN ! conventional QM/MM
         DO i = 1, SIZE(qmmm_env%qm%qm_atom_index)
            iparticle = qmmm_env%qm%qm_atom_index(i)
            mass = particle_set(iparticle)%atomic_kind%mass
            md_ener%ekin_qm = md_ener%ekin_qm + 0.5_dp*mass* &
                              (particle_set(iparticle)%v(1)*particle_set(iparticle)%v(1) &
                               + particle_set(iparticle)%v(2)*particle_set(iparticle)%v(2) &
                               + particle_set(iparticle)%v(3)*particle_set(iparticle)%v(3))
         END DO
      END IF

      IF (ASSOCIATED(qmmmx_env)) THEN ! doing force mixing
         CALL section_vals_val_get(force_env_section, "QMMM%FORCE_MIXING%RESTART_INFO%INDICES", i_vals=cur_indices)
         CALL section_vals_val_get(force_env_section, "QMMM%FORCE_MIXING%RESTART_INFO%LABELS", i_vals=cur_labels)
         DO i = 1, SIZE(cur_indices)
            IF (cur_labels(i) >= force_mixing_label_QM_dynamics) THEN ! this is a QM atom
               iparticle = cur_indices(i)
               mass = particle_set(iparticle)%atomic_kind%mass
               md_ener%ekin_qm = md_ener%ekin_qm + 0.5_dp*mass* &
                                 (particle_set(iparticle)%v(1)*particle_set(iparticle)%v(1) &
                                  + particle_set(iparticle)%v(2)*particle_set(iparticle)%v(2) &
                                  + particle_set(iparticle)%v(3)*particle_set(iparticle)%v(3))
            END IF
         END DO
      END IF

      IF (ASSOCIATED(qmmm_env) .AND. ASSOCIATED(qmmmx_env)) &
         CPABORT("get_part_ke: qmmm bug")
   END SUBROUTINE get_part_ke

! **************************************************************************************************

END MODULE md_conserved_quantities
