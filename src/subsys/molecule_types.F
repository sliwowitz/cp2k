!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Define the data structure for the molecule information.
!> \par History
!>      JGH (22.05.2004) add last_atom information
!>      Teodoro Laino [tlaino] 12.2008 - Preparing for VIRTUAL SITE constraints
!>                                       (patch by Marcel Baer)
!> \author MK (29.08.2003)
! **************************************************************************************************
MODULE molecule_types

   USE colvar_types,                    ONLY: colvar_counters,&
                                              colvar_release,&
                                              colvar_type
   USE kinds,                           ONLY: dp
   USE molecule_kind_types,             ONLY: colvar_constraint_type,&
                                              fixd_constraint_type,&
                                              g3x3_constraint_type,&
                                              g4x6_constraint_type,&
                                              get_molecule_kind,&
                                              molecule_kind_type,&
                                              vsite_constraint_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

! *** Global parameters (in this module) ***

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'molecule_types'

! *** Data types ***
! **************************************************************************************************
   TYPE local_colvar_constraint_type
      LOGICAL                       :: init = .FALSE.
      TYPE(colvar_type), POINTER     :: colvar => NULL()
      TYPE(colvar_type), POINTER     :: colvar_old => NULL()
      REAL(KIND=dp)               :: lambda = 0.0_dp, sigma = 0.0_dp
   END TYPE local_colvar_constraint_type

! **************************************************************************************************
   TYPE local_g3x3_constraint_type
      LOGICAL                       :: init = .FALSE.
      REAL(KIND=dp)               :: scale = 0.0_dp, scale_old = 0.0_dp, &
                                     imass1 = 0.0_dp, imass2 = 0.0_dp, imass3 = 0.0_dp
      REAL(KIND=dp), DIMENSION(3) :: fa = 0.0_dp, fb = 0.0_dp, fc = 0.0_dp, &
                                     f_roll1 = 0.0_dp, f_roll2 = 0.0_dp, f_roll3 = 0.0_dp, &
                                     ra_old = 0.0_dp, rb_old = 0.0_dp, rc_old = 0.0_dp, &
                                     va = 0.0_dp, vb = 0.0_dp, vc = 0.0_dp, &
                                     lambda = 0.0_dp, del_lambda = 0.0_dp, lambda_old = 0.0_dp, &
                                     r0_12 = 0.0_dp, r0_13 = 0.0_dp, r0_23 = 0.0_dp
      REAL(KIND=dp), DIMENSION(3, 3) :: amat = 0.0_dp
   END TYPE local_g3x3_constraint_type

! **************************************************************************************************
   TYPE local_g4x6_constraint_type
      LOGICAL                       :: init = .FALSE.
      REAL(KIND=dp)               :: scale = 0.0_dp, scale_old = 0.0_dp, imass1 = 0.0_dp, &
                                     imass2 = 0.0_dp, imass3 = 0.0_dp, imass4 = 0.0_dp
      REAL(KIND=dp), DIMENSION(3) :: fa = 0.0_dp, fb = 0.0_dp, fc = 0.0_dp, fd = 0.0_dp, fe = 0.0_dp, ff = 0.0_dp, &
                                     f_roll1 = 0.0_dp, f_roll2 = 0.0_dp, f_roll3 = 0.0_dp, &
                                     f_roll4 = 0.0_dp, f_roll5 = 0.0_dp, f_roll6 = 0.0_dp, &
                                     ra_old = 0.0_dp, rb_old = 0.0_dp, rc_old = 0.0_dp, &
                                     rd_old = 0.0_dp, re_old = 0.0_dp, rf_old = 0.0_dp, &
                                     va = 0.0_dp, vb = 0.0_dp, vc = 0.0_dp, vd = 0.0_dp, ve = 0.0_dp, vf = 0.0_dp, &
                                     r0_12 = 0.0_dp, r0_13 = 0.0_dp, r0_14 = 0.0_dp, &
                                     r0_23 = 0.0_dp, r0_24 = 0.0_dp, r0_34 = 0.0_dp
      REAL(KIND=dp), DIMENSION(6)   :: lambda = 0.0_dp, del_lambda = 0.0_dp, lambda_old = 0.0_dp
      REAL(KIND=dp), DIMENSION(6, 6) :: amat = 0.0_dp
   END TYPE local_g4x6_constraint_type

! **************************************************************************************************
   TYPE local_states_type
      INTEGER, DIMENSION(:), POINTER :: states => NULL() ! indices of Kohn-Sham states for molecule
      INTEGER                        :: nstates = 0 ! Kohn-Sham states for molecule
   END TYPE local_states_type

! **************************************************************************************************
   TYPE local_constraint_type
      TYPE(local_colvar_constraint_type), DIMENSION(:), POINTER :: lcolv => NULL()
      TYPE(local_g3x3_constraint_type), DIMENSION(:), POINTER :: lg3x3 => NULL()
      TYPE(local_g4x6_constraint_type), DIMENSION(:), POINTER :: lg4x6 => NULL()
   END TYPE local_constraint_type

! **************************************************************************************************
   TYPE global_constraint_type
      TYPE(colvar_counters)                    :: ncolv = colvar_counters()
      INTEGER                                  :: ntot = 0, nrestraint = 0
      INTEGER                                  :: ng3x3 = 0, ng3x3_restraint = 0
      INTEGER                                  :: ng4x6 = 0, ng4x6_restraint = 0
      INTEGER                                  :: nvsite = 0, nvsite_restraint = 0
      TYPE(fixd_constraint_type), DIMENSION(:), POINTER   :: fixd_list => NULL()
      TYPE(colvar_constraint_type), DIMENSION(:), POINTER :: colv_list => NULL()
      TYPE(g3x3_constraint_type), DIMENSION(:), POINTER   :: g3x3_list => NULL()
      TYPE(g4x6_constraint_type), DIMENSION(:), POINTER   :: g4x6_list => NULL()
      TYPE(vsite_constraint_type), DIMENSION(:), POINTER  :: vsite_list => NULL()
      TYPE(local_colvar_constraint_type), DIMENSION(:), POINTER :: lcolv => NULL()
      TYPE(local_g3x3_constraint_type), DIMENSION(:), POINTER :: lg3x3 => NULL()
      TYPE(local_g4x6_constraint_type), DIMENSION(:), POINTER :: lg4x6 => NULL()
   END TYPE global_constraint_type

! **************************************************************************************************
   TYPE molecule_type
      TYPE(molecule_kind_type), POINTER    :: molecule_kind => NULL() ! pointer to molecule kind information
      TYPE(local_states_type), DIMENSION(:), POINTER   :: lmi => NULL() ! local (spin)-states information
      TYPE(local_constraint_type), POINTER :: lci => NULL() ! local molecule constraint info
      INTEGER                              :: first_atom = 0 ! global index of first atom in molecule
      INTEGER                              :: last_atom = 0 ! global index of last atom in molecule
      INTEGER                              :: first_shell = 0 ! global index of first shell atom in molecule
      INTEGER                              :: last_shell = 0 ! global index of last shell atom in molecule
   END TYPE molecule_type

! *** Public data types ***

   PUBLIC :: local_colvar_constraint_type, &
             local_g3x3_constraint_type, &
             local_g4x6_constraint_type, &
             local_constraint_type, &
             local_states_type, &
             global_constraint_type, &
             molecule_type

! *** Public subroutines ***

   PUBLIC :: deallocate_global_constraint, &
             allocate_molecule_set, &
             deallocate_molecule_set, &
             get_molecule, &
             set_molecule, &
             set_molecule_set, &
             molecule_of_atom, &
             get_molecule_set_info

CONTAINS

! **************************************************************************************************
!> \brief   Deallocate a global constraint.
!> \param gci ...
!> \par History
!>      07.2003 created [fawzi]
!>      01.2014 moved from cp_subsys_release() into separate routine.
!> \author  Ole Schuett
! **************************************************************************************************
   SUBROUTINE deallocate_global_constraint(gci)
      TYPE(global_constraint_type), POINTER              :: gci

      INTEGER                                            :: i

      IF (ASSOCIATED(gci)) THEN
         ! List of constraints
         IF (ASSOCIATED(gci%colv_list)) THEN
            DO i = 1, SIZE(gci%colv_list)
               DEALLOCATE (gci%colv_list(i)%i_atoms)
            END DO
            DEALLOCATE (gci%colv_list)
         END IF

         IF (ASSOCIATED(gci%g3x3_list)) &
            DEALLOCATE (gci%g3x3_list)

         IF (ASSOCIATED(gci%g4x6_list)) &
            DEALLOCATE (gci%g4x6_list)

         ! Local information
         IF (ASSOCIATED(gci%lcolv)) THEN
            DO i = 1, SIZE(gci%lcolv)
               CALL colvar_release(gci%lcolv(i)%colvar)
               CALL colvar_release(gci%lcolv(i)%colvar_old)
            END DO
            DEALLOCATE (gci%lcolv)
         END IF

         IF (ASSOCIATED(gci%lg3x3)) &
            DEALLOCATE (gci%lg3x3)

         IF (ASSOCIATED(gci%lg4x6)) &
            DEALLOCATE (gci%lg4x6)

         IF (ASSOCIATED(gci%fixd_list)) &
            DEALLOCATE (gci%fixd_list)

         DEALLOCATE (gci)
      END IF
   END SUBROUTINE deallocate_global_constraint

! **************************************************************************************************
!> \brief   Allocate a molecule set.
!> \param molecule_set ...
!> \param nmolecule ...
!> \date    29.08.2003
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE allocate_molecule_set(molecule_set, nmolecule)
      TYPE(molecule_type), DIMENSION(:), POINTER         :: molecule_set
      INTEGER, INTENT(IN)                                :: nmolecule

      IF (ASSOCIATED(molecule_set)) CALL deallocate_molecule_set(molecule_set)

      ALLOCATE (molecule_set(nmolecule))

   END SUBROUTINE allocate_molecule_set

! **************************************************************************************************
!> \brief   Deallocate a molecule set.
!> \param molecule_set ...
!> \date    29.08.2003
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE deallocate_molecule_set(molecule_set)
      TYPE(molecule_type), DIMENSION(:), POINTER         :: molecule_set

      INTEGER                                            :: imolecule, j

      IF (ASSOCIATED(molecule_set)) THEN

         DO imolecule = 1, SIZE(molecule_set)
            IF (ASSOCIATED(molecule_set(imolecule)%lmi)) THEN
               DO j = 1, SIZE(molecule_set(imolecule)%lmi)
                  IF (ASSOCIATED(molecule_set(imolecule)%lmi(j)%states)) THEN
                     DEALLOCATE (molecule_set(imolecule)%lmi(j)%states)
                  END IF
               END DO
               DEALLOCATE (molecule_set(imolecule)%lmi)
            END IF
            IF (ASSOCIATED(molecule_set(imolecule)%lci)) THEN
               IF (ASSOCIATED(molecule_set(imolecule)%lci%lcolv)) THEN
                  DO j = 1, SIZE(molecule_set(imolecule)%lci%lcolv)
                     CALL colvar_release(molecule_set(imolecule)%lci%lcolv(j)%colvar)
                     CALL colvar_release(molecule_set(imolecule)%lci%lcolv(j)%colvar_old)
                  END DO
                  DEALLOCATE (molecule_set(imolecule)%lci%lcolv)
               END IF
               IF (ASSOCIATED(molecule_set(imolecule)%lci%lg3x3)) THEN
                  DEALLOCATE (molecule_set(imolecule)%lci%lg3x3)
               END IF
               IF (ASSOCIATED(molecule_set(imolecule)%lci%lg4x6)) THEN
                  DEALLOCATE (molecule_set(imolecule)%lci%lg4x6)
               END IF
               DEALLOCATE (molecule_set(imolecule)%lci)
            END IF
         END DO
         DEALLOCATE (molecule_set)

      END IF
      NULLIFY (molecule_set)

   END SUBROUTINE deallocate_molecule_set

! **************************************************************************************************
!> \brief   Get components from a molecule data set.
!> \param molecule ...
!> \param molecule_kind ...
!> \param lmi ...
!> \param lci ...
!> \param lg3x3 ...
!> \param lg4x6 ...
!> \param lcolv ...
!> \param first_atom ...
!> \param last_atom ...
!> \param first_shell ...
!> \param last_shell ...
!> \date    29.08.2003
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE get_molecule(molecule, molecule_kind, lmi, lci, lg3x3, lg4x6, lcolv, &
                           first_atom, last_atom, first_shell, last_shell)

      TYPE(molecule_type), INTENT(IN)                    :: molecule
      TYPE(molecule_kind_type), OPTIONAL, POINTER        :: molecule_kind
      TYPE(local_states_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: lmi
      TYPE(local_constraint_type), OPTIONAL, POINTER     :: lci
      TYPE(local_g3x3_constraint_type), OPTIONAL, &
         POINTER                                         :: lg3x3(:)
      TYPE(local_g4x6_constraint_type), OPTIONAL, &
         POINTER                                         :: lg4x6(:)
      TYPE(local_colvar_constraint_type), DIMENSION(:), &
         OPTIONAL, POINTER                               :: lcolv
      INTEGER, OPTIONAL                                  :: first_atom, last_atom, first_shell, &
                                                            last_shell

      IF (PRESENT(first_atom)) first_atom = molecule%first_atom
      IF (PRESENT(last_atom)) last_atom = molecule%last_atom
      IF (PRESENT(first_shell)) first_shell = molecule%first_shell
      IF (PRESENT(last_shell)) last_shell = molecule%last_shell
      IF (PRESENT(molecule_kind)) molecule_kind => molecule%molecule_kind
      IF (PRESENT(lmi)) lmi => molecule%lmi
      IF (PRESENT(lci)) lci => molecule%lci
      IF (PRESENT(lcolv)) THEN
         IF (ASSOCIATED(molecule%lci)) THEN
            lcolv => molecule%lci%lcolv
         ELSE
            CPABORT("The pointer lci is not associated")
         END IF
      END IF
      IF (PRESENT(lg3x3)) THEN
         IF (ASSOCIATED(molecule%lci)) THEN
            lg3x3 => molecule%lci%lg3x3
         ELSE
            CPABORT("The pointer lci is not associated")
         END IF
      END IF
      IF (PRESENT(lg4x6)) THEN
         IF (ASSOCIATED(molecule%lci)) THEN
            lg4x6 => molecule%lci%lg4x6
         ELSE
            CPABORT("The pointer lci is not associated")
         END IF
      END IF

   END SUBROUTINE get_molecule

! **************************************************************************************************
!> \brief   Set a molecule data set.
!> \param molecule ...
!> \param molecule_kind ...
!> \param lmi ...
!> \param lci ...
!> \param lcolv ...
!> \param lg3x3 ...
!> \param lg4x6 ...
!> \date    29.08.2003
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE set_molecule(molecule, molecule_kind, lmi, lci, lcolv, lg3x3, lg4x6)
      TYPE(molecule_type), INTENT(INOUT)                 :: molecule
      TYPE(molecule_kind_type), OPTIONAL, POINTER        :: molecule_kind
      TYPE(local_states_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: lmi
      TYPE(local_constraint_type), OPTIONAL, POINTER     :: lci
      TYPE(local_colvar_constraint_type), DIMENSION(:), &
         OPTIONAL, POINTER                               :: lcolv
      TYPE(local_g3x3_constraint_type), OPTIONAL, &
         POINTER                                         :: lg3x3(:)
      TYPE(local_g4x6_constraint_type), OPTIONAL, &
         POINTER                                         :: lg4x6(:)

      IF (PRESENT(molecule_kind)) molecule%molecule_kind => molecule_kind
      IF (PRESENT(lmi)) molecule%lmi => lmi
      IF (PRESENT(lci)) molecule%lci => lci
      IF (PRESENT(lcolv)) THEN
         IF (ASSOCIATED(molecule%lci)) THEN
            molecule%lci%lcolv => lcolv
         ELSE
            CPABORT("The pointer lci is not associated")
         END IF
      END IF
      IF (PRESENT(lg3x3)) THEN
         IF (ASSOCIATED(molecule%lci)) THEN
            molecule%lci%lg3x3 => lg3x3
         ELSE
            CPABORT("The pointer lci is not associated")
         END IF
      END IF
      IF (PRESENT(lg4x6)) THEN
         IF (ASSOCIATED(molecule%lci)) THEN
            molecule%lci%lg4x6 => lg4x6
         ELSE
            CPABORT("The pointer lci is not associated")
         END IF
      END IF

   END SUBROUTINE set_molecule

! **************************************************************************************************
!> \brief   Set a molecule data set.
!> \param molecule_set ...
!> \param first_atom ...
!> \param last_atom ...
!> \date    29.08.2003
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE set_molecule_set(molecule_set, first_atom, last_atom)
      TYPE(molecule_type), DIMENSION(:), INTENT(INOUT)   :: molecule_set
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: first_atom, last_atom

      INTEGER                                            :: imolecule

      IF (PRESENT(first_atom)) THEN
         IF (SIZE(first_atom) /= SIZE(molecule_set)) THEN
            CALL cp_abort(__LOCATION__, &
                          "The sizes of first_atom and molecule_set "// &
                          "are different")
         END IF

         DO imolecule = 1, SIZE(molecule_set)
            molecule_set(imolecule)%first_atom = first_atom(imolecule)
         END DO
      END IF

      IF (PRESENT(last_atom)) THEN
         IF (SIZE(last_atom) /= SIZE(molecule_set)) THEN
            CALL cp_abort(__LOCATION__, &
                          "The sizes of last_atom and molecule_set "// &
                          "are different")
         END IF

         DO imolecule = 1, SIZE(molecule_set)
            molecule_set(imolecule)%last_atom = last_atom(imolecule)
         END DO
      END IF

   END SUBROUTINE set_molecule_set

! **************************************************************************************************
!> \brief   finds for each atom the molecule it belongs to
!> \param molecule_set ...
!> \param atom_to_mol ...
! **************************************************************************************************
   SUBROUTINE molecule_of_atom(molecule_set, atom_to_mol)
      TYPE(molecule_type), DIMENSION(:), INTENT(IN)      :: molecule_set
      INTEGER, DIMENSION(:), INTENT(OUT)                 :: atom_to_mol

      INTEGER                                            :: first_atom, iatom, imol, last_atom

      DO imol = 1, SIZE(molecule_set)
         CALL get_molecule(molecule=molecule_set(imol), first_atom=first_atom, last_atom=last_atom)
         DO iatom = first_atom, last_atom
            atom_to_mol(iatom) = imol
         END DO ! iatom
      END DO ! imol

   END SUBROUTINE molecule_of_atom

! **************************************************************************************************
!> \brief returns information about molecules in the set.
!> \param molecule_set ...
!> \param atom_to_mol ...
!> \param mol_to_first_atom ...
!> \param mol_to_last_atom ...
!> \param mol_to_nelectrons ...
!> \param mol_to_nbasis ...
!> \param mol_to_charge ...
!> \param mol_to_multiplicity ...
!> \par History
!>       2011.06 created [Rustam Z Khaliullin]
!> \author Rustam Z Khaliullin
! **************************************************************************************************
   SUBROUTINE get_molecule_set_info(molecule_set, atom_to_mol, mol_to_first_atom, &
                                    mol_to_last_atom, mol_to_nelectrons, mol_to_nbasis, mol_to_charge, &
                                    mol_to_multiplicity)

      TYPE(molecule_type), DIMENSION(:), INTENT(IN)      :: molecule_set
      INTEGER, DIMENSION(:), INTENT(OUT), OPTIONAL :: atom_to_mol, mol_to_first_atom, &
         mol_to_last_atom, mol_to_nelectrons, mol_to_nbasis, mol_to_charge, mol_to_multiplicity

      INTEGER                                            :: first_atom, iatom, imol, last_atom, &
                                                            nbasis, nelec
      REAL(KIND=dp)                                      :: charge
      TYPE(molecule_kind_type), POINTER                  :: imol_kind

      DO imol = 1, SIZE(molecule_set)

         CALL get_molecule(molecule=molecule_set(imol), molecule_kind=imol_kind, &
                           first_atom=first_atom, last_atom=last_atom)

         IF (PRESENT(mol_to_nelectrons)) THEN
            CALL get_molecule_kind(imol_kind, nelectron=nelec)
            mol_to_nelectrons(imol) = nelec
         END IF

         IF (PRESENT(mol_to_multiplicity)) THEN
            ! RZK-warning: At the moment we can only get the total number
            !  of electrons (alpha+beta) and we do not have a way to get the multiplicity of mols.
            !  Therefore, the best we can do is to assume the singlet state for even number of electrons
            !  and doublet state for odd number of electrons (assume ne_alpha > ne_beta).
            !  The best way to implement a correct multiplicity subroutine in the future is to get
            !  the number of alpha and beta e- for each atom from init_atom_electronic_state. This way (as opposed to
            !  reading the multiplicities from file) the number of occupied and virtual orbitals
            !  will be consistent with atomic guess. A guess with broken symmetry will be easy to
            !  implement as well.
            CALL get_molecule_kind(imol_kind, nelectron=nelec)
            IF (MOD(nelec, 2) .EQ. 0) THEN
               mol_to_multiplicity(imol) = 1
            ELSE
               mol_to_multiplicity(imol) = 2
            END IF
         END IF

         IF (PRESENT(mol_to_charge)) THEN
            CALL get_molecule_kind(imol_kind, charge=charge)
            mol_to_charge(imol) = NINT(charge)
         END IF

         IF (PRESENT(mol_to_nbasis)) THEN
            CALL get_molecule_kind(imol_kind, nsgf=nbasis)
            mol_to_nbasis(imol) = nbasis
         END IF

         IF (PRESENT(mol_to_first_atom)) THEN
            mol_to_first_atom(imol) = first_atom
         END IF

         IF (PRESENT(mol_to_last_atom)) THEN
            mol_to_last_atom(imol) = last_atom
         END IF

         IF (PRESENT(atom_to_mol)) THEN
            DO iatom = first_atom, last_atom
               atom_to_mol(iatom) = imol
            END DO ! iatom
         END IF

      END DO ! imol

   END SUBROUTINE get_molecule_set_info

END MODULE molecule_types
