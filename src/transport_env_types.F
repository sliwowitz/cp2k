!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief CP2K transport environment and related C-interoperable types
!> \par History
!>       05.2013 created C-interoperable matrices [Hossein Bani-Hashemian]
!>       07.2013 created transport_env [Hossein Bani-Hashemian]
!>       11.2014 revised into CSR matrices [Hossein Bani-Hashemian]
!>       12.2014 merged csr_interop and transport [Hossein Bani-Hashemian]
!> \author Mohammad Hossein Bani-Hashemian
! **************************************************************************************************
MODULE transport_env_types

   USE ISO_C_BINDING,                   ONLY: &
        C_ASSOCIATED, C_BOOL, C_DOUBLE, C_FUNPTR, C_F_POINTER, C_INT, C_NULL_FUNPTR, C_NULL_PTR, &
        C_PTR
   USE cp_dbcsr_api,                    ONLY: dbcsr_csr_destroy,&
                                              dbcsr_csr_type,&
                                              dbcsr_deallocate_matrix,&
                                              dbcsr_release,&
                                              dbcsr_type
   USE kinds,                           ONLY: dp
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'transport_env_types'

   PUBLIC :: transport_env_type, cp2k_transport_parameters
   PUBLIC :: cp2k_csr_interop_type

! DO NOT change the ORDERING or the NAMES in the following data type
   TYPE, BIND(C) :: cp2k_transport_parameters
      INTEGER(C_INT)  :: n_occ = -1_C_INT
      INTEGER(C_INT)  :: n_atoms = -1_C_INT
      REAL(C_DOUBLE)  :: energy_diff = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: evoltfactor = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: e_charge = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: boltzmann = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: h_bar = -1.0_C_DOUBLE
      INTEGER(C_INT)  :: iscf = -1_C_INT
      INTEGER(C_INT)  :: method = -1_C_INT
      INTEGER(C_INT)  :: qt_formalism = -1_C_INT
      INTEGER(C_INT)  :: injection_method = -1_C_INT
      INTEGER(C_INT)  :: rlaxis_integration_method = -1_C_INT
      INTEGER(C_INT)  :: linear_solver = -1_C_INT
      INTEGER(C_INT)  :: matrixinv_method = -1_C_INT
      INTEGER(C_INT)  :: transport_neutral = -1_C_INT
      INTEGER(C_INT)  :: num_pole = -1_C_INT
      INTEGER(C_INT)  :: ordering = -1_C_INT
      INTEGER(C_INT)  :: row_ordering = -1_C_INT
      INTEGER(C_INT)  :: verbosity = -1_C_INT
      INTEGER(C_INT)  :: pexsi_np_symb_fact = -1_C_INT
      INTEGER(C_INT)  :: n_kpoint = -1_C_INT
      INTEGER(C_INT)  :: num_interval = -1_C_INT
      INTEGER(C_INT)  :: num_contacts = -1_C_INT
      INTEGER(C_INT)  :: stride_contacts = -1_C_INT
      INTEGER(C_INT)  :: tasks_per_energy_point = -1_C_INT
      INTEGER(C_INT)  :: tasks_per_pole = -1_C_INT
      INTEGER(C_INT)  :: gpus_per_point = -1_C_INT
      INTEGER(C_INT)  :: n_points_beyn = -1_C_INT
      INTEGER(C_INT)  :: ncrc_beyn = -1_C_INT
      INTEGER(C_INT)  :: tasks_per_integration_point = -1_C_INT
      INTEGER(C_INT)  :: n_points_inv = -1_C_INT
      INTEGER(C_INT)  :: cutout(2) = -1_C_INT
      REAL(C_DOUBLE)  :: colzero_threshold = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: eps_limit = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: eps_limit_cc = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: eps_decay = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: eps_singularity_curvatures = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: eps_mu = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: eps_eigval_degen = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: eps_fermi = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: energy_interval = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: min_interval = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: temperature = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: dens_mixing = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: n_rand_beyn = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: n_rand_cc_beyn = -1.0_C_DOUBLE
      REAL(C_DOUBLE)  :: svd_cutoff = -1.0_C_DOUBLE
      TYPE(C_PTR)     :: contacts_data = C_NULL_PTR
      TYPE(C_PTR)     :: nsgf = C_NULL_PTR
      TYPE(C_PTR)     :: zeff = C_NULL_PTR
      LOGICAL(C_BOOL) :: obc_equilibrium = .FALSE._C_BOOL
      LOGICAL(C_BOOL) :: extra_scf = .FALSE._C_BOOL
   END TYPE cp2k_transport_parameters

   TYPE transport_env_type
      TYPE(C_FUNPTR)                   :: ext_c_method_ptr = C_NULL_FUNPTR
      TYPE(cp2k_transport_parameters)  :: params = cp2k_transport_parameters()
      TYPE(dbcsr_type)              :: template_matrix_sym = dbcsr_type()
      TYPE(dbcsr_type)              :: template_matrix_nosym = dbcsr_type()
      TYPE(dbcsr_type)              :: csr_sparsity = dbcsr_type()
      TYPE(dbcsr_type), POINTER        :: dm_imag => NULL()
      TYPE(dbcsr_csr_type)                   :: s_matrix = dbcsr_csr_type()
      TYPE(dbcsr_csr_type)                   :: ks_matrix = dbcsr_csr_type()
      TYPE(dbcsr_csr_type)                   :: p_matrix = dbcsr_csr_type()
      TYPE(dbcsr_csr_type)                   :: imagp_matrix = dbcsr_csr_type()
      LOGICAL                          :: csr_screening = .FALSE.
      INTEGER, DIMENSION(:), POINTER   :: contacts_data => NULL()
      INTEGER, DIMENSION(:), POINTER   :: nsgf => NULL()
      REAL(dp), DIMENSION(:), POINTER  :: zeff => NULL()
   END TYPE transport_env_type

! DO NOT change the ORDERING or the NAMES in the following data type
   TYPE, BIND(C) :: cp2k_csr_interop_type
      INTEGER(C_INT) :: nrows_total = -1_C_INT
      INTEGER(C_INT) :: ncols_total = -1_C_INT
      INTEGER(C_INT) :: nze_total = -1_C_INT
      INTEGER(C_INT) :: nze_local = -1_C_INT
      INTEGER(C_INT) :: nrows_local = -1_C_INT
      INTEGER(C_INT) :: data_type = -1_C_INT
      INTEGER(C_INT) :: first_row = -1_C_INT
      TYPE(C_PTR)    :: rowptr_local = C_NULL_PTR
      TYPE(C_PTR)    :: colind_local = C_NULL_PTR
      TYPE(C_PTR)    :: nzerow_local = C_NULL_PTR
      TYPE(C_PTR)    :: nzvals_local = C_NULL_PTR
   END TYPE cp2k_csr_interop_type

   PUBLIC :: csr_interop_nullify, &
             csr_interop_matrix_get_info
   PUBLIC :: transport_env_release

CONTAINS

! **************************************************************************************************
!> \brief releases the transport_env
!> \param[inout] transport_env the transport_env to be released
!> \author Mohammad Hossein Bani-Hashemian
! **************************************************************************************************
   SUBROUTINE transport_env_release(transport_env)
      TYPE(transport_env_type), POINTER                  :: transport_env

      CHARACTER(len=*), PARAMETER :: routineN = 'transport_env_release'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      CPASSERT(ASSOCIATED(transport_env))

      IF (C_ASSOCIATED(transport_env%ext_c_method_ptr)) THEN
         CALL dbcsr_csr_destroy(transport_env%s_matrix)
         CALL dbcsr_csr_destroy(transport_env%ks_matrix)
         CALL dbcsr_csr_destroy(transport_env%p_matrix)
         CALL dbcsr_csr_destroy(transport_env%imagp_matrix)
         CALL dbcsr_release(transport_env%template_matrix_sym)
         CALL dbcsr_release(transport_env%template_matrix_nosym)
         CALL dbcsr_release(transport_env%csr_sparsity)
         CALL dbcsr_deallocate_matrix(transport_env%dm_imag)
      END IF

      transport_env%ext_c_method_ptr = C_NULL_FUNPTR

      IF (ASSOCIATED(transport_env%contacts_data)) DEALLOCATE (transport_env%contacts_data)
      IF (ASSOCIATED(transport_env%nsgf)) DEALLOCATE (transport_env%nsgf)
      IF (ASSOCIATED(transport_env%zeff)) DEALLOCATE (transport_env%zeff)

      DEALLOCATE (transport_env)

      CALL timestop(handle)

   END SUBROUTINE transport_env_release

! **************************************************************************************************
!> \brief nullifies (and zeroizes) a C-interoperable CSR matrix
!> \param[inout] csr_interop_mat the matrix to be nullified
!> \author Mohammad Hossein Bani-Hashemian
! **************************************************************************************************
   SUBROUTINE csr_interop_nullify(csr_interop_mat)

      TYPE(cp2k_csr_interop_type), INTENT(INOUT)         :: csr_interop_mat

      CHARACTER(LEN=*), PARAMETER :: routineN = 'csr_interop_nullify'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      csr_interop_mat%nrows_total = 0
      csr_interop_mat%ncols_total = 0
      csr_interop_mat%nze_total = 0
      csr_interop_mat%nze_local = 0
      csr_interop_mat%nrows_local = 0
      csr_interop_mat%data_type = 0
      csr_interop_mat%first_row = 0
      csr_interop_mat%rowptr_local = C_NULL_PTR
      csr_interop_mat%colind_local = C_NULL_PTR
      csr_interop_mat%nzerow_local = C_NULL_PTR
      csr_interop_mat%nzvals_local = C_NULL_PTR

      CALL timestop(handle)

   END SUBROUTINE csr_interop_nullify

! **************************************************************************************************
!> \brief gets the fields of a C-interoperable CSR matrix
!> \param[in] csr_interop_mat C-interoperable CSR matrix
!> \param[out] nrows_total     total number of rows
!> \param[out] ncols_total     total number of columns
!> \param[out] nze_local       number of local nonzero elements
!> \param[out] nze_total       total number of nonzero elements
!> \param[out] nrows_local     number of local rows
!> \param[out] data_type       data type
!> \param[out] first_row       index of the first row (C indexing)
!> \param[out] rowptr_local    row pointer (local - Fortran indexing)
!> \param[out] colind_local    column index (local - Fortran indexing)
!> \param[out] nzerow_local    number of nunzeros per row (index-i, local - Fortran indexing)
!> \param[out] nzvals_local    nonzero elements (local)
!> \author Mohammad Hossein Bani-Hashemian
! **************************************************************************************************
   SUBROUTINE csr_interop_matrix_get_info(csr_interop_mat, &
                                          nrows_total, ncols_total, nze_local, nze_total, nrows_local, data_type, &
                                          first_row, rowptr_local, colind_local, nzerow_local, nzvals_local)

      TYPE(cp2k_csr_interop_type), INTENT(IN)            :: csr_interop_mat
      INTEGER, INTENT(OUT), OPTIONAL                     :: nrows_total, ncols_total, nze_local, &
                                                            nze_total, nrows_local, data_type, &
                                                            first_row
      INTEGER, DIMENSION(:), INTENT(OUT), OPTIONAL, &
         POINTER                                         :: rowptr_local, colind_local, nzerow_local
      REAL(dp), DIMENSION(:), INTENT(OUT), OPTIONAL, &
         POINTER                                         :: nzvals_local

      CHARACTER(LEN=*), PARAMETER :: routineN = 'csr_interop_matrix_get_info'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      IF (PRESENT(nrows_total)) nrows_total = csr_interop_mat%nrows_total
      IF (PRESENT(ncols_total)) ncols_total = csr_interop_mat%ncols_total
      IF (PRESENT(nze_local)) nze_local = csr_interop_mat%nze_local
      IF (PRESENT(nze_total)) nze_total = csr_interop_mat%nze_total
      IF (PRESENT(nrows_local)) nrows_local = csr_interop_mat%nrows_local
      IF (PRESENT(data_type)) data_type = csr_interop_mat%data_type
      IF (PRESENT(first_row)) first_row = csr_interop_mat%first_row

      IF (PRESENT(rowptr_local)) CALL C_F_POINTER(csr_interop_mat%rowptr_local, rowptr_local, [nrows_local + 1])
      IF (PRESENT(colind_local)) CALL C_F_POINTER(csr_interop_mat%colind_local, colind_local, [nze_local])
      IF (PRESENT(nzerow_local)) CALL C_F_POINTER(csr_interop_mat%nzerow_local, nzerow_local, [nrows_local])
      IF (PRESENT(nzvals_local)) CALL C_F_POINTER(csr_interop_mat%nzvals_local, nzvals_local, [nze_local])

      CALL timestop(handle)

   END SUBROUTINE csr_interop_matrix_get_info

END MODULE transport_env_types

