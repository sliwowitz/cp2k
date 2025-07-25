!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \par History
!>      none
!> \author APSI & CJM & JGH
! **************************************************************************************************
MODULE header
   USE cp2k_info,                       ONLY: r_cwd,&
                                              r_datx,&
                                              r_host_name,&
                                              r_pid,&
                                              r_user_name
   USE kinds,                           ONLY: default_path_length
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE
   PUBLIC :: cp2k_header, cp2k_footer, vib_header
   PUBLIC :: fist_header, se_header, dftb_header, qs_header, tddfpt_header, rixs_header
   PUBLIC :: qmmm_header, atom_header, atom_footer, band_header, xtb_header, tblite_header
   PUBLIC :: tmc_header, tmc_ana_header
   PUBLIC :: sirius_header
   PUBLIC :: tddfpt_soc_header

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param iw ...
!> \param wdir ...
! **************************************************************************************************
   SUBROUTINE cp2k_header(iw, wdir)
      INTEGER, INTENT(IN)                                :: iw
      CHARACTER(LEN=default_path_length), OPTIONAL       :: wdir

      CHARACTER(LEN=default_path_length)                 :: cwd
      INTEGER                                            :: l

      IF (iw < 0) RETURN
      cwd = r_cwd
      IF (PRESENT(wdir)) cwd = wdir
      WRITE (UNIT=iw, &
             FMT="(/,A,T55,A26,/,A,T51,A30,/,A,T51,A30,/,A,T71,I10)") &
         "  **** **** ******  **  PROGRAM STARTED AT ", ADJUSTR(r_datx), &
         " ***** ** ***  *** **   PROGRAM STARTED ON ", ADJUSTR(r_host_name(1:30)), &
         " **    ****   ******    PROGRAM STARTED BY ", ADJUSTR(r_user_name(1:30)), &
         " ***** **    ** ** **   PROGRAM PROCESS ID ", r_pid
      WRITE (UNIT=iw, FMT="(A,T44,A37)") &
         "  **** **  *******  **  PROGRAM STARTED IN ", ADJUSTR(cwd(1:37))
      DO l = 38, LEN_TRIM(cwd), 37
         WRITE (UNIT=iw, FMT="(T44,A)") cwd(l:MIN(LEN_TRIM(cwd), l + 36))
      END DO

   END SUBROUTINE cp2k_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
!> \param wdir ...
! **************************************************************************************************
   SUBROUTINE cp2k_footer(iw, wdir)
      INTEGER, INTENT(IN)                                :: iw
      CHARACTER(LEN=default_path_length), OPTIONAL       :: wdir

      CHARACTER(LEN=default_path_length)                 :: cwd
      INTEGER                                            :: l

      IF (iw < 0) RETURN
      cwd = r_cwd
      IF (PRESENT(wdir)) cwd = wdir
      WRITE (UNIT=iw, &
             FMT="(A,T55,A26,/,A,T51,A30,/,A,T51,A30,/,A,T71,I10)") &
         "  **** **** ******  **  PROGRAM ENDED AT   ", ADJUSTR(r_datx), &
         " ***** ** ***  *** **   PROGRAM RAN ON     ", ADJUSTR(r_host_name(1:30)), &
         " **    ****   ******    PROGRAM RAN BY     ", ADJUSTR(r_user_name(1:30)), &
         " ***** **    ** ** **   PROGRAM PROCESS ID ", r_pid
      WRITE (UNIT=iw, FMT="(A,T44,A37)") &
         "  **** **  *******  **  PROGRAM STOPPED IN ", ADJUSTR(cwd(1:37))
      DO l = 38, LEN_TRIM(cwd), 37
         WRITE (UNIT=iw, FMT="(T44,A)") cwd(l:MIN(LEN_TRIM(cwd), l + 36))
      END DO
   END SUBROUTINE cp2k_footer

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE qmmm_header(iw)
      INTEGER, INTENT(in)                                :: iw

      IF (iw < 0) RETURN
      WRITE (iw, '(/,(T2,A79))') &
         "*******************************************************************************", &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**             #######  ##     ## ##     ## ##     ##                        **", &
         "**            ##     ## ###   ### ###   ### ###   ###                        **", &
         "**            ##     ## #### #### #### #### #### ####                        **", &
         "**            ##     ## ## ### ## ## ### ## ## ### ##                        **", &
         "**            ##  ## ## ##     ## ##     ## ##     ##                        **", &
         "**            ##    ##  ##     ## ##     ## ##     ##  T.Laino and F.Mohamed **", &
         "**             ##### ## ##     ## ##     ## ##     ##  2005 - 2015           **", &
         "**                                                                           **", &
         "**                                                     Calculation Started.. **", &
         "*******************************************************************************", &
         "*******************************************************************************"

   END SUBROUTINE qmmm_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE tddfpt_header(iw)
      INTEGER, INTENT(in)                                :: iw

      IF (iw < 0) RETURN
      WRITE (iw, '(/,(T2,A79))') &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**         ######## #######   #######   ######## #######  ########           **", &
         "**            ##    ##     ## ##     ## ##       ##   ##     ##              **", &
         "**            ##    ##     ## ##     ## ######   #######     ##              **", &
         "**            ##    ##     ## ##     ## ##       ##          ##              **", &
         "**            ##    #######   #######   ##       ##          ##              **", &
         "**                                                                           **", &
         "*******************************************************************************"

   END SUBROUTINE tddfpt_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE tddfpt_soc_header(iw)
      INTEGER, INTENT(in)                                :: iw

      IF (iw < 0) RETURN
      WRITE (iw, '(/,(T2,A79))') &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**                            ####### ####### #######                        **", &
         "**                        #   ##      ##   ## ##                             **", &
         "**                       ###  ######  ##   ## ##                             **", &
         "**                        #       ##  ##   ## ##                             **", &
         "**                            ######  ####### #######                        **", &
         "**                 Based on the xas module from Augustin Bussy               **", &
         "*******************************************************************************"

   END SUBROUTINE tddfpt_soc_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE fist_header(iw)
      INTEGER, INTENT(in)                                :: iw

      IF (iw > 0) THEN
         WRITE (iw, '( / )')
         WRITE (iw, '( 14(20x,a,/) )') &
            '  ************* ******** ***********   ', &
            ' ************* **********************  ', &
            ' ****     **** ****      ***********   ', &
            ' ******** ****  *******     ****       ', &
            ' *******  ****   *******   ****        ', &
            ' ****     ****       ****  ****        ', &
            ' ****     **** **********  ****        ', &
            '  ****     **** ********    ****       ', &
            ' FRONTIERS IN SIMULATION TECHNOLOGY    ', &
            '                                       ', &
            '    C.J. Mundy, S. Balasubramanian,    ', &
            ' Ken Bagchi, J. Hutter, Ari Seitsonen  ', &
            '  IFW Kuo, T. Laino, J. VandeVondele   ', &
            '             Version 1.0               ', &
            '                                       '
      END IF

   END SUBROUTINE fist_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE dftb_header(iw)
      INTEGER, INTENT(in)                                :: iw

      IF (iw < 0) RETURN
      WRITE (iw, '( / )')
      WRITE (iw, '( 11(12x,a,/) )') &
         '  #####   #####        # ######  ####### ####### ######   ', &
         ' #     # #     #      #  #     # #          #    #     #  ', &
         ' #     # #           #   #     # #          #    #     #  ', &
         ' #     #  #####     #    #     # #####      #    ######   ', &
         ' #   # #       #   #     #     # #          #    #     #  ', &
         ' #    #  #     #  #      #     # #          #    #     #  ', &
         '  #### #  #####  #       ######  #          #    ######   ', &
         '                                                          ', &
         '                        Version 0.2                       ', &
         '                   J. Hutter and T. Heine                 ', &
         '                                                          '
   END SUBROUTINE dftb_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
!> \param gfn_type ...
! **************************************************************************************************
   SUBROUTINE xtb_header(iw, gfn_type)
      INTEGER, INTENT(in)                                :: iw, gfn_type

      IF (iw < 0) RETURN
      WRITE (iw, '( / )')
      WRITE (iw, '( 7(16x,a,/) )') &
         '  #####   #####        #          ####### ######   ', &
         ' #     # #     #      #              #    #     #  ', &
         ' #     # #           #    ##   ##    #    #     #  ', &
         ' #     #  #####     #      ## ##     #    ######   ', &
         ' #   # #       #   #        ###      #    #     #  ', &
         ' #    #  #     #  #        ## ##     #    #     #  ', &
         '  #### #  #####  #        ##   ##    #    ######   '

      SELECT CASE (gfn_type)
      CASE (0)
         WRITE (iw, '( 34x,a )') ' GFN0-xTB '
      CASE (1)
         WRITE (iw, '( 34x,a )') ' GFN1-xTB '
      CASE (2)
         WRITE (iw, '( 34x,a )') ' GFN2-xTB '
      CASE DEFAULT
      END SELECT

      WRITE (iw, '( 2(16x,a,/) )') &
         '                 Version 1.1                       ', &
         '            J. Hutter and A. Hehn                  '

   END SUBROUTINE xtb_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
!> \param tb_type ...
! **************************************************************************************************
   SUBROUTINE tblite_header(iw, tb_type)
      INTEGER, INTENT(in)                                :: iw, tb_type

      IF (iw < 0) RETURN
      WRITE (iw, '( / )')
      WRITE (iw, '( 7(16x,a,/) )') &
         '  #####   #####        #     #   #     #       #          ', &
         ' #     # #     #      #    ##### #     #   # #####  ####  ', &
         ' #     # #           #       #   ###   #       #   #    # ', &
         ' #     #  #####     #        #   #  #  #   #   #   ###### ', &
         ' #   # #       #   #         #   #   # #   #   #   #      ', &
         ' #    #  #     #  #          #   #  #  # # #   #    #   # ', &
         '  #### #  #####  #           #   ###   ##  #   #     ###  '

      SELECT CASE (tb_type)
      CASE (1)
         WRITE (iw, '( 34x,a )') ' GFN1-xTB  '
      CASE (2)
         WRITE (iw, '( 34x,a )') ' GFN2-xTB  '
      CASE (3)
         WRITE (iw, '( 34x,a )') ' IPEA1-xTB '
      CASE DEFAULT
      END SELECT

      WRITE (iw, '( 2(16x,a,/) )') &
         '                 Version 0.0                       ', &
         '            Johann V. Pototschnig                  '

   END SUBROUTINE tblite_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE se_header(iw)
      INTEGER, INTENT(in)                                :: iw

      IF (iw < 0) RETURN
      WRITE (iw, '( / )')
      WRITE (iw, '( 12(12x,a,/) )') &
         '  #####   #####        # #     # ######  ######   #####  ', &
         ' #     # #     #      #  ##    # #     # #     # #     # ', &
         ' #     # #           #   # #   # #     # #     # #     # ', &
         ' #     #  #####     #    #  #  # #     # #     # #     # ', &
         ' #   # #       #   #     #   # # #     # #     # #     # ', &
         ' #    #  #     #  #      #    ## #     # #     # #     # ', &
         '  #### #  #####  #       #     # ######  ######   #####  ', &
         '                                                         ', &
         '                  T. Laino and J. Hutter                 ', &
         '                                                         ', &
         '                       2008 - 2015                       '

   END SUBROUTINE se_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE qs_header(iw)
      INTEGER, INTENT(IN)                                :: iw

      IF (iw < 0) RETURN
      WRITE (UNIT=iw, FMT="(/,(T2,A79))") &
         "*******************************************************************************", &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**     #####                         ##              ##                      **", &
         "**    ##   ##            ##          ##              ##                      **", &
         "**   ##     ##                       ##            ######                    **", &
         "**   ##     ##  ##   ##  ##   #####  ##  ##   ####   ##    #####    #####    **", &
         "**   ##     ##  ##   ##  ##  ##      ## ##   ##      ##   ##   ##  ##   ##   **", &
         "**   ##  ## ##  ##   ##  ##  ##      ####     ###    ##   ######   ######    **", &
         "**    ##  ###   ##   ##  ##  ##      ## ##      ##   ##   ##       ##        **", &
         "**     #######   #####   ##   #####  ##  ##  ####    ##    #####   ##        **", &
         "**           ##                                                    ##        **", &
         "**                                                                           **", &
         "**                                                ... make the atoms dance   **", &
         "**                                                                           **", &
         "**            Copyright (C) by CP2K developers group (2000-2025)             **", &
         "**                      J. Chem. Phys. 152, 194103 (2020)                    **", &
         "**                                                                           **", &
         "*******************************************************************************"
   END SUBROUTINE qs_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE atom_header(iw)
      INTEGER, INTENT(IN)                                :: iw

      IF (iw < 0) RETURN
      WRITE (iw, '( / )')
      WRITE (iw, '( 14(25x,a,/) )') &
         '  ****  ******  ****   ****   ', &
         ' **  ** ****** **  ** ******  ', &
         ' ******   **   **  ** **  **  ', &
         ' **  **   **    ****  **  **  ', &
         '                              ', &
         '    University of Zurich      ', &
         '        2009 - 2015           ', &
         '                              ', &
         '        Version 0.0           ', &
         '                                          '
   END SUBROUTINE atom_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE atom_footer(iw)
      INTEGER, INTENT(IN)                                :: iw

      IF (iw < 0) RETURN
      WRITE (iw, '( / )')
      WRITE (iw, '( 14(25x,a,/) )') &
         '    NORMAL TERMINATION OF     ', &
         '                              ', &
         '  ****  ******  ****   ****   ', &
         ' **  ** ****** **  ** ******  ', &
         ' ******   **   **  ** **  **  ', &
         ' **  **   **    ****  **  **  '

   END SUBROUTINE atom_footer

! **************************************************************************************************
!> \brief ...
!> \param iw ...
!> \param ni ...
!> \param nr ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE band_header(iw, ni, nr, np)
      INTEGER, INTENT(IN)                                :: iw, ni, nr, np

      IF (iw < 0) RETURN
      WRITE (UNIT=iw, FMT="(/,(T2,A79))") &
         "*******************************************************************************", &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**  ########     ###    ##    ## ########    Nudged Elastic Band Method      **", &
         "**  ##     ##   ## ##   ###   ## ##     ##   String Method                   **", &
         "**  ##     ##  ##   ##  ####  ## ##     ##                                   **"
      WRITE (UNIT=iw, FMT="(T2,A,T73,I5,1X,A)") &
         "**  ########  ##     ## ## ## ## ##     ##   Number of Images :", ni, "**", &
         "**  ##     ## ######### ##  #### ##     ##   Number of Replicas:", nr, "**", &
         "**  ##     ## ##     ## ##   ### ##     ##   Number of Procs/Rep:", np, "**"
      WRITE (UNIT=iw, FMT="(T2,A79)") &
         "**  ########  ##     ## ##    ## ########                                    **", &
         "**                                           T. Laino  2009 - 2015           **", &
         "*******************************************************************************"
      WRITE (UNIT=iw, FMT="(T2,A79,/)") &
         "*******************************************************************************"

   END SUBROUTINE band_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
!> \param nr ...
!> \param np ...
! **************************************************************************************************
   SUBROUTINE vib_header(iw, nr, np)
      INTEGER, INTENT(IN)                                :: iw, nr, np

      IF (iw < 0) RETURN
      WRITE (UNIT=iw, FMT="(/,(T2,A79))") &
         "*******************************************************************************", &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**  #  #   #    #                  #     #                      ##           **", &
         "**  #  #        ###   # ##   ###  ####         ##   # #    ###   #           **", &
         "**  #  #   #    #  #  ##    #  #   #     #    #  #  ## #  #  #   #           **", &
         "**   ##    #    #  #  #     # ##   #     #    #  #  #  #  # ##   #           **", &
         "**   ##    #    ###   #      # #    ##   #     ##   #  #   # #  ###          **", &
         "**                                                                           **", &
         "**   ##               ##                 #                                   **", &
         "**  #  #  # #    ###   #    #  #   ###         ###                           **", &
         "**  #  #  ## #  #  #   #    #  #  ##     #    ##                             **"
      WRITE (UNIT=iw, FMT="(T2,A,T73,I5,1X,A)") &
         "**  ####  #  #  # ##   #     ###    ##   #      ##  N. Replicas:", nr, "**", &
         "**  #  #  #  #   # #  ###      #  ###    #    ###   N. Procs/Rep:", np, "**"
      WRITE (UNIT=iw, FMT="(T2,A79)") &
         "**                           ##                                              **", &
         "**                                       T. Laino  and F. Schiffmann         **", &
         "**                                       2008 - 2015                         **", &
         "*******************************************************************************"
      WRITE (UNIT=iw, FMT="(T2,A79,/)") &
         "*******************************************************************************"

   END SUBROUTINE vib_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE sirius_header(iw)
      INTEGER, INTENT(IN)                                :: iw

      IF (iw < 0) RETURN
      WRITE (UNIT=iw, FMT="(/,(T2,A79))") &
         "*******************************************************************************", &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**   #### ##   ## ####  ##### #####           #### ## ####  ## ## ##  ####   **", &
         "**   ## ####   ## ## ## ###   #####          ##    ## ##  # ## ## ## ##      **", &
         "**   #### ## # ## ## ## #####  ###           ##### ## ####  ## ## ## #####   **", &
         "**   ##   ####### ## ## ##     ###              ## ## ## #  ## #####    ##   **", &
         "**   ##   ## # ## ####  ##     ###           ####  ## ##  # ## ##### ####    **", &
         "**                                                                           **", &
         "**   SIRIUS: Domain specific library for electronic structure calculations   **", &
         "**                                                                           **", &
         "*******************************************************************************"
   END SUBROUTINE sirius_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE tmc_header(iw)
      INTEGER, INTENT(IN)                                :: iw

      IF (iw < 0) RETURN
      WRITE (UNIT=iw, FMT="(/,(T2,A79))") &
         "*******************************************************************************", &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**                      ###########  ###     ###    ######                   **", &
         "**                          ###      ####   ####   ###                       **", &
         "**                          ###      ### # # ###  ###                        **", &
         "**                          ###      ###  #  ###   ###                       **", &
         "**                          ###      ###     ###    ######                   **", &
         "**  TreeMonteCarlo                                        Mandes Schoenherr  **", &
         "**                                                        2009 - 2015        **", &
         "*******************************************************************************", &
         "*******************************************************************************"
   END SUBROUTINE tmc_header

! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE tmc_ana_header(iw)
      INTEGER, INTENT(IN)                                :: iw

      IF (iw < 0) RETURN
      WRITE (UNIT=iw, FMT="(/,(T2,A79))") &
         "*******************************************************************************", &
         "*******************************************************************************", &
         "**                                                                           **", &
         "**  ###########  ###     ###    ######              ##    ###   ##    ##     **", &
         "**      ###      ####   ####   ###                 ####   ####  ##   ####    **", &
         "**      ###      ### # # ###  ###                 ##  ##  ## ## ##  ##  ##   **", &
         "**      ###      ###  #  ###   ###                ######  ##  ####  ######   **", &
         "**      ###      ###     ###    ######            ##  ##  ##   ###  ##  ##   **", &
         "**  Trajectory analyzer for TreeMonteCarlo ensembles      Mandes Schoenherr  **", &
         "**                                                        2013 - 2015        **", &
         "*******************************************************************************", &
         "*******************************************************************************"
   END SUBROUTINE tmc_ana_header
! **************************************************************************************************
!> \brief ...
!> \param iw ...
! **************************************************************************************************
   SUBROUTINE rixs_header(iw)
      INTEGER, INTENT(IN)                                :: iw

      IF (iw < 0) RETURN
      WRITE (iw, "(/,(T2,A79))") &
         "*******************************************************************************", &
         "**                   ########   ####  ##     ##   ######                     **", &
         "**                   ##     ##   ##    ##   ##   ##                          **", &
         "**                   ########    ##      ###      ######                     **", &
         "**                   ##   ##     ##     ## ##          ##                    **", &
         "**                   ##     ##  ####  ##     ##   ######                     **", &
         "**              Resonant Inelastic X-RAY Scattering calculations             **", &
         "*******************************************************************************"

   END SUBROUTINE rixs_header
END MODULE header

