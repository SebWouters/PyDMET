'''
    PyDMET: a python implementation of density matrix embedding theory
    Copyright (C) 2014, 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import HamInterface
import HamFull
import DMETham
import LinalgWrappers
import DMETorbitals
import SolveCorrelated
import MinimizeCostFunction
import DIIS
import numpy as np

class HubbardDMET:

    def __init__( self, lattice_size, cluster_size, HubbardU, antiPeriodic ):
   
        self.lattice_size = lattice_size
        self.cluster_size = cluster_size
        self.HubbardU     = HubbardU
        self.antiPeriodic = antiPeriodic
        self.impurityOrbs = self.ConstructImpurityOrbitals()
        self.impIndices   = []
        for count in range(0, len(self.impurityOrbs)):
            if ( self.impurityOrbs[count] == 1 ):
                self.impIndices.append( count )
        self.impIndices   = np.array( self.impIndices )
        self.Ham          = HamInterface.HamInterface(self.lattice_size, self.HubbardU, self.antiPeriodic)

    def ConstructImpurityOrbitals( self ):

        # Set the impurity orbitals lattice-style, e.g. impurityOrbs[row, col] = 1 for 2D lattice
        impurityOrbs = np.zeros( self.lattice_size, dtype=int )
        for count in range(0, np.prod( self.cluster_size )):
            copycount = count
            co = np.zeros([ len(self.cluster_size) ], dtype=int)
            for dim in range(0, len(self.cluster_size )):
                co[ dim ] = copycount % self.cluster_size[ dim ]
                copycount = ( copycount - co[ dim ] ) / self.cluster_size[ dim ]
            impurityOrbs[ tuple(co) ] = 1
        impurityOrbs = np.reshape( impurityOrbs, ( np.prod( self.lattice_size ) ), order='F' ) # HamFull assumes fortran
        return impurityOrbs

    def SolveGroundState( self, Nelectrons, umat_guess=None ):
    
        numImpOrbs  = np.sum( self.impurityOrbs )
        numBathOrbs = numImpOrbs
        assert( Nelectrons % 2 == 0 )
        numPairs = Nelectrons / 2

        if ( umat_guess == None ):
            # Start with a diagonal embedding potential
            u_startguess = (1.0 * self.HubbardU * numPairs) / np.prod(self.lattice_size)
            umat_new = u_startguess * np.identity( numImpOrbs, dtype=float )
        else:
            umat_new = np.array( umat_guess, copy=True )
        print "DMET :: Starting guess for umat ="
        print umat_new
        normOfDiff = 1.0
        threshold  = 1e-6 * numImpOrbs
        iteration  = 0
        theDIIS    = DIIS.DIIS(7)
        numNonDIIS = 4

        while ( normOfDiff > threshold ):

            iteration += 1
            print "*** DMET iteration",iteration,"***"
            if ( numImpOrbs > 1 ) and ( iteration > numNonDIIS ):
                umat_new = theDIIS.Solve()
            umat_old = np.array( umat_new, copy=True )

            # Augment the Hamiltonian with the embedding potential
            HamAugment = HamFull.HamFull(self.Ham, self.cluster_size, umat_new, umat_new)

            # Construct the bath orbitals
            energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamAugment.Tmat, numPairs, True )
            gs1RDM = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
            dmetOrbs, NelecEnvironment, DiscOccupation = DMETorbitals.ConstructBathOrbitals( self.impurityOrbs, gs1RDM, numImpOrbs )
            assert( abs( 2*numPairs - NelecEnvironment - 2*numImpOrbs ) < 1e-8 ) # For ground-state DMET
            NelecActiveSpace = 2 * numImpOrbs

            # Construct the DMET Hamiltonian and get the exact solution
            HamDMET = DMETham.DMETham(self.Ham, HamAugment, dmetOrbs, self.impurityOrbs, numImpOrbs, numBathOrbs)
            EnergyPerSiteCorr, OneRDMcorr, GSenergyFCI, GSvectorFCI = SolveCorrelated.SolveGS( HamDMET, NelecActiveSpace )

            Corr1RDMs = []
            HamDMETs  = []
            Corr1RDMs.append( OneRDMcorr )
            HamDMETs.append( HamDMET )
            umat_new = MinimizeCostFunction.MinimizeGS( umat_new, Corr1RDMs, HamDMETs, 1, NelecActiveSpace )
            normOfDiff = np.linalg.norm( umat_new - umat_old )
            
            if ( numImpOrbs > 1 ) and ( iteration >= numNonDIIS ):
                error = umat_new - umat_old
                error = np.reshape( error, error.shape[0]*error.shape[1] )
                theDIIS.append( error, umat_new )
            
            print "   DMET :: The energy per site (correlated problem) =",EnergyPerSiteCorr
            print "   DMET :: The 2-norm of u_new - u_old =",normOfDiff
        
        print "DMET :: Convergence reached. Converged u-matrix:"
        print umat_new
        print "***************************************************"
        return ( EnergyPerSiteCorr , umat_new )
        
    def SolveResponse( self, umat_guess, Nelectrons, omega, eta, numBathOrbs, toSolve ):
    
        # It doesn't make sense to match unnormalized RDMs.
        normalizedRDMs = True

        # Errortype 1 means sum( norm ( RDM difference ) )
        # Errortype 2 means norm( sum ( RDM difference ) )
        errorType = 1
        
        # If False, the real part of the RDM of |Psi^1> and |phi^1> is matched.
        # If True, the RDMs of the 3 separate kets a^{(dagger)} |Psi^0> and Re{ |Psi^1> } and Im{ |Psi^1> } are matched.
        doDDMRGstates = False
        
        # Define a few constants
        numImpOrbs = np.sum( self.impurityOrbs )
        assert( numBathOrbs >= numImpOrbs )
        assert( Nelectrons%2==0 )
        numPairs = Nelectrons / 2
        assert( (toSolve=='A') or (toSolve=='R') or (toSolve=='F') or (toSolve=='B') ) # LDOS addition/removal or LDDR forward/backward
        
        # Set up a few parameters for the self-consistent response DMET
        umat_new_GS   = np.array( umat_guess, copy=True )
        umats_new_RESP = []
        for cnt in range( numImpOrbs ):
            umats_new_RESP.append( np.array( umat_guess, copy=True ) )
        isConverged   = False
        threshold_GS  = 1e-6 * numImpOrbs
        threshold_RE  = 1e-5 * numImpOrbs
        maxiter       = 1000
        iteration     = 0
        #theDIISs = []
        #for cnt in range( numImpOrbs ):
        #    theDIISs.append( DIIS.DIIS(7) )

        while (( not isConverged ) and ( iteration < maxiter )):
        
            iteration += 1
            print "*** DMET iteration",iteration,"***"
            #if ( iteration > 1 ):
            #    for cnt in range( numImpOrbs ):
            #        umats_new_RESP[cnt] = theDIISs[cnt].Solve()
            umat_old_GS = np.array( umat_new_GS, copy=True )
            umats_old_RESP = []
            for cnt in range( numImpOrbs ):
                umats_old_RESP.append( np.array( umats_new_RESP[cnt], copy=True ) )

            # Augment the Hamiltonian with the embedding potential
            HamAugments = []
            for cnt in range( numImpOrbs ):
                HamAugments.append( HamFull.HamFull(self.Ham, self.cluster_size, umat_new_GS, umats_new_RESP[cnt]) )

            # Get the RHF ground-state 1-RDM
            energiesRHF = []
            solutionsRHF = []
            groundstate1RDMs = []
            for cnt in range( numImpOrbs ):
                en_RHF, sol_RHF = LinalgWrappers.RestrictedHartreeFock( HamAugments[cnt].Tmat, numPairs, True )
                if (( iteration == 1 ) and ( cnt == 0 )): #For the initial loop all u-matrices are equal to the initial guess ground-state one
                    chemical_potential_mu = 0.5 * ( en_RHF[ numPairs-1 ] + en_RHF[ numPairs ] )
                    if ( toSolve == 'A' ) or ( toSolve == 'R' ):
                        omegabis = omega + chemical_potential_mu # Shift omega with the chemical potential for the retarded Green's function
                    else:
                        omegabis = omega
                groundstate1RDMs.append( DMETorbitals.Construct1RDM_groundstate( sol_RHF, numPairs ) )
                solutionsRHF.append( sol_RHF )
                energiesRHF.append( en_RHF )
            
            # Get the RHF mean-field response 1-RDMs
            # For each impurity site, there's a different set of DMET orbitals and HamDMET
            HamDMETs = []
            for cnt in range( numImpOrbs ):
            
                orb_i = self.impIndices[cnt]
                 
                # Obtain the mean-fieldish response in the total orbital space: normalized RDMs ( < wfn | a^+ a | wfn > / < wfn | wfn > ) and their normalization ( < wfn | wfn > ) are returned
                if (toSolve=='A'):
                    RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_addition( orb_i, omegabis, eta, energiesRHF[cnt], solutionsRHF[cnt], groundstate1RDMs[cnt], numPairs )
                if (toSolve=='R'):
                    RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_removal(  orb_i, omegabis, eta, energiesRHF[cnt], solutionsRHF[cnt], groundstate1RDMs[cnt], numPairs )
                if (toSolve=='F'):
                    RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_forward(  orb_i, omegabis, eta, energiesRHF[cnt], solutionsRHF[cnt], groundstate1RDMs[cnt], numPairs )
                if (toSolve=='B'):
                    RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = DMETorbitals.Construct1RDM_backward( orb_i, omegabis, eta, energiesRHF[cnt], solutionsRHF[cnt], groundstate1RDMs[cnt], numPairs )
                
                # Make the desired average of RDMs to calculate the bath orbitals
                if ( doDDMRGstates == True ):
                    if ( normalizedRDMs == True ):
                        weighted1RDM_i = ( groundstate1RDMs[cnt] + RDM_A + RDM_R + RDM_I ) * 0.25
                    else:
                        weighted1RDM_i = ( groundstate1RDMs[cnt] + S_A * RDM_A + S_R * RDM_R + S_I * RDM_I ) / ( 1.0 + S_A + S_R + S_I )
                else:
                    if ( normalizedRDMs == True ):
                        weighted1RDM_i = ( groundstate1RDMs[cnt] + ( ( S_R * RDM_R + S_I * RDM_I ) / ( S_R + S_I ) ) ) * 0.5
                    else:
                        weighted1RDM_i = ( groundstate1RDMs[cnt] + S_R * RDM_R + S_I * RDM_I ) / ( 1.0 + S_R + S_I )
                
                # Construct the bath orbitals
                dmetOrbs_i, NelecEnvironment_i, DiscOccupation_i = DMETorbitals.ConstructBathOrbitals( self.impurityOrbs, weighted1RDM_i, numBathOrbs )
                NelecActiveSpaceGuess_i = int( round( Nelectrons - NelecEnvironment_i ) + 0.001 ) # Now it should be of integer type
                if ( orb_i == 0 ):
                    NelecActiveSpace = NelecActiveSpaceGuess_i
                else:
                    assert( NelecActiveSpace == NelecActiveSpaceGuess_i )
                print "   DMET :: Response (impurity", orb_i, ") : Sum( min( NOON, 2-NOON ) , pure environment orbitals ) =", DiscOccupation_i
                HamDMET_i = DMETham.DMETham( self.Ham, HamAugments[cnt], dmetOrbs_i, self.impurityOrbs, numImpOrbs, numBathOrbs )
                HamDMETs.append( HamDMET_i )
                
                # Sanity check
                if ( False ):
                    num_pairs_imp = NelecActiveSpace/2
                    num_AS_orbs = numImpOrbs + numBathOrbs
                    en_imp, orbs_imp = LinalgWrappers.RestrictedHartreeFock( HamDMET_i.Tmat, num_pairs_imp, True )
                    gs1RDM_imp = DMETorbitals.Construct1RDM_groundstate( orbs_imp, num_pairs_imp )
                    if (toSolve=='A'):
                        RDM_A_imp, RDM_R_imp, RDM_I_imp, x, y, z = DMETorbitals.Construct1RDM_addition( orb_i, omegabis, eta, en_imp, orbs_imp, gs1RDM_imp, num_pairs_imp )
                    if (toSolve=='R'):
                        RDM_A_imp, RDM_R_imp, RDM_I_imp, x, y, z = DMETorbitals.Construct1RDM_removal(  orb_i, omegabis, eta, en_imp, orbs_imp, gs1RDM_imp, num_pairs_imp )
                    if (toSolve=='F'):
                        RDM_A_imp, RDM_R_imp, RDM_I_imp, x, y, z = DMETorbitals.Construct1RDM_forward(  orb_i, omegabis, eta, en_imp, orbs_imp, gs1RDM_imp, num_pairs_imp )
                    if (toSolve=='B'):
                        RDM_A_imp, RDM_R_imp, RDM_I_imp, x, y, z = DMETorbitals.Construct1RDM_backward( orb_i, omegabis, eta, en_imp, orbs_imp, gs1RDM_imp, num_pairs_imp )
                    print "   Projector error A =", np.linalg.norm( RDM_A_imp - np.dot( np.dot( dmetOrbs_i[ :, :num_AS_orbs ].T , RDM_A ), dmetOrbs_i[ :, :num_AS_orbs ] ) )
                    print "   Projector error R =", np.linalg.norm( RDM_R_imp - np.dot( np.dot( dmetOrbs_i[ :, :num_AS_orbs ].T , RDM_R ), dmetOrbs_i[ :, :num_AS_orbs ] ) )
                    print "   Projector error I =", np.linalg.norm( RDM_I_imp - np.dot( np.dot( dmetOrbs_i[ :, :num_AS_orbs ].T , RDM_I ), dmetOrbs_i[ :, :num_AS_orbs ] ) )
                
            print "   DMET :: Response : Number of electrons in the active space =", NelecActiveSpace

            # Get the exact solution for each of the impurity orbitals
            totalGFvalue = 0.0
            averageGSenergyPerSite = 0.0
            ED_RDM_0 = []
            ED_RDM_A = []
            ED_RDM_R = []
            ED_RDM_I = []
            ED_RDM_RESP = []
            for orb_i in range( numImpOrbs ):
                GSsiteE, GS1RDM, GSfciE, GSvector = SolveCorrelated.SolveGS( HamDMETs[ orb_i ], NelecActiveSpace )
                GFvalue, RDM_A, RDM_R, RDM_I, S_A, S_R, S_I = SolveCorrelated.SolveResponse( HamDMETs[ orb_i ], NelecActiveSpace, orb_i, omegabis, eta, toSolve, GSfciE, GSvector )
                totalGFvalue += GFvalue
                averageGSenergyPerSite += GSsiteE
                ED_RDM_0.append( GS1RDM )
                if ( doDDMRGstates == True ):
                    if ( normalizedRDMs == True ):
                        ED_RDM_A.append( RDM_A )
                        ED_RDM_R.append( RDM_R )
                        ED_RDM_I.append( RDM_I )
                    else:
                        ED_RDM_A.append( S_A * RDM_A )
                        ED_RDM_R.append( S_R * RDM_R )
                        ED_RDM_I.append( S_I * RDM_I )
                else:
                    if ( normalizedRDMs == True ):
                        ED_RDM_RESP.append( ( S_R * RDM_R + S_I * RDM_I ) / ( S_R + S_I ) )
                    else:
                        ED_RDM_RESP.append( S_R * RDM_R + S_I * RDM_I )
            averageGSenergyPerSite = ( 1.0 * averageGSenergyPerSite ) / numImpOrbs
            if ( iteration==1 ):
                notSelfConsistentTotalGF = totalGFvalue

            # Find the new u-matrices with the two-step algorithm
            umat_new_GS = MinimizeCostFunction.MinimizeGS( umat_new_GS, ED_RDM_0, HamDMETs, numImpOrbs, NelecActiveSpace )
            if ( doDDMRGstates == True ):
                for cnt in range( numImpOrbs ):
                    umats_new_RESP[cnt] = MinimizeCostFunction.RespDDMRG( umats_new_RESP[cnt], ED_RDM_A[cnt], ED_RDM_R[cnt], ED_RDM_I[cnt], HamDMETs[cnt], cnt, NelecActiveSpace, omegabis, eta, toSolve, normalizedRDMs, errorType )
            else:
                for cnt in range( numImpOrbs ):
                    umats_new_RESP[cnt] = MinimizeCostFunction.RespNORMAL( umats_new_RESP[cnt], ED_RDM_RESP[cnt], HamDMETs[cnt], cnt, NelecActiveSpace, omegabis, eta, toSolve, normalizedRDMs )
            isConverged = True
            normOfDiff_GS = np.linalg.norm( umat_new_GS - umat_old_GS )
            if ( normOfDiff_GS > threshold_GS ):
                isConverged = False
            normsOfDiff_RE = []
            for cnt in range( numImpOrbs ): 
                normsOfDiff_RE.append( np.linalg.norm( umats_new_RESP[cnt] - umats_old_RESP[cnt] ) )
                if ( normsOfDiff_RE[cnt] > threshold_RE ):
                    isConverged = False
            
            #for cnt in range( numImpOrbs ):
            #    error = umats_new_RESP[cnt] - umats_old_RESP[cnt]
            #    error = np.reshape( error, error.shape[0]*error.shape[1] )
            #    theDIISs[cnt].append( error, umats_new_RESP[cnt] )
            
            print "   DMET :: The average ground-state energy per site =",averageGSenergyPerSite
            print "   DMET :: The Green's function value (correlated problem) =",totalGFvalue
            print "   DMET :: The 2-norm of u_new_GS - u_old_GS =",normOfDiff_GS
            for cnt in range( numImpOrbs ):
                print "   DMET :: The 2-norm of u_new_RESP[",cnt,"] - u_old_RESP[",cnt,"] =",normsOfDiff_RE[cnt]
            print umat_new_GS
            for cnt in range( numImpOrbs ):
                print umats_new_RESP[cnt]
        
        print "DMET :: Convergence reached. Converged GS u-matrix:"
        print umat_new_GS
        print "Converged RESP u-matrices:"
        for cnt in range( numImpOrbs ):
            print umats_new_RESP[cnt]
        print "***************************************************"
        return ( averageGSenergyPerSite, totalGFvalue, notSelfConsistentTotalGF )

