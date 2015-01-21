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

    def __init__( self, lattice_size, cluster_size, HubbardU, antiPeriodic, skew2by2cell=False ):
        self.lattice_size = lattice_size
        self.cluster_size = cluster_size
        self.HubbardU     = HubbardU
        self.antiPeriodic = antiPeriodic
        self.skew2by2cell = skew2by2cell
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
        
        if ( self.skew2by2cell ):
            impurityOrbs[0, 0] = 1
            impurityOrbs[1, 0] = 1
            impurityOrbs[1, 1] = 1
            impurityOrbs[2, 1] = 1
        else:
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

        while ( normOfDiff >= threshold ): 

            iteration += 1
            print "*** DMET iteration",iteration,"***"
            if ( numImpOrbs > 1 ) and ( iteration > numNonDIIS ):
                umat_new = theDIIS.Solve()
            umat_old = np.array( umat_new, copy=True )

            # Augment the Hamiltonian with the embedding potential
            HamAugment = HamFull.HamFull(self.Ham, self.cluster_size, umat_new, self.skew2by2cell)

            # Get the RHF ground state 1RDM and construct the bath orbitals
            energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamAugment.Tmat, numPairs, True )
            groundstate1RDM = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
            dmetOrbs, NelecEnvironment, DiscOccupation = DMETorbitals.ConstructBathOrbitals( self.impurityOrbs, groundstate1RDM, numBathOrbs )
            NelecActiveSpace = Nelectrons - NelecEnvironment # Floating point number
            assert( abs( NelecActiveSpace - 2*numImpOrbs ) < 1e-8 ) # For ground-state DMET
            NelecActiveSpace = int( round( NelecActiveSpace ) + 0.001 )

            # Construct the DMET Hamiltonian and get the exact solution
            HamDMET = DMETham.DMETham(self.Ham, HamAugment, dmetOrbs, self.impurityOrbs, numImpOrbs, numBathOrbs)
            EnergyPerSiteCorr, OneRDMcorr, GSenergyFCI, GSvectorFCI = SolveCorrelated.SolveGS( HamDMET, NelecActiveSpace )

            umat_new = MinimizeCostFunction.Minimize( umat_new, OneRDMcorr, HamDMET, NelecActiveSpace )
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
        
    def SolveResponse( self, umat_guess, Nelectrons, omega, eta, numBathOrbs, toSolve, prefactResponseRDM=0.5 ):
        
        # Define a few constants
        numImpOrbs = np.sum( self.impurityOrbs )
        assert( numBathOrbs >= numImpOrbs )
        assert( Nelectrons%2==0 )
        numPairs = Nelectrons / 2
        assert( (toSolve=='A') or (toSolve=='R') or (toSolve=='F') or (toSolve=='B') ) # LDOS addition/removal or LDDR forward/backward
        assert( (prefactResponseRDM>=0.0) and (prefactResponseRDM<=1.0) )
        
        # Set up a few parameters for the self-consistent response DMET
        umat_new    = np.array( umat_guess, copy=True )
        umat_old    = np.array( umat_new,   copy=True )
        normOfDiff  = 1.0
        normOfDiff2 = 1.0
        threshold   = 1e-6 * numImpOrbs
        maxiter     = 1000
        iteration   = 0
        theDIIS     = DIIS.DIIS(7)
        startedDIIS = False
        maxdelta    = 0.1
        maxdeltanow = maxdelta

        while ( normOfDiff >= threshold ) and ( iteration < maxiter ):
        
            iteration += 1
            print "*** DMET iteration",iteration,"***"
            if ( numImpOrbs > 1 ) and ( startedDIIS ):
                umat_new = theDIIS.Solve()
            umat_old2 = np.array( umat_old, copy=True )
            umat_old  = np.array( umat_new, copy=True )

            # Augment the Hamiltonian with the embedding potential
            HamAugment = HamFull.HamFull(self.Ham, self.cluster_size, umat_new, self.skew2by2cell)

            # Get the RHF ground-state 1-RDM
            energiesRHF, solutionRHF = LinalgWrappers.RestrictedHartreeFock( HamAugment.Tmat, numPairs, True )
            if ( iteration == 1 ):
                chemical_potential_mu = 0.5 * ( energiesRHF[ numPairs-1 ] + energiesRHF[ numPairs ] )
                if ( toSolve == 'A' ) or ( toSolve == 'R' ):
                    omegabis = omega + chemical_potential_mu # Shift omega with the chemical potential for the retarded Green's function
                else:
                    omegabis = omega
            groundstate1RDM = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
            
            # Get the RHF mean-field response 1-RDMs
            response1RDMs = []
            for orbital_i in self.impIndices:
                if (toSolve=='A'):
                    response1RDM = DMETorbitals.Construct1RDM_addition( orbital_i, omegabis, eta, energiesRHF, solutionRHF, groundstate1RDM, numPairs )
                if (toSolve=='R'):
                    response1RDM = DMETorbitals.Construct1RDM_removal(  orbital_i, omegabis, eta, energiesRHF, solutionRHF, groundstate1RDM, numPairs )
                if (toSolve=='F'):
                    response1RDM = DMETorbitals.Construct1RDM_forward(  orbital_i, omegabis, eta, energiesRHF, solutionRHF, groundstate1RDM, numPairs )
                if (toSolve=='B'):
                    response1RDM = DMETorbitals.Construct1RDM_backward( orbital_i, omegabis, eta, energiesRHF, solutionRHF, groundstate1RDM, numPairs )
                response1RDMs.append( response1RDM )

            HamDMETs = []
            for orbital_i in range(0, numImpOrbs):
                # The response1RDM was calculated based on a normalized wavefunction: make a weighted average
                weighted1RDM_i = ( 1.0 - prefactResponseRDM ) * groundstate1RDM + prefactResponseRDM * response1RDMs[ orbital_i ]
                # For each impurity site, there's a different set of DMET orbitals
                dmetOrbs_i, NelecEnvironment_i, DiscOccupation_i = DMETorbitals.ConstructBathOrbitals( self.impurityOrbs, weighted1RDM_i, numBathOrbs )
                NelecActiveSpaceGuess_i = int( round( Nelectrons - NelecEnvironment_i ) + 0.001 ) # Now it should be of integer type
                if ( orbital_i == 0 ):
                    NelecActiveSpace = NelecActiveSpaceGuess_i
                else:
                    assert( NelecActiveSpace == NelecActiveSpaceGuess_i )
                print "   DMET :: Response (impurity", orbital_i, ") : Number of electrons not in impurity or bath orbitals =", NelecEnvironment_i
                print "   DMET :: Response (impurity", orbital_i, ") : Sum( min( NOON, 2-NOON ) , pure environment orbitals ) =", DiscOccupation_i
                HamDMET_i = DMETham.DMETham( self.Ham, HamAugment, dmetOrbs_i, self.impurityOrbs, numImpOrbs, numBathOrbs )
                HamDMETs.append( HamDMET_i )

            # Get the exact solution for each of the impurity orbitals
            totalGFvalue = 0.0
            averageGSenergyPerSite = 0.0
            GS_1RDMs   = []
            RESP_1RDMs = []
            for orbital_i in range(0, numImpOrbs):
                GSenergyPerSite, GS_1RDM, GSenergyFCI, GSvectorFCI = SolveCorrelated.SolveGS( HamDMETs[ orbital_i ], NelecActiveSpace )
                GFvalue, RESP_1RDM = SolveCorrelated.SolveResponse( HamDMETs[ orbital_i ], NelecActiveSpace, orbital_i, omegabis, eta, toSolve, GSenergyFCI, GSvectorFCI )
                totalGFvalue += GFvalue
                averageGSenergyPerSite += GSenergyPerSite
                GS_1RDMs.append( GS_1RDM )
                RESP_1RDMs.append( RESP_1RDM )
            averageGSenergyPerSite = ( 1.0 * averageGSenergyPerSite ) / numImpOrbs
            if ( iteration==1 ):
                notSelfConsistentTotalGF = totalGFvalue

            if ( normOfDiff2 / normOfDiff < 1e-2 ): # Limit cycle with period 2
                maxdeltanow = 0.1 * np.pi * maxdelta
            umat_new = MinimizeCostFunction.MinimizeResponse( umat_new, GS_1RDMs, RESP_1RDMs, HamDMETs, NelecActiveSpace, omegabis, eta, toSolve, prefactResponseRDM, maxdeltanow )
            maxdeltanow = maxdelta
            normOfDiff  = np.linalg.norm( umat_new - umat_old  )
            normOfDiff2 = np.linalg.norm( umat_new - umat_old2 )
            
            if ( numImpOrbs > 1 ) and (( normOfDiff < 1e-3 ) or ( startedDIIS )):
                startedDIIS = True
                error = umat_new - umat_old
                error = np.reshape( error, error.shape[0]*error.shape[1] )
                theDIIS.append( error, umat_new )
            
            print "   DMET :: The average ground-state energy per site =",averageGSenergyPerSite
            print "   DMET :: The Green's function value (correlated problem) =",totalGFvalue
            print "   DMET :: The 2-norm of u_new - u_old =",normOfDiff
            print umat_new
        
        print "DMET :: Convergence reached. Converged u-matrix:"
        print umat_new
        print "***************************************************"
        return ( averageGSenergyPerSite, totalGFvalue, notSelfConsistentTotalGF )
        
        
