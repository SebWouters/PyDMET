'''
    PyDMET: a python implementation of density matrix embedding theory
    Copyright (C) 2014 Sebastian Wouters
    
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
import SolveCorrelatedProblem
import SolveCorrelatedResponse
import MinimizeCostFunction
import numpy as np

class HubbardDMET:

    def __init__( self, lattice_size, cluster_size, HubbardU, antiPeriodic ):
        self.lattice_size = lattice_size
        self.cluster_size = cluster_size
        self.HubbardU     = HubbardU
        self.antiPeriodic = antiPeriodic
        self.impurityOrbs = self.ConstructImpurityOrbitals()
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
        
    def SolveGroundState( self, Nelectrons ):
    
        numImpOrbs  = np.sum( self.impurityOrbs )
        numBathOrbs = numImpOrbs
        assert( Nelectrons%2==0 )
        numPairs = Nelectrons / 2

        # Start with a diagonal embedding potential
        u_startguess = (1.0 * self.HubbardU * numPairs) / np.prod(self.lattice_size)
        print "DMET :: Starting guess for umat =",u_startguess,"* I"
        umat_new = u_startguess * np.identity( numImpOrbs, dtype=float )
        normOfDiff = 1.0
        threshold  = 1e-6 * numImpOrbs
        iteration  = 0

        while ( normOfDiff >= threshold ): 

            iteration += 1
            print "*** DMET iteration",iteration,"***"
            umat_old = np.array( umat_new, copy=True )

            # Augment the Hamiltonian with the embedding potential
            HamAugment = HamFull.HamFull(self.Ham, self.cluster_size, umat_new)

            # Get the RHF solution
            energiesRHF, solutionRHF = LinalgWrappers.SortedEigSymmetric( HamAugment.Tmat )
            if ( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] < 1e-8 ):
                print "ERROR: The single particle gap is zero!"
                assert( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] >= 1e-8 )

            # Get the RHF ground state 1RDM and construct the bath orbitals
            groundstate1RDM = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
            dmetOrbs, NelecEnvironment, DiscOccupation = DMETorbitals.ConstructBathOrbitals( self.impurityOrbs, groundstate1RDM, numBathOrbs )
            NelecActiveSpace = Nelectrons - NelecEnvironment # Floating point number
            assert( abs( NelecActiveSpace - 2*numImpOrbs ) < 1e-8 ) # For ground-state DMET
            NelecActiveSpace = int( round( NelecActiveSpace ) + 0.001 )

            # Construct the DMET Hamiltonian and get the exact solution
            HamDMET = DMETham.DMETham(self.Ham, HamAugment, dmetOrbs, self.impurityOrbs, numImpOrbs, numBathOrbs)
            EnergyPerSiteCorr, OneRDMcorr = SolveCorrelatedProblem.Solve( HamDMET, NelecActiveSpace )

            umat_new = MinimizeCostFunction.Minimize( umat_new, OneRDMcorr, HamDMET, NelecActiveSpace )
            normOfDiff = np.linalg.norm( umat_new - umat_old )
            print "   DMET :: The energy per site (correlated problem) =",EnergyPerSiteCorr
            print "   DMET :: The 2-norm of u_new - u_old =",normOfDiff
        
        print "DMET :: Convergence reached. Converged u-matrix:"
        print umat_new
        print "***************************************************"
        return EnergyPerSiteCorr
        
    def SolveResponse( self, Nelectrons, orbital_i, omega, eta, numBathOrbs, toSolve, prefactResponseRDM=0.5 ):
        
        numImpOrbs  = np.sum( self.impurityOrbs )
        assert( numBathOrbs >= numImpOrbs )
        assert( Nelectrons%2==0 )
        numPairs = Nelectrons / 2
        assert( (toSolve=='A') or (toSolve=='R') or (toSolve=='F') or (toSolve=='B') ) # LDOS addition/removal or LDDR forward/backward
        assert( (prefactResponseRDM>=0.0) and (prefactResponseRDM<=1.0) )

        # Start with a diagonal embedding potential
        u_startguess = (1.0 * self.HubbardU * numPairs) / np.prod(self.lattice_size)
        print "DMET :: Starting guess for umat =",u_startguess,"* I"
        umat_new = u_startguess * np.identity( numImpOrbs, dtype=float )
        normOfDiff = 1.0
        threshold  = 1e-6 * numImpOrbs
        iteration  = 0

        while ( normOfDiff >= threshold ):
        
            iteration += 1
            print "*** DMET iteration",iteration,"***"
            umat_old = np.array( umat_new, copy=True )

            # Augment the Hamiltonian with the embedding potential
            HamAugment = HamFull.HamFull(self.Ham, self.cluster_size, umat_new)

            # Get the RHF ground state solution
            energiesRHF, solutionRHF = LinalgWrappers.SortedEigSymmetric( HamAugment.Tmat )
            if ( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] < 1e-8 ):
                print "ERROR: The single particle gap is zero!"
                assert( energiesRHF[ numPairs ] - energiesRHF[ numPairs-1 ] >= 1e-8 )

            # Get the RHF ground state 1RDM and the mean-fieldish response 1RDM
            groundstate1RDM  = DMETorbitals.Construct1RDM_groundstate( solutionRHF, numPairs )
            if (toSolve=='A'):
                response1RDM = DMETorbitals.Construct1RDM_addition( orbital_i, omega, eta, energiesRHF, solutionRHF, numPairs )
            if (toSolve=='R'):
                response1RDM = DMETorbitals.Construct1RDM_removal(  orbital_i, omega, eta, energiesRHF, solutionRHF, numPairs )
            if (toSolve=='F'):
                response1RDM = DMETorbitals.Construct1RDM_forward(  orbital_i, omega, eta, energiesRHF, solutionRHF, numPairs )
            if (toSolve=='B'):
                response1RDM = DMETorbitals.Construct1RDM_backward( orbital_i, omega, eta, energiesRHF, solutionRHF, numPairs )

            # The response1RDM was calculated based on a normalized wavefunction: make a linco and construct the bath orbitals
            weighted1RDM = (1.0 - prefactResponseRDM) * groundstate1RDM + prefactResponseRDM * response1RDM
            dmetOrbs, NelecEnvironment, DiscOccupation = DMETorbitals.ConstructBathOrbitals( self.impurityOrbs, weighted1RDM, numBathOrbs )
            NelecActiveSpace = int( round( Nelectrons - NelecEnvironment ) + 0.001 ) # Now it should be of integer type
            print "   DMET :: Response : Number of electrons not in impurity or bath orbitals =",NelecEnvironment
            print "   DMET :: Response : The sum of discarded occupations = sum( min( NOON, 2-NOON ) , pure environment orbitals ) =", DiscOccupation

            # Construct the DMET Hamiltonian and get the exact solution
            HamDMET = DMETham.DMETham(self.Ham, HamAugment, dmetOrbs, self.impurityOrbs, numImpOrbs, numBathOrbs)
            GSenergyPerSite, GFvalue, GS_1RDM, RESP_1RDM = SolveCorrelatedResponse.Solve( HamDMET, NelecActiveSpace, orbital_i, omega, eta, toSolve )

            umat_new = MinimizeCostFunction.MinimizeResponse( umat_new, GS_1RDM, RESP_1RDM, HamDMET, NelecActiveSpace, orbital_i, omega, eta, toSolve, prefactResponseRDM )
            normOfDiff = np.linalg.norm( umat_new - umat_old )
            print "   DMET :: The energy per site (correlated problem) =",GSenergyPerSite
            print "   DMET :: The Green's function value (correlated problem) =",GFvalue
            print "   DMET :: The 2-norm of u_new - u_old =",normOfDiff
            print umat_new
        
        print "DMET :: Convergence reached. Converged u-matrix:"
        print umat_new
        print "***************************************************"
        return ( GSenergyPerSite , GFvalue )
        
        
