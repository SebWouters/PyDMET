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
import DMETresponseOrbs
import SolveCorrelatedProblem
import MinimizeCostFunction
import numpy as np

class HubbardResponseDMET:

    def __init__( self, lattice_size, cluster_size, HubbardU, Nelectrons, antiPeriodic=True ):
        self.lattice_size = lattice_size
        self.cluster_size = cluster_size
        self.HubbardU     = HubbardU
        self.Nelectrons   = Nelectrons
        self.antiPeriodic = antiPeriodic
        assert( self.Nelectrons % 2 == 0 )
        
    def Solve( self ):
    
        Ham = HamInterface.HamInterface(self.lattice_size, self.HubbardU, self.antiPeriodic)

        # Set the impurity orbitals lattice-style, e.g. impurityOrbs[row, col] = 1 for 2D lattice
        impurityOrbs = np.zeros(self.lattice_size, dtype=int)
        for count in range(0, np.prod( self.cluster_size )):
            copycount = count
            co = np.zeros([ Ham.dim ], dtype=int)
            for dim in range(0, Ham.dim):
                co[ dim ] = copycount % self.cluster_size[ dim ]
                copycount = ( copycount - co[ dim ] ) / self.cluster_size[ dim ]
            impurityOrbs[ tuple(co) ] = 1
        impurityOrbs = np.reshape( impurityOrbs, ( np.prod( self.lattice_size ) ), order='F' ) # HamFull assumes fortran
        numImpOrbs  = np.sum( impurityOrbs )
        numBathOrbs = numImpOrbs

        # Start with a diagonal embedding potential
        u_startguess = (0.5 * self.HubbardU * self.Nelectrons) / np.prod(self.lattice_size)
        print "DMET :: Starting guess for umat =",u_startguess,"* I"
        umat_new = u_startguess * np.identity( numImpOrbs, dtype=float )
        normOfDiff = 1.0
        threshold  = 1e-6 * numImpOrbs

        while ( normOfDiff >= threshold ): 

            umat_old = np.array( umat_new, copy=True )

            # Augment the Hamiltonian with the embedding potential
            HamAugment = HamFull.HamFull(Ham, self.cluster_size, umat_new)

            # Get the RHF solution
            eigenvals, solutionRHF = LinalgWrappers.SortedEigSymmetric( HamAugment.Tmat )
            numPairs = self.Nelectrons / 2
            if ( eigenvals[ numPairs ] - eigenvals[ numPairs-1 ] < 1e-8 ):
                print "ERROR: The single particle gap is zero!"
                assert( eigenvals[ numPairs ] - eigenvals[ numPairs-1 ] >= 1e-8 )

            # Get the RHF ground state 1RDM for the embedding orbitals and construct the bath orbitals
            groundstate1RDM = DMETorbitals.Construct1RDM_groundstate( solutionRHF, self.Nelectrons/2 )
            dmetOrbs = DMETorbitals.ConstructBathOrbitals( impurityOrbs, groundstate1RDM, numBathOrbs )
            
            orbital_i = 0
            omega = 1.23
            eta = 0.05
            add1RDM, addOverlap = DMETorbitals.Construct1RDM_addition( orbital_i, omega, eta, eigenvals, solutionRHF, self.Nelectrons/2 )
            rem1RDM, remOverlap = DMETorbitals.Construct1RDM_removal( orbital_i, omega, eta, eigenvals, solutionRHF, self.Nelectrons/2 )
            fow1RDM, fowOverlap = DMETorbitals.Construct1RDM_forward( orbital_i, omega, eta, eigenvals, solutionRHF, self.Nelectrons/2 )
            print "addOverlap =", addOverlap
            print "remOverlap =", remOverlap
            print "fowOverlap =", fowOverlap
            exit()

            # Construct the DMET Hamiltonian and get the exact solution
            HamDMET = DMETham.DMETham(Ham, HamAugment, dmetOrbs, impurityOrbs, numImpOrbs, numBathOrbs)
            EnergyPerSiteCorr, OneRDMcorr = SolveCorrelatedProblem.Solve( HamDMET, 2*numImpOrbs ) # Number of active space electrons is equal to 2*numImpOrbs

            umat_new = MinimizeCostFunction.Minimize( umat_new, OneRDMcorr, HamDMET, 2*numImpOrbs ) # Number of active space electrons is equal to 2*numImpOrbs
            normOfDiff = np.linalg.norm( umat_new - umat_old )
            print "DMET :: The energy per site (correlated problem) =",EnergyPerSiteCorr
            print "DMET :: The 2-norm of u_new - u_old =",normOfDiff
        
        print "DMET :: Convergence reached. Converged u-matrix:"
        print umat_new
        print "***************************************************"
        return EnergyPerSiteCorr
        
        
