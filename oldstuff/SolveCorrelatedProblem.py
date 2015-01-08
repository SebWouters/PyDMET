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

# This file solves the DMET impurity + bath ground state problem exactly

import numpy as np
import ctypes
import sys
import os
sys.path.append('/home/seba/CheMPS2/build/PyCheMPS2')
import PyCheMPS2

def Solve( HamDMET, NelecActiveSpace, printoutput=False ):

    # Set the seed of the random number generator and cout.precision
    Initializer = PyCheMPS2.PyInitialize()
    Initializer.Init()

    # Setting up the Hamiltonian
    L = HamDMET.getDmetNumOrbitals()
    Group = 0
    orbirreps = np.zeros([L], dtype=ctypes.c_int)
    HamCheMPS2 = PyCheMPS2.PyHamiltonian(L, Group, orbirreps)
    HamCheMPS2.setEconst( HamDMET.HamOrig.getEconst() )
    for cnt1 in range(0, L):
        for cnt2 in range(0, L):
            HamCheMPS2.setTmat(cnt1, cnt2, HamDMET.getTmatCorr(cnt1, cnt2))
            for cnt3 in range(0, L):
                for cnt4 in range(0, L):
                    HamCheMPS2.setVmat(cnt1, cnt2, cnt3, cnt4, HamDMET.getVmatCorr(cnt1,cnt2,cnt3,cnt4))
    
    if ( printoutput==False ):
        sys.stdout.flush()
        old_stdout = sys.stdout.fileno()
        new_stdout = os.dup(old_stdout)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, old_stdout)
        os.close(devnull)
    
    if ( L <= 10 ): # Do FCI
    
        assert( NelecActiveSpace % 2 == 0 )
        Nel_up       = NelecActiveSpace / 2
        Nel_down     = NelecActiveSpace / 2
        Irrep        = 0
        maxMemWorkMB = 100.0
        FCIverbose   = 2
        
        theFCI = PyCheMPS2.PyFCI( HamCheMPS2, Nel_up, Nel_down, Irrep, maxMemWorkMB, FCIverbose )
        GSvector = np.zeros( [ theFCI.getVecLength() ], dtype=ctypes.c_double )
        theFCI.FillRandom( theFCI.getVecLength() , GSvector )
        EnergyCheMPS2 = theFCI.GSDavidson( GSvector )
        SpinSquared = theFCI.CalcSpinSquared( GSvector )
        TwoRDM = np.zeros( [ L**4 ], dtype=ctypes.c_double )
        theFCI.Fill2RDM( GSvector, TwoRDM )
        TwoRDM = TwoRDM.reshape( [L, L, L, L], order='F' )
        
    else: # Do DMRG
    
        TwoS  = 0
        Irrep = 0
        SpinSquared = 0  # As CheMPS2 factorizes configuration state functions into matrix product states
        assert( NelecActiveSpace % 2 == 0 )
        Prob  = PyCheMPS2.PyProblem( HamCheMPS2, TwoS, NelecActiveSpace, Irrep )

        OptScheme = PyCheMPS2.PyConvergenceScheme(4) # 4 instructions
        #OptScheme.setInstruction(instruction, D, Econst, maxSweeps, noisePrefactor)
        OptScheme.setInstruction(0,  500, 1e-10,  3, 0.05)
        OptScheme.setInstruction(1, 1000, 1e-10,  3, 0.05)
        OptScheme.setInstruction(2, 2000, 1e-10,  3, 0.05)
        OptScheme.setInstruction(3, 2000, 1e-10, 10, 0.00) # Last instruction a few iterations without noise

        theDMRG = PyCheMPS2.PyDMRG( Prob, OptScheme )
        EnergyCheMPS2 = theDMRG.Solve()
        theDMRG.calc2DMandCorrelations()
        TwoRDM = np.zeros( [L, L, L, L], dtype=ctypes.c_double )
        for orb1 in range(0, L):
            for orb2 in range(0, L):
                for orb3 in range(0, L):
                    for orb4 in range(0, L):
                        TwoRDM[ orb1, orb2, orb3, orb4 ] = theDMRG.get2DMA( orb1, orb2, orb3, orb4 )

        # theDMRG.deleteStoredMPS()
        theDMRG.deleteStoredOperators()

    if ( printoutput==False ):
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)

    if ( abs(SpinSquared) > 1e-8 ):
        print "Exact solution :: WARNING : < S^2 > =", SpinSquared

    # Calculate the 1RDM from the 2RDM
    OneRDM = np.einsum( 'ikjk->ij', TwoRDM ) / ( NelecActiveSpace - 1 )
            
    # Calculate the exact energy of the correlated problem for debugging purposes
    if ( False ):
        EnergyDebug = HamDMET.HamOrig.getEconst()
        for orb1 in range(0, L):
            for orb2 in range(0, L):
                EnergyDebug += OneRDM[ orb1, orb2 ] * HamDMET.getTmatCorr( orb1, orb2 )
                for orb3 in range(0, L):
                    for orb4 in range(0, L):
                        EnergyDebug += 0.5 * TwoRDM[ orb1, orb2, orb3, orb4 ] * HamDMET.getVmatCorr( orb1, orb2, orb3, orb4 )
        print "abs( EnergyDebug - EnergyCheMPS2 ) =", abs( EnergyDebug - EnergyCheMPS2 )

    # Calculate the energy per site based on the 1RDM and 2RDM
    EnergyPerSite = HamDMET.HamOrig.getEconst()
    for orb1 in range(0, HamDMET.numImpOrbs):
        for orb2 in range(0, L):
            EnergyPerSite += OneRDM[ orb1, orb2 ] * HamDMET.getTmatCorr( orb1, orb2 )
            for orb3 in range(0, L):
                for orb4 in range(0, L):
                    EnergyPerSite += 0.5 * TwoRDM[ orb1, orb2, orb3, orb4 ] * HamDMET.getVmatCorr( orb1, orb2, orb3, orb4 )
    EnergyPerSite = EnergyPerSite / HamDMET.numImpOrbs
    
    return ( EnergyPerSite, OneRDM )

    
