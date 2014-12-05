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

import numpy as np
import ctypes
import sys
import os
sys.path.append('/home/seba/CheMPS2/build/PyCheMPS2')
import PyCheMPS2

def Solve( HamDMET, NelecActiveSpace, orb_i, omega, eta, toSolve, printoutput=False ):

    # We should solve for one of these cases: LDOS addition; LDOS removal; LDDR forward; LDDR backward
    assert( (toSolve=='A') or (toSolve=='R') or (toSolve=='F') or (toSolve=='B') )

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
    
    # Killing output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        old_stdout = sys.stdout.fileno()
        new_stdout = os.dup(old_stdout)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, old_stdout)
        os.close(devnull)
    
    # FCI ground state calculation
    assert( NelecActiveSpace % 2 == 0 )
    Nel_up       = NelecActiveSpace / 2
    Nel_down     = NelecActiveSpace / 2
    Irrep        = 0
    maxMemWorkMB = 100.0
    FCIverbose   = 2
    theFCI = PyCheMPS2.PyFCI( HamCheMPS2, Nel_up, Nel_down, Irrep, maxMemWorkMB, FCIverbose )
    GSvector = np.zeros( [ theFCI.getVecLength() ], dtype=ctypes.c_double )
    theFCI.FillRandom( theFCI.getVecLength() , GSvector )
    GSenergy = theFCI.GSDavidson( GSvector )
    SpinSquared = theFCI.CalcSpinSquared( GSvector )
    GS_2RDM = np.zeros( [ L**4 ], dtype=ctypes.c_double )
    theFCI.Fill2RDM( GSvector, GS_2RDM )
    GS_2RDM = GS_2RDM.reshape( [L, L, L, L], order='F' )

    # FCI response calculation
    Re2RDMresponse = np.zeros( [ L**4 ], dtype=ctypes.c_double )
    Im2RDMresponse = np.zeros( [ L**4 ], dtype=ctypes.c_double )
    if (toSolve=='A'):
        ReGF, ImGF = theFCI.RetardedGF_addition( omega, eta, orb_i, orb_i, True, GSenergy, GSvector, HamCheMPS2, Re2RDMresponse, Im2RDMresponse )
    if (toSolve=='R'):
        ReGF, ImGF = theFCI.RetardedGF_removal(  omega, eta, orb_i, orb_i, True, GSenergy, GSvector, HamCheMPS2, Re2RDMresponse, Im2RDMresponse )
    if (toSolve=='F'):
        ReGF, ImGF = theFCI.DensityResponseGF_forward(  omega, eta, orb_i, orb_i, GSenergy, GSvector, Re2RDMresponse, Im2RDMresponse )
    if (toSolve=='B'):
        ReGF, ImGF = theFCI.DensityResponseGF_backward( omega, eta, orb_i, orb_i, GSenergy, GSvector, Re2RDMresponse, Im2RDMresponse )
    Re2RDMresponse = Re2RDMresponse.reshape( [L, L, L, L], order='F' )
    Im2RDMresponse = Im2RDMresponse.reshape( [L, L, L, L], order='F' )
    RESP_2RDM = Re2RDMresponse + Im2RDMresponse # Real( <Psi| Operator |Psi> ) = <Real(Psi)| Operator | Real(Psi)> + <Imag(Psi)| Operator | Imag(Psi)>
    GreenFunctionValue = ReGF + 1j * ImGF

    # Reviving output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)

    # Check if the ground state was a spin singlet
    if ( abs(SpinSquared) > 1e-8 ):
        print "Exact solution :: WARNING : < S^2 > =", SpinSquared

    # Calculate the 1RDM from the 2RDM
    GS_1RDM   = np.einsum( 'ikjk->ij', GS_2RDM ) / ( NelecActiveSpace - 1 )
    RESP_1RDM = np.einsum( 'ikjk->ij', RESP_2RDM )
    RespTrace = np.trace( RESP_1RDM )
    if (toSolve=='F') or (toSolve=='B'):
        Normalization = NelecActiveSpace / RespTrace
    if (toSolve=='A'):
        Normalization = ( NelecActiveSpace + 1 ) / RespTrace
    if (toSolve=='R'):
        Normalization = ( NelecActiveSpace - 1 ) / RespTrace
    RESP_1RDM = RESP_1RDM * Normalization # Now 1RDM for response as if calculated from normalized wave function
    
    # Calculate the energy per site based on the 1RDM and 2RDM
    GSenergyPerSite = HamDMET.HamOrig.getEconst()
    for orb1 in range(0, HamDMET.numImpOrbs):
        for orb2 in range(0, L):
            GSenergyPerSite += GS_1RDM[ orb1, orb2 ] * HamDMET.getTmatCorr( orb1, orb2 )
            for orb3 in range(0, L):
                for orb4 in range(0, L):
                    GSenergyPerSite += 0.5 * GS_2RDM[ orb1, orb2, orb3, orb4 ] * HamDMET.getVmatCorr( orb1, orb2, orb3, orb4 )
    GSenergyPerSite = GSenergyPerSite / HamDMET.numImpOrbs
    
    return ( GSenergyPerSite, GreenFunctionValue, GS_1RDM, RESP_1RDM )

    
