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
import math as m
import ctypes
import sys
sys.path.append('/home/seba/CheMPS2/build/PyCheMPS2')
sys.path.append('/home/seba/CheMPS2/build/PyTests')
import PyCheMPS2
import ReadinHamiltonianPsi4

def FCIGroundState( Ham, Nel_up, Nel_down, Irrep, startRandom ):

    maxMemWorkMB = 10.0
    FCIverbose = 0
    theFCI = PyCheMPS2.PyFCI(Ham, Nel_up, Nel_down, Irrep, maxMemWorkMB, FCIverbose)
    GSvector = np.zeros([ theFCI.getVecLength() ], dtype=ctypes.c_double)
   
    if (startRandom):
        theFCI.FillRandom( theFCI.getVecLength() , GSvector )
    else:
        GSvector[ theFCI.LowestEnergyDeterminant() ] = 1.0
      
    EnergyFCI = theFCI.GSDavidson( GSvector )
    theFCI.CalcSpinSquared( GSvector )
    return ( EnergyFCI , GSvector, theFCI )



    
def DensityResponseGF( theFCI, GSenergy, GSvector, orb_alpha , orb_beta , omega, eta ):

    RePart , ImPart = theFCI.DensityResponseGF( omega, eta, orb_alpha, orb_beta, GSenergy, GSvector )
    return RePart + 1j * ImPart


Initializer = PyCheMPS2.PyInitialize()
Initializer.Init()

Ham = ReadinHamiltonianPsi4.Read('/home/seba/CheMPS2/tests/matrixelements/N2_N14_S0_d2h_I0.dat')


Nel_up = 7
Nel_down = 7
Irrep = 0
startRandom = False

GSenergy, GSvector, theFCI = FCIGroundState( Ham, Nel_up, Nel_down, Irrep, startRandom )
print "                 FCI energy =",GSenergy
print "The result should be close to -107.648250974014"

isUp = True
orb_alpha = 0
orb_beta  = 0
omega     = -15.3
eta       = 0.05
RePart , ImPart = theFCI.RetardedGF( omega, eta, orb_alpha, orb_beta, isUp, GSenergy, GSvector, Ham )
print "LDOS( omega =", omega, "; eta =", eta, "; alpha =", orb_alpha, "; beta =", orb_beta, ") =", - ImPart / m.pi
print "                              The result should be close to 5.15931306704457"

orb_alpha = 3
orb_beta  = 3
omega     = 0.7358
eta       = 0.001
RePart , ImPart = theFCI.DensityResponseGF( omega, eta, orb_alpha, orb_beta, GSenergy, GSvector )
print "LDDR( omega =", omega, "; eta =", eta, "; alpha =", orb_alpha, "; beta =", orb_beta, ") =", - ImPart / m.pi
print "                                The result should be close to 1.58755549193702"

del theFCI
del Ham
del Initializer


