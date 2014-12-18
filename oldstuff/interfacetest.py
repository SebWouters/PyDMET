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
    FCIverbose = 2
    theFCI = PyCheMPS2.PyFCI(Ham, Nel_up, Nel_down, Irrep, maxMemWorkMB, FCIverbose)
    GSvector = np.zeros([ theFCI.getVecLength() ], dtype=ctypes.c_double)
   
    if (startRandom):
        theFCI.FillRandom( theFCI.getVecLength() , GSvector )
    else:
        GSvector[ theFCI.LowestEnergyDeterminant() ] = 1.0
      
    EnergyFCI = theFCI.GSDavidson( GSvector )
    theFCI.CalcSpinSquared( GSvector )
    return ( EnergyFCI , GSvector, theFCI )

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

Re2RDMadd = np.zeros( [ Ham.getL()*Ham.getL()*Ham.getL()*Ham.getL() ], dtype=ctypes.c_double )
Im2RDMadd = np.zeros( [ Ham.getL()*Ham.getL()*Ham.getL()*Ham.getL() ], dtype=ctypes.c_double )
RePartAdd , ImPartAdd = theFCI.RetardedGF_addition( omega, eta, orb_alpha, orb_beta, isUp, GSenergy, GSvector, Ham, Re2RDMadd, Im2RDMadd )
Re2RDMadd = Re2RDMadd.reshape( [Ham.getL(), Ham.getL(), Ham.getL(), Ham.getL()], order='F' )
Im2RDMadd = Im2RDMadd.reshape( [Ham.getL(), Ham.getL(), Ham.getL(), Ham.getL()], order='F' )

Re2RDMrem = np.zeros( [ Ham.getL()*Ham.getL()*Ham.getL()*Ham.getL() ], dtype=ctypes.c_double )
Im2RDMrem = np.zeros( [ Ham.getL()*Ham.getL()*Ham.getL()*Ham.getL() ], dtype=ctypes.c_double )
RePartRem , ImPartRem = theFCI.RetardedGF_removal( omega, eta, orb_alpha, orb_beta, isUp, GSenergy, GSvector, Ham, Re2RDMrem, Im2RDMrem )
Re2RDMrem = Re2RDMrem.reshape( [Ham.getL(), Ham.getL(), Ham.getL(), Ham.getL()], order='F' )
Im2RDMrem = Im2RDMrem.reshape( [Ham.getL(), Ham.getL(), Ham.getL(), Ham.getL()], order='F' )

print "LDOS( omega =", omega, "; eta =", eta, "; alpha =", orb_alpha, "; beta =", orb_beta, ") =", - (ImPartAdd + ImPartRem) / m.pi
print "                              The result should be close to 5.15931306704457"
print "Addition GF =", RePartAdd + 1j*ImPartAdd," and removal GF =", RePartRem + 1j*ImPartRem

orb_alpha = 3
orb_beta  = 3
omega     = 0.7358
eta       = 0.001
RePart , ImPart = theFCI.DensityResponseGF( omega, eta, orb_alpha, orb_beta, GSenergy, GSvector )
print "LDDR( omega =", omega, "; eta =", eta, "; alpha =", orb_alpha, "; beta =", orb_beta, ") =", - ImPart / m.pi
print "                                The result should be close to 1.58755549193702"

Re2RDMfw = np.zeros( [ Ham.getL()*Ham.getL()*Ham.getL()*Ham.getL() ], dtype=ctypes.c_double )
Im2RDMfw = np.zeros( [ Ham.getL()*Ham.getL()*Ham.getL()*Ham.getL() ], dtype=ctypes.c_double )
RePartFW , ImPartFW = theFCI.DensityResponseGF_forward( omega, eta, orb_alpha, orb_beta, GSenergy, GSvector, Re2RDMfw, Im2RDMfw )
Re2RDMfw = Re2RDMfw.reshape( [Ham.getL(), Ham.getL(), Ham.getL(), Ham.getL()], order='F' )
Im2RDMfw = Im2RDMfw.reshape( [Ham.getL(), Ham.getL(), Ham.getL(), Ham.getL()], order='F' )

Re2RDMbw = np.zeros( [ Ham.getL()*Ham.getL()*Ham.getL()*Ham.getL() ], dtype=ctypes.c_double )
Im2RDMbw = np.zeros( [ Ham.getL()*Ham.getL()*Ham.getL()*Ham.getL() ], dtype=ctypes.c_double )
RePartBW , ImPartBW = theFCI.DensityResponseGF_backward( omega, eta, orb_alpha, orb_beta, GSenergy, GSvector, Re2RDMbw, Im2RDMbw )
Re2RDMbw = Re2RDMbw.reshape( [Ham.getL(), Ham.getL(), Ham.getL(), Ham.getL()], order='F' )
Im2RDMbw = Im2RDMbw.reshape( [Ham.getL(), Ham.getL(), Ham.getL(), Ham.getL()], order='F' )

print "LDDR( omega =", omega, "; eta =", eta, "; alpha =", orb_alpha, "; beta =", orb_beta, ") =", - (ImPartFW - ImPartBW) / m.pi
print "                                The result should be close to 1.58755549193702"
print "Forward GF =", RePartFW + 1j*ImPartFW," and backward GF =", RePartBW + 1j*ImPartBW

del theFCI
del Ham
del Initializer

