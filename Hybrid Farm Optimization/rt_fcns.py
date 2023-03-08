'Real-time operation functions'
import numpy as np
import pandas as pd
#%% Degradation model function
def deg_model(powers, SOC, batt_capacity, EOL):
    # Degradation model
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
    deg_per_cycle = [0., 1 / 1000000., 1 / 200000., 1 / 60000., 1 / 40000.,
                          1 / 20000., 1 / 15000., 1 / 11000., 1 / 10000., 1 / 8000., 1 / 7000.,
                          1 / 6000.]
    energy = 0
    for P in powers:
        energy = energy + abs(P)
    benefits = []
    en100 = energy/2/batt_capacity*100
    DOD = max(SOC) - min(SOC)
    for d in range(len(DOD_index) - 1):
        if DOD >= DOD_index[d] and DOD <= DOD_index[d + 1]:
            deg = deg_per_cycle[d] + (deg_per_cycle[d + 1] - deg_per_cycle[d]) * (
                        DOD - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
            break

    DOD1 = max(en100-DOD,0)
    if DOD1>100:
        deg1 = deg_per_cycle[-1]
    for d in range(len(DOD_index) - 1):
        if DOD1 >= DOD_index[d] and DOD1 <= DOD_index[d + 1]:
            deg1 = deg_per_cycle[d] + (deg_per_cycle[d + 1] - deg_per_cycle[d]) * (
                        DOD1 - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
            break

    total_deg = (deg + deg1)*(1-EOL)

    return total_deg

#%% BMS model function
def BMS_model(E_Prev, P_ESS_h, HyF_Parameters):
    # Unwrapping ESS parameters
    Eff = HyF_Parameters['ESS Efficiency']
    Emax = HyF_Parameters['ESS Capacity']
    Pnom = HyF_Parameters['ESS Nominal Power']
    # Unwrapping powers
    if P_ESS_h < 0:
        P_Cha_h = -P_ESS_h  # All powers positive locally
        P_Dis_h = 0
    if P_ESS_h == 0:
        P_Cha_h = 0
        P_Dis_h = 0
    if P_ESS_h > 0:
        P_Cha_h = 0
        P_Dis_h = P_ESS_h
    # SOP & SOC models
    Pcha_h = min(Pnom, P_Cha_h)
    Pdis_h = min(Pnom, P_Dis_h)
    E_Post = E_Prev + P_Cha_h*Eff - P_Dis_h/Eff
    if E_Post > Emax:
        E_Post = Emax
        P_Cha_h = (E_Post - E_Prev)/Eff
    if E_Post < 0:
        E_Post = 0
        P_Dis_h = (E_Prev - E_Post)*Eff
    P_Real = P_Dis_h - P_Cha_h
    SOC = (E_Post/Emax) * 100

    return P_Real, SOC, E_Post

#%% Efficiency model
def Eff_model(P_pu):
    Inv_curve_P = [0.0, 0.06, 0.08, 0.12, 0.16, 0.295, 0.6945, 1.0]
    array = np.asarray(Inv_curve_P)
    idx = (np.abs(array - P_pu)).argmin()
    P_Ppu_nearest = array[idx]
    Inv_curve_Eff = [0.53, 0.91, 0.945, 0.9649, 0.9702, 0.95, 0.9250, 0.9250]

    Eff_curve = {}
    for Eff, P in enumerate(Inv_curve_P):
        Eff_curve[f'{P}'] = Inv_curve_Eff[Eff]

    return Eff_curve[f'{P_Ppu_nearest}']

#%% Real-time operation
from matplotlib import pyplot as plt
def RT_operation(Pgen_real, P_PCC_toDel, HyF_Parameters):
    ESS_SOCinit = HyF_Parameters['ESS Initial SOC']
    ESS_Cap = HyF_Parameters['ESS Capacity']
    Inv_Pnom = HyF_Parameters['Inverter Pnom']
    P_ESS_Sch = []  # Initializing ESS scheduled powers array by EMS
    P_ESS_Real = []  # Initializing real ESS powers array allowed by BMS
    P_PPC = []  # Initializing real PPC powers array
    SOC = [ESS_SOCinit]  # Initializing SOC array
    E_Prev_h = ESS_SOCinit*ESS_Cap/100  # Initializing energy stored at initial hour
    for h in range(len(P_PCC_toDel)):  # Running real-time operation
        if HyF_Parameters['Config']['Variable Efficiency']:
            P_PCC_toDel[h] = P_PCC_toDel[h] / Eff_model(abs(P_PCC_toDel[h]/Inv_Pnom))
        if Pgen_real[h] >= P_PCC_toDel[h]:  # There is a surplus
            P_ESS_h = -(Pgen_real[h] - P_PCC_toDel[h])  # Redirecting surplus to ESS (Pcha < 0)
        if Pgen_real[h] < P_PCC_toDel[h]:  # There is a deficit
            P_ESS_h = -(Pgen_real[h] - P_PCC_toDel[h])  # Using ESS as backup
        P_ESS_real_h, SOC_h, E_Post_h = BMS_model(E_Prev_h, P_ESS_h, HyF_Parameters)  # Sending powers schdules to BMS
        P_PPC_h = Pgen_real[h] + P_ESS_real_h
        if HyF_Parameters['Config']['Variable Efficiency']:
            P_PPC_h = np.round(P_PPC_h * Eff_model(P_PCC_toDel[h]/Inv_Pnom),2)
        P_ESS_Sch.append(P_ESS_h)
        P_ESS_Real.append(P_ESS_real_h)
        P_PPC.append(np.round(P_PPC_h,2))
        SOC.append(SOC_h)  # SOC at the end of each hour
        E_Prev_h = E_Post_h  # Updating stored energy for next hour

    return SOC, P_ESS_Sch, P_ESS_Real, np.round(P_PPC,2)


