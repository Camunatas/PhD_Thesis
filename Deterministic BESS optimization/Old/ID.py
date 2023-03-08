from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

H = 6 #Hours del Intervalo
SOCiMD = 68000 #Estado Inicial Batería en kWh para MD
ESS_Energy = 68000 #Energía Disponible en la Batería en kWh
ESS_Power = 17000 #Potencia de la Batería en kW
ESS_Efficiency = 0.92 #Eficiencia de la Batería en %
ESS_Cost = 0.005 #Coste por Usar la Batería en kWh
Deviation_Cost = 1.3 #Coste por Incurrir en Desvío 30% superior al Precio de Mercado

#Lectura Datos
Data = pd.read_excel('Datos.xlsx', sheet_name = 'Sheet1', nrows = H)

#Lectura Precios
P = list(Data['Precio'])

# Lectura Previsión PV (5% Error)
PV_Av_ID = list(Data['FV5'])

Schedule_P_DM = [0,0,0,0,0,0,0,0,2000,4000,6000,8000,10000,12000,14000,12000,10000,8000,6000,4000,2000,0,0,0]
ESS_P_DM = [0,0,0,0,0,0,0,0,2000,4000,6000,8000,10000,12000,14000,12000,10000,8000,6000,4000,2000,0,0,0]

model = ConcreteModel()

#Times
model.time = range(H) #Rango de 0 a 5 horas Para tener del resto de Variables 6 valores
model.time2 = range(1,H+1) #Rango de 1 a 6 horas Para poder calcular el SOC
model.time3 = range(H+1) #Rango de 0 a 6 horas Para tener 6 valores de SOC

#Variables
#PV Plant Power
model.PV_P1 = Var(model.time, bounds = (0, None))
model.PV_P2 = Var(model.time, bounds = (0, None))

#Battery Power
model.SOC = Var(model.time3, initialize=SOCiMD, bounds = (0, ESS_Energy))
model.NC = Var(model.time, domain=Binary)
model.ND = Var(model.time, domain=Binary)
model.ESS_C = Var(model.time, bounds = (0, ESS_Power))
model.ESS_D1 = Var(model.time, bounds = (0, ESS_Power))
model.ESS_D2 = Var(model.time, bounds = (0, ESS_Power))
model.Purchase_ESS1 = Var(model.time, bounds = (0, ESS_Power))
model.Purchase_ESS2 = Var(model.time, bounds = (0, ESS_Power))

#Desvíos
model.Deviation_P = Var(model.time, bounds = (0, None))
model.Deviation_P1 = Var(model.time)
model.Deviation_P2 = Var(model.time)
model.Purchase_Grid = Var(model.time, bounds = (0, None))

#Restricciones
"""
def c1_rule(model, t3):
    return model.SOC[t3] >= 0
model.c1 = Constraint( model.time3, rule=c1_rule )

def c2_rule(model, t3):
    return model.SOC[t3] <= ESS_Energy
model.c2 = Constraint( model.time3, rule=c2_rule )

def c3_rule(model, t2):
    return model.ESS_D[t2-1]/ESS_Efficiency <= model.SOC[t2]
model.c3 = Constraint( model.time2, rule=c3_rule )

def c4_rule(model, t2):
    return model.ESS_C[t2-1]*ESS_Efficiency <= ESS_Energy - model.SOC[t2]
model.c4 = Constraint( model.time2, rule=c4_rule )
"""
def c5_rule(model, t1):
    return ESS_Power*model.NC[t1] >= model.ESS_C[t1]
model.c5 = Constraint( model.time, rule=c5_rule )

def c6_rule(model, t1):
    return ESS_Power*model.ND[t1] >= model.ESS_D1[t1] + model.ESS_D2[t1]
model.c6 = Constraint( model.time, rule=c6_rule )

def c7_rule(model, t1):
    return model.NC[t1] + model.ND[t1] <= 1
model.c7 = Constraint( model.time, rule=c7_rule )

def c8_rule(model, t1):
    return PV_Av_ID[t1] >= model.PV_P1[t1] + model.PV_P2[t1] + model.ESS_C[t1] - (model.Purchase_ESS1[t1] + model.Purchase_ESS2[t1])
model.c8 = Constraint( model.time, rule=c8_rule )

def c9_rule(model, t1):
    return model.ESS_C[t1] >= model.Purchase_ESS1[t1] +  model.Purchase_ESS2[t1]
model.c9 = Constraint( model.time, rule=c9_rule )

def c10_rule(model, t2):
    return model.SOC[t2] == model.SOC[t2-1] + model.ESS_C[t2-1]*ESS_Efficiency - (model.ESS_D1[t2-1]/ESS_Efficiency + model.ESS_D2[t2-1]/ESS_Efficiency)
model.c10 = Constraint( model.time2, rule=c10_rule )

def c11_rule(model, t1):
    return model.Deviation_P[t1] >= model.Deviation_P1[t1]
model.c11 = Constraint( model.time, rule=c11_rule )

def c12_rule(model, t1):
    return model.Deviation_P[t1] >= model.Deviation_P2[t1]
model.c12 = Constraint( model.time, rule=c12_rule )

def c13_rule(model, t1):
    return model.Deviation_P1[t1] == Schedule_P_DM[t1] - (((model.PV_P1[t1] + model.ESS_D1[t1]) - model.Purchase_ESS1[t1]) + model.Purchase_Grid[t1])
model.c13 = Constraint( model.time, rule=c13_rule )

def c14_rule(model, t1):
    return model.Deviation_P2[t1] == (((model.PV_P1[t1] + model.ESS_D1[t1]) - model.Purchase_ESS1[t1]) + model.Purchase_Grid[t1]) - Schedule_P_DM[t1]
model.c14 = Constraint( model.time, rule=c14_rule )

#Funcion Objetivo
model.obj = Objective(
    expr = sum(((P[t1]*(((model.ESS_D1[t1] + model.ESS_D2[t1] + model.PV_P1[t1] + model.PV_P2[t1]) - (model.Purchase_ESS1[t1] + model.Purchase_ESS2[t1])) - Schedule_P_DM[t1])) - ((ESS_Cost*((model.ESS_C[t1] + model.ESS_D1[t1] + model.ESS_D2[t1]) - ESS_P_DM[t1])) + (Deviation_Cost*(P[t1]*model.Deviation_P[t1])) + (P[t1]*model.Purchase_Grid[t1])) for t1 in model.time)),
    sense = maximize)

opt = SolverFactory('cbc')

results = opt.solve(model, tee=True)

model.pprint()

"""plt.subplot(3, 1, 1)
plt.plot(model.time, PV_Av_DM, 'm', model.time, [model.PV_P[t1]() for t1 in model.time], 'y', model.time, [model.ESS_D[t1]() for t1 in model.time], 'r', model.time, [model.ESS_C[t1]() for t1 in model.time], 'g', model.time, [model.Purchase_ESS[t1]() for t1 in model.time], 'b')
plt.axis([0, 23, 0, 20000])
plt.xlabel('Tiempo')
plt.ylabel('Potencia')
plt.title('Sistema PV + ESS')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(model.time, [model.SOC[t1]() for t1 in model.time], 'k')
plt.axis([0, 23, 0, 80000])
plt.xlabel('Tiempo')
plt.ylabel('Energía')
plt.title('SOC')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(model.time, P, 'k')
plt.axis([0, 23, 0, 1.25])
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.title('Variación Precio')
plt.grid(True)

plt.tight_layout()
plt.show()"""