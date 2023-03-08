from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


H = 24 #Hours of Day
D = 2 #Simulation Days
ESS_Energy = 68000 #Energy of the Battery kWh
ESS_Power = 17000 #Power of the Battery kW
ESS_Efficiency = 0.92 #Efficiency of the Battery %
ESS_Cost = 0.005 #Cost of Use Battery $/kWh
Deviation_Cost = 1.1 #Cost of Deviation 30% higher than the Market Price

#Preload Data Initiation
SOCi = 68000 #Initial State of Charge kWh for MD
SOCf =[68000]
Schedule_DM = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #Precarga 25 Datos Schedule Mercado Diario
ESS_DM = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #Precarga 25 Datos Uso Batería Mercado Diario
Purchase_ESS_DM = []
Schedule_ID = [0,0,0,0]
Purchase_Grid_ID = [0,0,0,0]
Purchase_ESS_ID = [0,0,0,0]
Deviation_ID = []
Operation_RT = []
Purchase_Grid_RT = []
Purchase_ESS_RT = []
Deviation_RT = []
PV_RT = []
ESS_D_RT = []
ESS_C_RT = []

#Read Data from Excel
Data = pd.read_excel('Datos.xlsx', sheet_name = 'Sheet1', nrows = 8760)

#Daily Market Function
def MD(d, SOCiMD, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost):
    #Read Prices_real from Excel
    P = list(Data['Precio'][(24*(d+1)):(24+(24*(d+1)))]) #Leer Datos Día Siguiente

    #Read PV Forecast from Excel (10% Error)
    PV_Av_DM = list(Data['FV10'][(24*(d+1)):(24+(24*(d+1)))]) #Leer Datos Día Siguiente

    model = ConcreteModel()

    #Times
    model.time = range(24) #Rango de 0 a 23 horas Para tener del resto de Variables 24 valores
    model.time2 = range(1,24+1) #Rango de 1 a 24 horas Para poder calcular el SOC
    model.time3 = range(24+1) #Rango de 0 a 24 horas Para tener 25 valores de SOC

    #Variables
    #PV Plant Power
    model.PV_P = Var(model.time, bounds = (0, None))

    #Battery Power
    model.SOC = Var(model.time3, bounds = (0, ESS_Energy)) #State of Charge
    model.NC = Var(model.time, domain=Binary)
    model.ND = Var(model.time, domain=Binary)
    model.ESS_C = Var(model.time, bounds = (0, ESS_Power)) #ESS Charge
    model.ESS_D = Var(model.time, bounds = (0, ESS_Power)) #ESS Discharge
    model.Purchase_ESS = Var(model.time, bounds = (0, ESS_Power)) #ESS Purchase from Grid

    #Constraints
    def c1_rule(model, t1):
        return ESS_Power*model.NC[t1] >= model.ESS_C[t1]
    model.c1 = Constraint( model.time, rule=c1_rule )

    def c2_rule(model, t1):
        return (ESS_Power*model.ND[t1]) >= model.ESS_D[t1]
    model.c2 = Constraint( model.time, rule=c2_rule )

    def c3_rule(model, t1):
        return (model.NC[t1] + model.ND[t1]) <= 1
    model.c3 = Constraint( model.time, rule=c3_rule )

    def c4_rule(model, t1):
        return PV_Av_DM[t1] >= (model.PV_P[t1] + (model.ESS_C[t1] - model.Purchase_ESS[t1]))
    model.c4 = Constraint( model.time, rule=c4_rule )

    def c5_rule(model, t1):
        return model.ESS_C[t1] >= model.Purchase_ESS[t1]
    model.c5 = Constraint( model.time, rule=c5_rule )

    def c6_rule(model, t2):
        return model.SOC[t2] == (model.SOC[t2-1] + (model.ESS_C[t2-1]*ESS_Efficiency - model.ESS_D[t2-1]/ESS_Efficiency))
    model.c6 = Constraint( model.time2, rule=c6_rule )

    def c7_rule(model):
        return model.SOC[0] == SOCiMD
    model.c7 = Constraint( rule=c7_rule )

    #Objective Function 
    model.obj = Objective(
        expr = sum(((P[t1]*(model.ESS_D[t1] + model.PV_P[t1] - model.Purchase_ESS[t1])) - ((ESS_Cost*(model.ESS_C[t1] + model.ESS_D[t1]))) for t1 in model.time)),
        sense = maximize)

    opt = SolverFactory('cbc')

    opt.solve(model)

    """model.pprint()"""

    Schedule = [(model.PV_P[t1]() + model.ESS_D[t1]()) - model.Purchase_ESS[t1]() for t1 in model.time]
    Purchase_ESS = [model.Purchase_ESS[t1]() for t1 in model.time]
    PBAT_Data = [model.ESS_C[t1]() + model.ESS_D[t1]() for t1 in model.time]

    return Schedule, Purchase_ESS, PBAT_Data



#Intraday Darket Function
def ID(Schedule_DM, ESS_DM, H_ID, FinalH_ID, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost):
    #Read Prices_real from Excel
    P = list(Data['Precio'][h+(24*d):(FinalH_ID)+(24*d)])

    #Read PV Forecast from Excel (5% Error)
    PV_Av_ID = list(Data['FV5'][h+(24*d):(FinalH_ID)+(24*d)])

    #Schedule of Daily Market
    Schedule_P_DM = Schedule_DM [h+(24*d):(FinalH_ID)+(24*d)]

    #ESS Daily Market Operation
    ESS_P_DM = ESS_DM [h+(24*d):(FinalH_ID)+(24*d)]

    model = ConcreteModel()

    #Times
    model.time = range(H_ID) #Rango de 0 a 5 horas Para tener del resto de Variables 6 valores
    model.time2 = range(1,H_ID+1) #Rango de 1 a 6 horas Para poder calcular el SOC
    model.time3 = range(H_ID+1) #Rango de 0 a 6 horas Para tener 6 valores de SOC

    #Variables
    #PV Plant Power
    model.PV_P1 = Var(model.time, bounds = (0, None))
    model.PV_P2 = Var(model.time, bounds = (0, None))

    #Battery Power
    model.SOC = Var(model.time3, bounds = (0, ESS_Energy))
    model.NC = Var(model.time, domain=Binary)
    model.ND = Var(model.time, domain=Binary)
    model.ESS_C = Var(model.time, bounds = (0, ESS_Power))
    model.ESS_D1 = Var(model.time, bounds = (0, ESS_Power))
    model.ESS_D2 = Var(model.time, bounds = (0, ESS_Power))
    model.Purchase_ESS1 = Var(model.time, bounds = (0, ESS_Power))
    model.Purchase_ESS2 = Var(model.time, bounds = (0, ESS_Power))

    #Deviations
    model.Deviation_P = Var(model.time, bounds = (0, None))
    model.Deviation_P1 = Var(model.time)
    model.Deviation_P2 = Var(model.time)
    model.Purchase_Grid = Var(model.time, bounds = (0, None))

    #Constraints
    def c7_rule(model, t1):
        return ESS_Power*model.NC[t1] >= model.ESS_C[t1]
    model.c7 = Constraint( model.time, rule=c7_rule )

    def c8_rule(model, t1):
        return ESS_Power*model.ND[t1] >= (model.ESS_D1[t1] + model.ESS_D2[t1])
    model.c8 = Constraint( model.time, rule=c8_rule )

    def c9_rule(model, t1):
        return (model.NC[t1] + model.ND[t1]) <= 1
    model.c9 = Constraint( model.time, rule=c9_rule )

    def c10_rule(model, t1):
        return PV_Av_ID[t1] >= (model.PV_P1[t1] + model.PV_P2[t1] + (model.ESS_C[t1] - (model.Purchase_ESS1[t1] + model.Purchase_ESS2[t1])))
    model.c10 = Constraint( model.time, rule=c10_rule )

    def c11_rule(model, t1):
        return model.ESS_C[t1] >= (model.Purchase_ESS1[t1] +  model.Purchase_ESS2[t1])
    model.c11 = Constraint( model.time, rule=c11_rule )

    def c12_rule(model, t2):
        return model.SOC[t2] == (model.SOC[t2-1] + model.ESS_C[t2-1]*ESS_Efficiency - ((model.ESS_D1[t2-1]/ESS_Efficiency) + (model.ESS_D2[t2-1]/ESS_Efficiency)))
    model.c12 = Constraint( model.time2, rule=c12_rule )

    def c13_rule(model, t1):
        return model.Deviation_P[t1] >= model.Deviation_P1[t1]
    model.c13 = Constraint( model.time, rule=c13_rule )

    def c14_rule(model, t1):
        return model.Deviation_P[t1] >= model.Deviation_P2[t1]
    model.c14 = Constraint( model.time, rule=c14_rule )

    def c15_rule(model, t1):
        return model.Deviation_P1[t1] == (Schedule_P_DM[t1] - (((model.PV_P1[t1] + model.ESS_D1[t1]) - model.Purchase_ESS1[t1]) + model.Purchase_Grid[t1]))
    model.c15 = Constraint( model.time, rule=c15_rule )

    def c16_rule(model, t1):
        return model.Deviation_P2[t1] == ((((model.PV_P1[t1] + model.ESS_D1[t1]) - model.Purchase_ESS1[t1]) + model.Purchase_Grid[t1]) - Schedule_P_DM[t1])
    model.c16 = Constraint( model.time, rule=c16_rule )

    def c17_rule(model):
        return model.SOC[0] == SOCi
    model.c17 = Constraint( rule=c17_rule )

    #Objective Function
    model.obj = Objective(
        expr = sum(((P[t1]*(((model.ESS_D1[t1] + model.ESS_D2[t1] + model.PV_P1[t1] + model.PV_P2[t1]) - (model.Purchase_ESS1[t1] + model.Purchase_ESS2[t1])) - Schedule_P_DM[t1])) - ((ESS_Cost*((model.ESS_C[t1] + model.ESS_D1[t1] + model.ESS_D2[t1]) - ESS_P_DM[t1])) + (Deviation_Cost*(P[t1]*model.Deviation_P[t1])) + (P[t1]*model.Purchase_Grid[t1])) for t1 in model.time)),
        sense = maximize)

    opt = SolverFactory('cbc')

    opt.solve(model)

    """model.pprint()"""
    
    SOC_DM = model.SOC[H_ID]()
    Schedule = [(model.PV_P1[t1]() + model.PV_P2[t1]() + model.ESS_D1[t1]() + model.ESS_D2[t1]() + model.Purchase_Grid[t1]()) - (model.Purchase_ESS1[t1]() + model.Purchase_ESS2[t1]()) for t1 in model.time]
    Purchase_Grid = [model.Purchase_Grid[t1]() for t1 in model.time]
    Purchase_ESS = [-model.Purchase_ESS1[t1]() - model.Purchase_ESS2[t1]() for t1 in model.time]
    Deviation = [model.Deviation_P[t1]() for t1 in model.time]

    return SOC_DM, Schedule, Purchase_Grid, Purchase_ESS, Deviation



#Real Time Function
def TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost):
    #Read Prices_real from Excel
    P = list(Data['Precio'][h+(24*d):(FinalH_RT)+(24*d)])

    #Read Real PV from Excel (0% Error)
    PV_Av_RT1 = list(Data['FV0'][h+(24*d):h+(24*d)+1])

    #Read PV Forecast from Excel (3.5% Error)
    PV_Av_RT2 = list(Data['FV35'][(h+1)+(24*d):(FinalH_RT)+(24*d)])
    
    PV_Av_RT = PV_Av_RT1 + PV_Av_RT2

    #Schedule of Intraday Market
    Schedule_P_ID = Schedule_ID [h+(24*d):(FinalH_RT)+(24*d)]

    model = ConcreteModel()

    #Times
    model.time = range(H_RT) #Rango de 0 a 3 horas Para tener del resto de Variables 6 valores
    model.time2 = range(1,H_RT+1) #Rango de 1 a 4 horas Para poder calcular el SOC
    model.time3 = range(H_RT+1) #Rango de 0 a 4 horas Para tener 5 valores de SOC

    #Variables
    #PV Plant Power
    model.PV_P = Var(model.time, bounds = (0, None))

    #Battery Power
    model.SOC = Var(model.time3, bounds = (0, ESS_Energy))
    model.NC = Var(model.time, domain=Binary)
    model.ND = Var(model.time, domain=Binary)
    model.ESS_C = Var(model.time, bounds = (0, ESS_Power))
    model.ESS_D = Var(model.time, bounds = (0, ESS_Power))
    model.Purchase_ESS = Var(model.time, bounds = (0, ESS_Power))

    #Deviations
    model.Deviation_P = Var(model.time, bounds = (0, None))
    model.Deviation_P1 = Var(model.time)
    model.Deviation_P2 = Var(model.time)
    model.Purchase_Grid = Var(model.time, bounds = (0, None))

    #Constraints
    def c18_rule(model, t1):
        return ESS_Power*model.NC[t1] >= model.ESS_C[t1]
    model.c18 = Constraint( model.time, rule=c18_rule )

    def c19_rule(model, t1):
        return ESS_Power*model.ND[t1] >= model.ESS_D[t1]
    model.c19 = Constraint( model.time, rule=c19_rule )

    def c20_rule(model, t1):
        return (model.NC[t1] + model.ND[t1]) <= 1
    model.c20 = Constraint( model.time, rule=c20_rule )

    def c21_rule(model, t1):
        return PV_Av_RT[t1] >= (model.PV_P[t1] + (model.ESS_C[t1] - model.Purchase_ESS[t1]))
    model.c21 = Constraint( model.time, rule=c21_rule )

    def c22_rule(model, t1):
        return model.ESS_C[t1] >= model.Purchase_ESS[t1]
    model.c22 = Constraint( model.time, rule=c22_rule )

    def c23_rule(model, t2):
        return model.SOC[t2] == (model.SOC[t2-1] + model.ESS_C[t2-1]*ESS_Efficiency - (model.ESS_D[t2-1]/ESS_Efficiency))
    model.c23 = Constraint( model.time2, rule=c23_rule )

    def c24_rule(model, t1):
        return model.Deviation_P[t1] >= model.Deviation_P1[t1]
    model.c24 = Constraint( model.time, rule=c24_rule )

    def c25_rule(model, t1):
        return model.Deviation_P[t1] >= model.Deviation_P2[t1]
    model.c25 = Constraint( model.time, rule=c25_rule )

    def c26_rule(model, t1):
        return model.Deviation_P1[t1] == (Schedule_P_ID[t1] - (((model.PV_P[t1] + model.ESS_D[t1]) - model.Purchase_ESS[t1]) + model.Purchase_Grid[t1]))
    model.c26 = Constraint( model.time, rule=c26_rule )

    def c27_rule(model, t1):
        return model.Deviation_P2[t1] == ((((model.PV_P[t1] + model.ESS_D[t1]) - model.Purchase_ESS[t1]) + model.Purchase_Grid[t1]) - Schedule_P_ID[t1])
    model.c27 = Constraint( model.time, rule=c27_rule )

    def c28_rule(model):
        return model.SOC[0] == SOCi
    model.c28 = Constraint( rule=c28_rule )

    #Objective Function
    model.obj = Objective(
        expr = sum(((P[t1]*(((model.ESS_D[t1] + model.PV_P[t1]) - (model.Purchase_ESS[t1])) - Schedule_P_ID[t1])) - ((ESS_Cost*(model.ESS_C[t1] + model.ESS_D[t1])) + (Deviation_Cost*(P[t1]*model.Deviation_P[t1])) + (P[t1]*model.Purchase_Grid[t1])) for t1 in model.time)),
        sense = maximize)

    opt = SolverFactory('cbc')

    opt.solve(model)  
  
    """model.pprint()"""

    SOC = model.SOC[1]() #Precarga del SOC para el Siguiente Horizonte
    Schedule = [((model.PV_P[0]() + model.ESS_D[0]()) + model.Purchase_Grid[0]()) - model.Purchase_ESS[0]()]
    Purchase_Grid = [model.Purchase_Grid[0]()]
    Purchase_ESS = [-model.Purchase_ESS[0]()]
    Deviation = [model.Deviation_P[0]()] #Deviations in RT
    PV = [model.PV_P[0]()]
    ESS_D = [model.ESS_D[0]()]
    ESS_C = [-model.ESS_C[0]()]
    
    return SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C



#Market Sequence
for d in range(D): #Rango de 0 a 364 días
    for h in range(H): #Rango de 0 a 23 horas
        if h == 0:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 4

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        if h == 1:
            print('Hour',h)
            #Intraday 2
            #Temporal Horizon Intraday 2
            H_ID = 24

            #Final Hour Intraday 2
            FinalH_ID = 25

            #Call to Function Intraday 2
            SOC_DM, Schedule, Purchase_Grid, Purchase_ESS, Deviation = ID(Schedule_DM, ESS_DM, H_ID, FinalH_ID, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)
            
            Schedule_ID[h+(24*d):] = Schedule
            Purchase_Grid_ID[h+(24*d):] = Purchase_Grid
            Purchase_ESS_ID[h+(24*d):] = Purchase_ESS
            Deviation_ID[h+(24*d):] = Deviation
            
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 5

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)
            
        elif h == 2:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 6

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 3:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 7

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 4:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 8

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h ==5:
            print('Hour',h)
            #Intraday 3
            #Temporal Horizon Intraday 3
            H_ID = 20

            #Final Hour Intraday 3
            FinalH_ID = 25

            #Call to Function Intraday 3
            SOC_DM, Schedule, Purchase_Grid, Purchase_ESS, Deviation = ID(Schedule_DM, ESS_DM, H_ID, FinalH_ID, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            Schedule_ID[h+(24*d):] = Schedule
            Purchase_Grid_ID[h+(24*d):] = Purchase_Grid
            Purchase_ESS_ID[h+(24*d):] = Purchase_ESS
            Deviation_ID[h+(24*d):] = Deviation
            
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 9

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC  #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 6:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 10

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 7:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 11

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h ==8:
            print('Hour',h)
            #Intraday 4
            #Temporal Horizon Intraday 4
            H_ID = 17

            #Final Hour Intraday 4
            FinalH_ID = 25

            #Call to Function Intraday 4
            SOC_DM, Schedule, Purchase_Grid, Purchase_ESS, Deviation = ID(Schedule_DM, ESS_DM, H_ID, FinalH_ID, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCiMD = SOC_DM
            Schedule_ID[h+(24*d):] = Schedule
            Purchase_Grid_ID[h+(24*d):] = Purchase_Grid
            Purchase_ESS_ID[h+(24*d):] = Purchase_ESS
            Deviation_ID[h+(24*d):] = Deviation

            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 12

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 9:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 13

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 10:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 14

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 11:
            print('Hour',h)
            #Daily Market
            #Call to Function Daily Market
            Schedule, Purchase_ESS, PBAT_Data = MD(d, SOCiMD, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost)

            Schedule_DM.extend(Schedule)
            Purchase_ESS_DM.extend(Purchase_ESS)
            ESS_DM.extend(PBAT_Data)

            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 15

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 12:
            print('Hour',h)
            #Intraday 5
            #Temporal Horizon Intraday 5
            H_ID = 13

            #Final Hour Intraday 5
            FinalH_ID = 25

            #Call to Function Intraday 5
            SOC_DM, Schedule, Purchase_Grid, Purchase_ESS, Deviation = ID(Schedule_DM, ESS_DM, H_ID, FinalH_ID, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            Schedule_ID[h+(24*d):] = Schedule
            Purchase_Grid_ID[h+(24*d):] = Purchase_Grid
            Purchase_ESS_ID[h+(24*d):] = Purchase_ESS
            Deviation_ID[h+(24*d):] = Deviation

            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 16

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 13:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 17

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 14:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 18

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 15:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 19

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)


        elif h ==16:
            print('Hour',h)
            #Intraday 6
            #Temporal Horizon Intraday 6
            H_ID = 9

            #Final Hour Intraday 6
            FinalH_ID = 25

            #Call to Function Intraday 6
            SOC_DM, Schedule, Purchase_Grid, Purchase_ESS, Deviation = ID(Schedule_DM, ESS_DM, H_ID, FinalH_ID, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            Schedule_ID[h+(24*d):] = Schedule
            Purchase_Grid_ID[h+(24*d):] = Purchase_Grid
            Purchase_ESS_ID[h+(24*d):] = Purchase_ESS
            Deviation_ID[h+(24*d):] = Deviation

            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 20

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 17:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 21

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC  #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 18:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 22

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 19:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 23

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 20:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 24

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 21:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 25

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 22:
            print('Hour',h)
            #Intraday 1
            #Temporal Horizon Intraday 1
            H_ID = 27

            #Final Hour Intraday 1
            FinalH_ID = 49 #49 porque ya es del dia Siguiente (en vez de 25)

            #Call to Function Intraday 1
            SOC_DM, Schedule, Purchase_Grid, Purchase_ESS, Deviation = ID(Schedule_DM, ESS_DM, H_ID, FinalH_ID, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            Schedule_ID[h+(24*d):] = Schedule
            Purchase_Grid_ID[h+(24*d):] = Purchase_Grid
            Purchase_ESS_ID[h+(24*d):] = Purchase_ESS
            Deviation_ID[h+(24*d):] = Deviation

            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 26

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

        elif h == 23:
            print('Hour',h)
            #Real Time
            #Temporal Horizon Real Time
            H_RT = 4

            #Final Hour Real Time
            FinalH_RT = 27

            #Call to Function Real Time
            SOC, Schedule, Purchase_Grid, Purchase_ESS, Deviation, PV, ESS_D, ESS_C = TR(Schedule_ID, H_RT, FinalH_RT, h, d, SOCi, ESS_Energy, ESS_Power, ESS_Efficiency, ESS_Cost, Deviation_Cost)

            SOCi = SOC #Save SOC for Next Hour of Real Time
            SOCf.extend([SOC])
            Operation_RT.extend(Schedule)
            Purchase_Grid_RT.extend(Purchase_Grid)
            Purchase_ESS_RT.extend(Purchase_ESS)
            Deviation_RT.extend(Deviation)
            PV_RT.extend(PV)
            ESS_D_RT.extend(ESS_D)
            ESS_C_RT.extend(ESS_C)

#Plots

SOCf = [i * (100/ESS_Energy) for i in SOCf] #Convert Energy kWh to Percentage %

plt.subplot(4, 1, 1)
plt.plot(range(D*H+25), Schedule_DM, 'b', label='Schedule')
#plt.plot(range(D*H+25), ESS_DM, 'r', label='Battery')
#plt.plot(range(D*H), Purchase_ESS_DM, 'k', label='Purchase_ESS')
plt.axis([0, (D*H), -17000, 40000])
plt.xlabel('Time (Hours)')
plt.ylabel('Power (kW)')
plt.title('Daily')
plt.legend(loc='right')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(range(D*H+25), Schedule_ID, 'y', label='Schedule')
plt.plot(range(D*H+25), Purchase_ESS_ID, 'b', label='Purchase_ESS')
plt.plot(range(D*H+25), Purchase_Grid_ID, 'g', label='Purchase_Grid')
plt.plot(range(D*H+25), Deviation_ID, 'r', label='Deviation')
plt.axis([0, (D*H), -17000, 40000])
plt.xlabel('Time (Hours)')
plt.ylabel('Power (kW)')
plt.title('Intraday')
plt.legend(loc='right')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(range(D*H), Operation_RT, 'y', label='Schedule')
#plt.plot(range(D*H), PPVT, 'm', label='PPVT') #PV Power Available
#plt.plot(range(D*H), PV_RT, 'y', label='PV_P') #PV Power Use
#plt.plot(range(D*H), ESS_D_RT, 'r', label='Discharge')
#plt.plot(range(D*H), ESS_C_RT, 'g', label='Charge')
plt.plot(range(D*H), Purchase_ESS_RT, 'b', label='Purchase_ESS')
plt.plot(range(D*H), Purchase_Grid_RT, 'g', label='Purchase_Grid')
plt.plot(range(D*H), Deviation_RT, 'r', label='Deviation')
plt.axis([0, (D*H), -17000, 40000])
plt.xlabel('Time (Hours)')
plt.ylabel('Power (kW)')
plt.title('Real Time')
plt.legend(loc='right')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(range(D*H+1), SOCf, 'k')
plt.axis([1, (D*H+1), 0, 100])
plt.xlabel('Time (Hours)')
plt.ylabel('SOC (%)')
plt.title('SOC')
plt.grid(True)

plt.tight_layout()
plt.show()