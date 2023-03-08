#%% Importing libraries
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np

#%% Daily market bidding
def DM(HyF_Parameters, Gen_Pred_DM, Price_Pred_DM):
    # Unpacking hybrid farm parameters dictionary
    if HyF_Parameters['Config']['ESS DM Participation']:
        ESS_Emax = HyF_Parameters['ESS Capacity']
    else:
        ESS_Emax = 0.0000001
    ESS_Pmax = HyF_Parameters['ESS Nominal Power']
    ESS_Eff = HyF_Parameters['ESS Efficiency']
    ESS_Cost = HyF_Parameters['ESS Replacement Cost']
    ESS_SOCi = HyF_Parameters['ESS Initial SOC']
    Config = HyF_Parameters['Config']
    Inv_Pnom = HyF_Parameters['Inverter Pnom']
    # Model initialization
    model = pyo.ConcreteModel()
    # Model variables
    model.time = pyo.RangeSet(0, len(Price_Pred_DM) - 1)
    model.E = pyo.Var(model.time, bounds=(0, ESS_Emax),
                      initialize=0)  # ESS energy at the end of the hour
    model.charging = pyo.Var(model.time, domain=pyo.Binary)     # Charge verifier
    model.discharging = pyo.Var(model.time, domain=pyo.Binary)  # Discharge verifier
    model.ESS_C = pyo.Var(model.time, bounds=(0, ESS_Pmax))     # Energy being charged from turbine during period
    model.ESS_D = pyo.Var(model.time, bounds=(0, ESS_Pmax))     # Energy being discharged during period
    model.ESS_Purch = pyo.Var(model.time, bounds=(0, ESS_Pmax)) # Energy being purchased during period
    model.DOD = pyo.Var(bounds=(0, 100))                        # ESS max DOD
    model.Inv_Eff = pyo.Var(model.time, bounds=(0, 1))          # PCC inverter efficiency
    model.ESSInv_Eff = pyo.Var(model.time, bounds=(0, 1))       # ESS and PCC inverter efficiency combination
    model.Inv_Ppu = pyo.Var(model.time, bounds=(0, 1),
                            initialize=0)                       # PCC inverter power in p.u.
    model.deg_cost = pyo.Var()                                  # Degradation costs
    model.max_SOC = pyo.Var(bounds=(ESS_SOCi, 100))             # ESS max SOC
    model.min_SOC = pyo.Var(bounds=(0, ESS_SOCi))               # ESS min SOC
    model.WTG_Psold = pyo.Var(model.time, bounds=(0, None),
                              initialize=0)                     # Generated power directly sold
    model.WTG_Pgen = pyo.Var(model.time, initialize=0)          # Generated power
    Gen_Pred_DM = np.array(Gen_Pred_DM)

    # Model constraints
    def ESS_Pcha_max_Rule(model, t):  # Forces limit on charging power
        return (ESS_Pmax * model.charging[t]) >= model.ESS_C[t] + model.ESS_Purch[t]
    model.ESS_Pcha_max = pyo.Constraint(model.time, rule=ESS_Pcha_max_Rule)

    def ESS_Pdis_max_Rule(model, t):  # Forces limit on discharging power
        return (ESS_Pmax * model.discharging[t]) >= model.ESS_D[t]
    model.ESS_Pdis_max = pyo.Constraint(model.time, rule=ESS_Pdis_max_Rule)

    def ESS_cha_dis_Rule(model, t):  # Prevents orders of charge and discharge simultaneously
        return (model.charging[t] + model.discharging[t]) <= 1
    model.ESS_cha_dis = pyo.Constraint(model.time, rule=ESS_cha_dis_Rule)

    def ESS_SOC_end_Rule(model):
        return model.E[len(Price_Pred_DM) - 1] == 0.0
    model.ESS_SOC_end = pyo.Constraint(rule=ESS_SOC_end_Rule)

    def ESS_SOCmax_Rule(model, t):
        return model.max_SOC >= model.E[t] * (100 // ESS_Emax)
    model.ESS_SOCmax = pyo.Constraint(model.time, rule=ESS_SOCmax_Rule)

    def ESS_SOCmin_Rule(model, t):
        return model.min_SOC <= model.E[t] * (100 // ESS_Emax)
    model.ESS_SOCmin = pyo.Constraint(model.time, rule=ESS_SOCmin_Rule)

    def ESS_DODmax_Rule(model):
        return model.DOD == model.max_SOC - model.min_SOC
    model.ESS_DODmax = pyo.Constraint(rule=ESS_DODmax_Rule)

    def ESS_Purch_limit_Rule(model, t):
        # return model.ESS_C[t] >= model.ESS_Purch[t]
        return model.ESS_Purch[t] == 0
    model.ESS_Purch_limit = pyo.Constraint(model.time, rule=ESS_Purch_limit_Rule)

    def WTG_Pgen_set_Rule(model, t):  # Storing generted power on a variable
        if Gen_Pred_DM.size == 0 or len(Gen_Pred_DM) < len(Price_Pred_DM):
            return model.WTG_Pgen[t] == 0
        else:
            return model.WTG_Pgen[t] == Gen_Pred_DM[t]
    model.WTG_Pgen_set = pyo.Constraint(model.time, rule=WTG_Pgen_set_Rule)


    # Limiting output power
    def PCC_pmax_rule(model, t):  # Net power in pu for inverter curved
        return model.WTG_Psold[t] + model.ESS_D[t] <= Inv_Pnom

    model.PCC_pmax = pyo.Constraint(model.time, rule=PCC_pmax_rule)

    # Implementing variable efficiency model if enabled
    if Config['Variable Efficiency']:
        # Inverter curve
        Inv_curve_P = [0, 0.06, 0.06, 0.08, 0.08, 0.12, 0.12, 0.16, 0.16, 0.295, 0.295, 0.6945, 0.6945, 1]
        Inv_curve_Eff = [0.53, 0.53, 0.91, 0.91, 0.945, 0.945, 0.9649, 0.9649, 0.9702, 0.9702, 0.95, 0.95,
                         0.9250, 0.9250]

        def Ppu_rule(model, t):  # Net power in pu for inverter curved
            return model.Inv_Ppu[t] == (model.WTG_Psold[t] + model.ESS_C[t] + model.ESS_D[t] / Inv_Pnom)

        model.InvEff = pyo.Constraint(model.time, rule=Ppu_rule)

        # Inverter efficiency piecewise constraint
        model.eff = pyo.Piecewise(model.time, model.Inv_Eff, model.Inv_Ppu,  # Inverter efficiency function
                              pw_pts=Inv_curve_P,
                              pw_repn='INC',
                              pw_constr_type='EQ',
                              f_rule=Inv_curve_Eff,
                              unbounded_domain_var=True)
        # Modified constraints for variable efficiency
        def ESS_Eff_rule(model, t):  # Combines ESS & PCC inverter efficiencies
            return model.ESSInv_Eff[t] == model.Inv_Eff[t] * ESS_Eff

        model.ESS_Eff = pyo.Constraint(model.time, rule=ESS_Eff_rule)

        def ESS_SOCt_Rule(model, t):  # The E must be the result of (E + charge*eff - discharge/eff)
            if t == 0:
                soc_prev = ESS_SOCi * ESS_Emax / 100
            else:
                soc_prev = model.E[t - 1]
            return model.E[t] == soc_prev + \
                   (model.ESS_C[t] * ESS_Eff + model.ESS_Purch[t] * model.ESSInv_Eff[t]) \
                   - model.ESS_D[t] / model.ESSInv_Eff[t]
            # (model.ESS_C[t] + model.ESS_Purch[t]) * ESS_Eff - model.ESS_D[t] / ESS_Eff
        model.ESS_SOCt = pyo.Constraint(model.time, rule=ESS_SOCt_Rule)

        def ESS_Pflow_Rule(model, t):  # Plant power balance constraint
            return model.WTG_Pgen[t] == model.WTG_Psold[t] / model.Inv_Eff[t] + model.ESS_C[t]
        model.ESS_Pflow = pyo.Constraint(model.time, rule=ESS_Pflow_Rule)

    if not Config['Variable Efficiency']:
        def Inv_Constant_Eff_Rule(model, t):
            return model.Inv_Eff[t] == 1
        model.Inv_Constant_Eff = pyo.Constraint(model.time, rule=Inv_Constant_Eff_Rule)

        # Modified constraints for variable efficiency
        def ESS_Eff_rule(model, t):  # Combines ESS & PCC inverter efficiencies
            return model.ESSInv_Eff[t] == model.Inv_Eff[t] * ESS_Eff

        model.ESS_Eff = pyo.Constraint(model.time, rule=ESS_Eff_rule)

        def ESS_SOCt_Rule(model, t):  # The E must be the result of (E + charge*eff - discharge/eff)
            if t == 0:
                soc_prev = ESS_SOCi * ESS_Emax / 100
            else:
                soc_prev = model.E[t - 1]
            return model.E[t] == soc_prev + \
            (model.ESS_C[t] + model.ESS_Purch[t]) * ESS_Eff - model.ESS_D[t] / ESS_Eff
        model.ESS_SOCt = pyo.Constraint(model.time, rule=ESS_SOCt_Rule)

        def ESS_Pflow_Rule(model, t):  # Plant power balance constraint
            return model.WTG_Pgen[t] == model.WTG_Psold[t] + model.ESS_C[t]

        model.ESS_Pflow = pyo.Constraint(model.time, rule=ESS_Pflow_Rule)
    # Implementing degradation model if enabled
    if Config['Degradation']:
        # Degradation model
        DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
        deg_cost_per_cycle = [0., ESS_Cost / 1000000., ESS_Cost / 200000., ESS_Cost / 60000., ESS_Cost / 40000.,
                              ESS_Cost / 20000., ESS_Cost / 15000., ESS_Cost / 11000., ESS_Cost / 10000.,
                              ESS_Cost / 8000., ESS_Cost / 7000., ESS_Cost / 6000.]
        model.deg = pyo.Piecewise(model.deg_cost, model.DOD,  # range and domain variables
                                  pw_pts=DOD_index,
                                  pw_constr_type='EQ',
                                  f_rule=deg_cost_per_cycle,
                                  pw_repn='INC')

        def EN_rule(model):
            return sum((model.ESS_D[t1] + model.ESS_C[t1]) / 2. * (100 // ESS_Emax)
                       for t1 in model.time)

        model.EN = pyo.Expression(rule=EN_rule)  # Half of the total energy throughput in %

        model.DOD1 = pyo.Var(bounds=(0, 100))

        def DOD1_rule(model):
            return model.DOD1 >= model.EN - model.DOD

        model.DOD1_con = pyo.Constraint(rule=DOD1_rule)
        model.deg_cost1 = pyo.Var(domain=pyo.NonNegativeReals)
        model.deg1 = pyo.Piecewise(model.deg_cost1, model.DOD1,  # range and domain variables
                                   pw_pts=DOD_index,
                                   pw_constr_type='EQ',
                                   f_rule=deg_cost_per_cycle,
                                   pw_repn='INC')

        # Objective Function: Maximize profitability and reduce degradation costs
        model.obj = pyo.Objective(
            expr=sum((Price_Pred_DM[t] * (model.WTG_Psold[t] + model.ESS_D[t] - model.ESS_Purch[t]))
                     for t in model.time) - model.deg_cost - model.deg_cost1, sense=pyo.maximize)
    if not Config['Degradation']:
        # Objective Function: Maximize profitability
        model.obj = pyo.Objective(
            expr=sum((Price_Pred_DM[t] * (model.WTG_Psold[t] + model.ESS_D[t] - model.ESS_Purch[t]))
                     for t in model.time), sense=pyo.maximize)

    # Applying the solver
    if HyF_Parameters['Config']['Variable Efficiency']: # Switching to nonlinear solver if efficiency is variable
        opt = SolverFactory('ipopt')
    if not HyF_Parameters['Config']['Variable Efficiency']:
        opt = SolverFactory('cbc')
    opt.solve(model)
    # model.pprint()

    # Extracting data from model
    Gen_Pred_DM = [model.WTG_Pgen[t]() for t in model.time]
    WTG_Psold = [model.WTG_Psold[t]() for t in model.time]
    ESS_C = [model.ESS_C[t]() for t in model.time]
    ESS_D = [model.ESS_D[t]() for t in model.time]
    ESS_P = [model.ESS_Purch[t]() for t in model.time]
    SOC_E = [model.E[t]() for t in model.time]
    SOC_E.insert(0, ESS_SOCi * (ESS_Emax / 100))
    SOC = [i * (100 / ESS_Emax) for i in SOC_E]

    # Clearing Nonetypes and switching sings of charging powers
    for i in range(len(Price_Pred_DM)):
        if ESS_D[i] is None:
            ESS_D[i] = 0
        if ESS_C[i] is None:
            ESS_C[i] = 0
        else:
            ESS_C[i] = - ESS_C[i]
        if ESS_P[i] is None:
            ESS_P[i] = 0
        else:
            ESS_P[i] = - ESS_P[i]

    return Gen_Pred_DM, WTG_Psold, ESS_C, ESS_D, ESS_P, SOC

#%% Intraday market bidding
def ID(HyF_Parameters, Gen_Pred, P_PCC_toDel, Price_Real_DM, Price_Pred_ID,
       Dev_Costs_Down, Dev_Costs_Up, ESS_P_Prev, P_SOCdump):
    # Unpacking hybrid farm parameters dictionary
    ESS_Emax = HyF_Parameters['ESS Capacity']
    ESS_Pmax = HyF_Parameters['ESS Nominal Power']
    ESS_Eff = HyF_Parameters['ESS Efficiency']
    ESS_Cost = HyF_Parameters['ESS Replacement Cost']
    ESS_SOCi = HyF_Parameters['ESS Initial SOC']
    Config = HyF_Parameters['Config']
    # Model initialization
    model = pyo.ConcreteModel()
    # Model variables
    model.time = pyo.RangeSet(0, len(Price_Pred_ID)-1)
    model.E = pyo.Var(model.time, bounds=(0, ESS_Emax),
                      initialize=0)                                             # ESS energy at the end of the hour
    model.charging = pyo.Var(model.time, domain=pyo.Binary)                     # Charge verifier
    model.discharging = pyo.Var(model.time, domain=pyo.Binary)                  # Discharge verifier
    model.ESS_C = pyo.Var(model.time, bounds=(0, ESS_Pmax), initialize=0)       # Energy charged during period
    model.ESS_D = pyo.Var(model.time, bounds=(0, ESS_Pmax), initialize=0)       # Energy disrcharged for deviations
    model.ESS_S = pyo.Var(model.time, bounds=(0, ESS_Pmax), initialize=0)    # Energy sold by ESS during period
    model.ID_Purch = pyo.Var(model.time, bounds=(0, None), initialize=0)        # Energy purchased on ID during period
    model.WTG_Psold = pyo.Var(model.time, bounds=(0, None),
                              initialize=0)                                     # Generated power directly sold
    model.WTG_Pdel = pyo.Var(model.time, bounds=(0, None),
                              initialize=0)                                     # Generated power directly delivered
    model.WTG_Pgen = pyo.Var(model.time, initialize=0)                          # Generated power
    model.Pcurt = pyo.Var(model.time, bounds=(0, None), initialize=0)           # Curtailed power
    model.PCC = pyo.Var(model.time, initialize=0)                               # PCC power delivered
    Gen_Pred = np.array(Gen_Pred)
    model.Dev_Cost = pyo.Var(model.time, initialize=0)                          # Hourly deviation costs
    model.Dev_Up = pyo.Var(model.time, bounds=(0, None),
                           initialize=0)                                        # Hourly deviation up
    model.Dev_Down = pyo.Var(model.time, bounds=(0, None),
                           initialize=0)                                        # Hourly deviation down
    model.DOD = pyo.Var(bounds=(0, 100))  # ESS max DOD
    model.deg_cost = pyo.Var()  # Degradation costs
    model.max_SOC = pyo.Var(bounds=(ESS_SOCi, 100))  # ESS max SOC
    model.min_SOC = pyo.Var(bounds=(0, ESS_SOCi))  # ESS min SOC

    # Keeping ESS previous purchase array values positive
    for i in range(len(ESS_P_Prev)):
        ESS_P_Prev[i] = abs(ESS_P_Prev[i])

    # Model constraints
    def ESS_Pcha_max_Rule(model, t):  # Forces limit on charging power
        return (ESS_Pmax * model.charging[t]) >= model.ESS_C[t] + ESS_P_Prev[t]
    model.ESS_Pcha_max = pyo.Constraint(model.time, rule=ESS_Pcha_max_Rule)

    def ESS_Pdis_max_Rule(model, t):  # Forces limit on discharging power
        return (ESS_Pmax * model.discharging[t]) >= model.ESS_D[t] + model.ESS_S[t]
    model.ESS_Pdis_max = pyo.Constraint(model.time, rule=ESS_Pdis_max_Rule)

    def ESS_cha_dis_Rule(model, t):  # Prevents orders of charge and discharge simultaneously
        return (model.charging[t] + model.discharging[t]) <= 1
    model.ESS_cha_dis = pyo.Constraint(model.time, rule=ESS_cha_dis_Rule)

    def ESS_SOCt_Rule(model, t):  # The E must be the result of (E + charge*eff - discharge/eff)
        if t == 0:
            E_prev = ESS_SOCi * ESS_Emax / 100
        else:
            E_prev = model.E[t - 1]
        return model.E[t] == E_prev + \
                (model.ESS_C[t] + ESS_P_Prev[t]) * ESS_Eff - (model.ESS_D[t] + model.ESS_S[t]) / ESS_Eff
    model.ESS_SOCt = pyo.Constraint(model.time, rule=ESS_SOCt_Rule)

    def Dev_Up_Rule(model,t):
        return model.Dev_Up[t] == model.PCC[t] - P_PCC_toDel[t]
    model.Dev_Up_r = pyo.Constraint(model.time, rule=Dev_Up_Rule)

    def Dev_Down_Rule(model,t):
        return model.Dev_Down[t] == P_PCC_toDel[t] - model.PCC[t]
    model.Dev_Down_r = pyo.Constraint(model.time, rule=Dev_Down_Rule)

    def Dev_Cost_Rule(model,t):
        return model.Dev_Cost[t] == Price_Real_DM[t] * \
               (model.Dev_Down[t] * Dev_Costs_Down[t] - model.Dev_Up[t] * Dev_Costs_Up[t]) / 100
    model.Dev_Cost_r = pyo.Constraint(model.time, rule=Dev_Cost_Rule)

    def WTG_Pgen_set_Rule(model, t):      # Storing generted power on a variable
        if Gen_Pred.size == 0 or len(Gen_Pred) < len(Price_Pred_ID):
            return model.WTG_Pgen[t] == 0
        else:
            return model.WTG_Pgen[t] == Gen_Pred[t]
    model.WTG_Pgen_set = pyo.Constraint(model.time, rule=WTG_Pgen_set_Rule)

    def ESS_Pflow_Rule(model, t):        # Plant power balance constraint
        return model.WTG_Pgen[t] == model.WTG_Pdel[t] + model.WTG_Psold[t] \
               + model.ESS_C[t] + model.Pcurt[t]
    model.ESS_Pflow = pyo.Constraint(model.time, rule=ESS_Pflow_Rule)

    def PCC_Rule(model,t):              # PCC constraint, used for deviation control
        return model.PCC[t] == model.WTG_Pdel[t] + model.ESS_D[t]  \
               + model.ID_Purch[t] - ESS_P_Prev[t]
    model.PCC_r = pyo.Constraint(model.time, rule=PCC_Rule)

    def ESS_SOCmax_Rule(model, t):
        return model.max_SOC >= model.E[t] * (100 // ESS_Emax)
    model.ESS_SOCmax = pyo.Constraint(model.time, rule=ESS_SOCmax_Rule)

    def ESS_SOCmin_Rule(model, t):
        return model.min_SOC <= model.E[t] * (100 // ESS_Emax)
    model.ESS_SOCmin = pyo.Constraint(model.time, rule=ESS_SOCmin_Rule)

    def ESS_DODmax_Rule(model):
        return model.DOD == model.max_SOC - model.min_SOC
    model.ESS_DODmax = pyo.Constraint(rule=ESS_DODmax_Rule)

    # Disabling arbitrage variables: Generated power sold and wind turbine
    if not Config['ID Arbitrage']:
        def Dis_Psold_Rule(model, t):
            return model.WTG_Psold[t] == 0
        model.Dis_Psold = pyo.Constraint(model.time, rule=Dis_Psold_Rule)

        def Dis_ESSSold_Rule(model, t):
            return model.ESS_S[t] == 0
        model.Dis_ESSSold = pyo.Constraint(model.time, rule=Dis_ESSSold_Rule)

    if Config['Degradation']:
        # Degradation model
        DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
        deg_cost_per_cycle = [0., ESS_Cost / 1000000., ESS_Cost / 200000., ESS_Cost / 60000., ESS_Cost / 40000.,
                              ESS_Cost / 20000., ESS_Cost / 15000., ESS_Cost / 11000., ESS_Cost / 10000.,
                              ESS_Cost / 8000., ESS_Cost / 7000., ESS_Cost / 6000.]
        model.deg = pyo.Piecewise(model.deg_cost, model.DOD,  # range and domain variables
                                  pw_pts=DOD_index,
                                  pw_constr_type='EQ',
                                  f_rule=deg_cost_per_cycle,
                                  pw_repn='INC')

        def EN_rule(model):
            return sum((model.ESS_D[t1] + model.ESS_C[t1]) / 2. * (100 // ESS_Emax)
                       for t1 in model.time)

        model.EN = pyo.Expression(rule=EN_rule)  # Half of the total energy throughput in %

        model.DOD1 = pyo.Var(bounds=(0, 100))

        def DOD1_rule(model):
            return model.DOD1 >= model.EN - model.DOD

        model.DOD1_con = pyo.Constraint(rule=DOD1_rule)
        model.deg_cost1 = pyo.Var(domain=pyo.NonNegativeReals)
        model.deg1 = pyo.Piecewise(model.deg_cost1, model.DOD1,  # range and domain variables
                                   pw_pts=DOD_index,
                                   pw_constr_type='EQ',
                                   f_rule=deg_cost_per_cycle,
                                   pw_repn='INC')
        # Objective Function: Maximize profitability reducing deviation costs & degradation costs
        model.obj = pyo.Objective(
            expr=sum(Price_Pred_ID[t] * (model.WTG_Psold[t]  + model.ESS_S[t] - model.ID_Purch[t])
                     - model.Dev_Cost[t] - model.deg_cost1 - model.deg_cost
                     for t in model.time), sense=pyo.maximize)

    if not Config['Degradation']:
        # Objective Function: Maximize profitability reducing deviation costs
        model.obj = pyo.Objective(
            expr=sum(Price_Pred_ID[t] * (model.WTG_Psold[t]  + model.ESS_S[t] - model.ID_Purch[t])
                     - model.Dev_Cost[t] for t in model.time), sense=pyo.maximize)

    # Applying the solver
    opt = SolverFactory('cbc')
    opt.solve(model)

    # Debugging tools
    # model.pprint()
    # filename = os.path.join(os.path.dirname(__file__), 'model.lp')
    # model.write(filename, io_options={'symbolic_solver_labels': True})

    # Extracting data from model
    WTG_Psold = [model.WTG_Psold[t]() for t in model.time]
    WTG_Pdel = [model.WTG_Psold[t]() for t in model.time]
    ID_Purch = [model.ID_Purch[t]() for t in model.time]
    ESS_C = [model.ESS_C[t]() for t in model.time]
    ESS_D = [model.ESS_D[t]() for t in model.time]
    ESS_S = [model.ESS_S[t]() for t in model.time]
    SOC_E = [model.E[t]() for t in model.time]
    SOC_E.insert(0, ESS_SOCi * (ESS_Emax / 100))
    SOC = [i * (100 / ESS_Emax) for i in SOC_E]

    return Gen_Pred, WTG_Psold, WTG_Pdel, [-a for a in ESS_C], ESS_D, [-a for a in ESS_P_Prev], ESS_S, SOC, ID_Purch

#%% Dumping SOC excess on next ID market
def SOCdump_ID(HyF_Parameters, Price_Pred_ID, ID):
    # Unpacking hybrid farm parameters dictionary
    if HyF_Parameters['Config']['SOC Dump']:
        ESS_Emax = HyF_Parameters['ESS Capacity']
    else:
        ESS_Emax = 0.0000001
    ESS_Pmax = HyF_Parameters['ESS Nominal Power']
    ESS_Eff = HyF_Parameters['ESS Efficiency']
    ESS_Cost = HyF_Parameters['ESS Replacement Cost']
    Config = HyF_Parameters['Config']
    # Setting available SOC for dumping, avoiding negative values
    ESS_SOCi = max(HyF_Parameters['ESS Initial SOC'] - HyF_Parameters['ESS dumping SOC'], 0)
    # Setting time window until next ID
    if ID == 2:
        dump_len = 4
    if ID == 3:
        dump_len = 3
    if ID == 4:
        dump_len = 5
    if ID == 5:
        dump_len = 3
    if ID == 6:
        dump_len = len(Price_Pred_ID)
    # Slicing ID prices
    dump_prices = Price_Pred_ID[:dump_len]
    # Model initialization
    model = pyo.ConcreteModel()
    # Model variables
    model.time = pyo.RangeSet(0, dump_len - 1)
    model.E = pyo.Var(model.time, bounds=(0, ESS_Emax),
                      initialize=0)  # ESS energy at the end of the hour
    model.ESS_D = pyo.Var(model.time, bounds=(0, ESS_Pmax))  # Energy being discharged during period
    model.DOD = pyo.Var(bounds=(0, 100))  # ESS max DOD
    model.deg_cost = pyo.Var()  # Degradation costs
    model.max_SOC = pyo.Var(bounds=(ESS_SOCi, 100))  # ESS max SOC
    model.min_SOC = pyo.Var(bounds=(0, ESS_SOCi))  # ESS min SOC

    def ESS_SOCt_Rule(model, t):  # The E must be the result of (E + charge*eff - discharge/eff)
        if t == 0:
            soc_prev = ESS_SOCi * ESS_Emax / 100
        else:
            soc_prev = model.E[t - 1]
        return model.E[t] == soc_prev - model.ESS_D[t] / ESS_Eff

    model.ESS_SOCt = pyo.Constraint(model.time, rule=ESS_SOCt_Rule)

    def ESS_SOC_end_Rule(model):
        return model.E[dump_len - 1] == 0.0

    model.ESS_SOC_end = pyo.Constraint(rule=ESS_SOC_end_Rule)

    def ESS_SOCmax_Rule(model, t):
        return model.max_SOC >= model.E[t] * (100 // ESS_Emax)

    model.ESS_SOCmax = pyo.Constraint(model.time, rule=ESS_SOCmax_Rule)

    def ESS_SOCmin_Rule(model, t):
        return model.min_SOC <= model.E[t] * (100 // ESS_Emax)

    model.ESS_SOCmin = pyo.Constraint(model.time, rule=ESS_SOCmin_Rule)

    def ESS_DODmax_Rule(model):
        return model.DOD == model.max_SOC - model.min_SOC

    model.ESS_DODmax = pyo.Constraint(rule=ESS_DODmax_Rule)

    # Degradation model
    if Config['Degradation']:
        DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
        deg_cost_per_cycle = [0., ESS_Cost / 1000000., ESS_Cost / 200000., ESS_Cost / 60000., ESS_Cost / 40000.,
                              ESS_Cost / 20000., ESS_Cost / 15000., ESS_Cost / 11000., ESS_Cost / 10000.,
                              ESS_Cost / 8000., ESS_Cost / 7000., ESS_Cost / 6000.]
        model.deg = pyo.Piecewise(model.deg_cost, model.DOD,  # range and domain variables
                                  pw_pts=DOD_index,
                                  pw_constr_type='EQ',
                                  f_rule=deg_cost_per_cycle,
                                  pw_repn='INC')

        def EN_rule(model):
            return sum(model.ESS_D[t1] / 2. * (100 // ESS_Emax)
                       for t1 in model.time)

        model.EN = pyo.Expression(rule=EN_rule)  # Half of the total energy throughput in %

        model.DOD1 = pyo.Var(bounds=(0, 100))

        def DOD1_rule(model):
            return model.DOD1 >= model.EN - model.DOD

        model.DOD1_con = pyo.Constraint(rule=DOD1_rule)
        model.deg_cost1 = pyo.Var(domain=pyo.NonNegativeReals)
        model.deg1 = pyo.Piecewise(model.deg_cost1, model.DOD1,  # range and domain variables
                                   pw_pts=DOD_index,
                                   pw_constr_type='EQ',
                                   f_rule=deg_cost_per_cycle,
                                   pw_repn='INC')

        # Objective Function: Sell energy surplus at maximum price reducing degradation costs
        model.obj = pyo.Objective(
            expr=sum(dump_prices[t] * model.ESS_D[t] for t in model.time) - model.deg_cost - model.deg_cost1,
            sense=pyo.maximize)
    if not Config['Degradation']:
        # Objective Function: Sell energy surplus at maximum price
        model.obj = pyo.Objective(
            expr=sum(dump_prices[t] * model.ESS_D[t] for t in model.time), sense=pyo.maximize)

    # Applying the solver
    opt = SolverFactory('cbc')
    opt.solve(model)
    # model.pprint()

    # Extracting data from model
    ESS_D = [model.ESS_D[t]() for t in model.time]

    # Clearing Nonetypes and switching sings of charging powers
    for i in range(dump_len):
        if ESS_D[i] is None:
            ESS_D[i] = 0

    # Adapting ESS_D array length for current ID
    while len(ESS_D) != len(Price_Pred_ID):
        ESS_D.append(0)

    # print(f'Dumping on ID {ID}:')
    # print(f'\t - Dumps: {ESS_D}')
    # print(f'\t - Dumps length: {len(ESS_D)}')
    # print(f'\t - Dumping time window: {dump_len}')

    return ESS_D