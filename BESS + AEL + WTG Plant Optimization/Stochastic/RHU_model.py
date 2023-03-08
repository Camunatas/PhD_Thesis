#%% Load python libraries
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np

#%% Get RHU parameters
def get_RHU_parameters():
    RHU_Parameters = {}
    # RHU configuration
    RHU_Parameters['Degradation'] = False  # Degradation mode enabler
    RHU_Parameters['Arbitrage'] = True  # Arbitrage service enabler
    RHU_Parameters['Purging'] = True  # AEL purging enabler
    # AEL parameters
    RHU_Parameters['AEL Maximum power'] = 1  # [MW]
    RHU_Parameters['AEL Minimum power'] = 0.3  # [MW]
    RHU_Parameters['AEL efficiency'] = 0.75  # [p.u.]
    RHU_Parameters['Tank capacity'] = 1e100  # [kg]
    RHU_Parameters['Initial hydrogen level'] = 0  # [kg]
    RHU_Parameters['AEL investment cost'] = 500 * RHU_Parameters['AEL Maximum power']  # [€/kW]
    RHU_Parameters['Lifetime hours'] = 100e3  # Operating ours before EOL
    RHU_Parameters['Lifetime cycles'] = 5000  # Start/stop cycles before EOL
    RHU_Parameters['Initial state'] = 0  # Initial state (0= cold, 5 = hot)
    RHU_Parameters['Cold start time'] = 0.2  # [h]
    RHU_Parameters['Off time'] = 5  # [h]
    RHU_Parameters['HHV'] = 0.0394  # [kg*MWh] Hydrogen High Heating Value
    RHU_Parameters['Idle state power'] = 0.04  # [MW] Idle state power consumption
    RHU_Parameters['Off state power'] = 0.000  # [MW] Off state power consumption
    RHU_Parameters['AEL initial H2 in O2'] = 0  # [%] Initial H2 concentration in O2
    # BESS paremeters
    RHU_Parameters['Batt_SOCi'] = 0  # [p.u.] Initial SOC
    RHU_Parameters['Batt_E'] = 10  # [MWh] Default Battery Capacity
    RHU_Parameters['Batt_P'] = 2.5  # [MW] Default Battery Power
    RHU_Parameters['Batt_Eff'] = 0.9  # [p.u.] Default Battery Efficiency
    RHU_Parameters['Batt_Cost'] = 50 * RHU_Parameters['Batt_E']  # [€/kWh] Default Battery Cost
    RHU_Parameters['Batt_EOL'] = 0.8  # [p.u.] BESS capacity at EOL
    # Degradation models regulators
    RHU_Parameters['K_AEL'] = 1
    RHU_Parameters['K_BESS'] = 0.5
    # Market parameters models regulators
    RHU_Parameters['Dev_coef'] = 0.5  # [p.u.] Deviation cost coefficient

    return RHU_Parameters

#%% RHU model for day-ahead market bidding
def RHU_DM(RHU_Parameters, P_gen, Price_El, Price_H2):
    # Unpack parameters
    AEL_Pmax = RHU_Parameters['AEL Maximum power']
    AEL_Eff = RHU_Parameters['AEL efficiency']
    AEL_Cap = RHU_Parameters['Tank capacity']
    AEL_initial_H = RHU_Parameters['Initial hydrogen level']
    AEL_cost = RHU_Parameters['AEL investment cost'] * 1000 * RHU_Parameters['K_AEL']
    AEL_cycles = RHU_Parameters['Lifetime cycles']
    AEL_hours = RHU_Parameters['Lifetime hours']
    AEL_state_i = RHU_Parameters['Initial state']
    AEL_t_start = RHU_Parameters['Cold start time']
    AEL_t_off = RHU_Parameters['Off time']
    AEL_P_on_min = RHU_Parameters['AEL Minimum power']
    AEL_HHV = RHU_Parameters['HHV']
    AEL_P_idle = RHU_Parameters['Idle state power']
    AEL_P_off = RHU_Parameters['Off state power']
    AEL_initial_imp = RHU_Parameters['AEL initial H2 in O2']
    deg_model = RHU_Parameters['Degradation']
    BESS_SOCi = RHU_Parameters['Batt_SOCi']
    BESS_capacity = RHU_Parameters['Batt_E']
    BESS_maxpower = RHU_Parameters['Batt_P']
    BESS_efficiency = RHU_Parameters['Batt_Eff']
    BESS_cost = RHU_Parameters['Batt_Cost'] * 1000 * RHU_Parameters['K_AEL']

    # Initialise model
    model = pyo.ConcreteModel()
    model.time = pyo.RangeSet(0, min(len(Price_El), len(P_gen)) - 1)

    # AEL variables
    model.H_AEL = pyo.Var(model.time, bounds=(0, AEL_Cap), initialize=0)            # Stored hydrogen
    model.P_AEL = pyo.Var(model.time, bounds=(0, AEL_Pmax), initialize=0)           # AEL power
    model.P_on_AEL = pyo.Var(model.time, bounds=(0, AEL_Pmax), initialize=0)        # AEL 'On' mode power
    model.P_idle_AEL = pyo.Var(model.time, bounds=(0, AEL_P_idle), initialize=0)    # AEL 'Idle' mode power
    model.P_off_AEL = pyo.Var(model.time, bounds=(0, AEL_P_off), initialize=0)      # AEL 'Off' mode power
    model.P_start_AEL = pyo.Var(model.time, bounds=(0, AEL_Pmax), initialize=0)     # AEL cold start power
    model.Q_AEL = pyo.Var(model.time, initialize=0)                                 # AEL hydrogen production
    model.off_AEL = pyo.Var(model.time, domain=pyo.Binary)                          # AEL 'Off' mode verifier
    model.on_AEL = pyo.Var(model.time, domain=pyo.Binary)                           # AEL 'On' mode verifier
    model.idle_AEL = pyo.Var(model.time, domain=pyo.Binary)                         # AEL 'Idle' mode verifier
    model.cool_AEL = pyo.Var(model.time, domain=pyo.Binary)                         # AEL 'Cool' mode verifier
    model.state_AEL = pyo.Var(model.time, bounds=(0, AEL_t_off))                    # AEL state at the end of period
    model.on_frac_AEL = pyo.Var(model.time, bounds=(0, 1))                          # AEL start power fraction
    model.imp_AEL = pyo.Var(model.time, bounds=(0, 2), initialize=0)                # AEL O2 impurity
    model.cont_AEL = pyo.Var(model.time, bounds=(0, 2), initialize=0)               # AEL O2 contamination process
    model.purg_AEL = pyo.Var(model.time, domain=pyo.Binary)                         # AEL purging mode verifier
    # BESS variables
    model.E_BESS = pyo.Var(model.time, bounds=(0, BESS_capacity),
                        initialize=0)                                               # BESS stored energy at the end of t
    model.charging_BESS = pyo.Var(model.time, domain=pyo.Binary)                    # BESS charge verifier
    model.discharging_BESS = pyo.Var(model.time, domain=pyo.Binary)                 # BESS discharge verifier
    model.P_C_BESS = pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # BESS charging power during period
    model.P_D_BESS = pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # BESS discharging power during period
    model.DOD_BESS = pyo.Var(bounds=(0, 100))
    model.deg_cost_BESS = pyo.Var()                                                 # BESS daily degradation cost
    model.max_SOC_BESS = pyo.Var(bounds=(BESS_SOCi, 100))                           # BESS maximum SOC
    model.min_SOC_BESS = pyo.Var(bounds=(0, BESS_SOCi))                             # BESS minimum SOC
    # System variables
    model.P_WTG_BESS= pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # Power from WTG to BESS
    model.P_WTG_AEL= pyo.Var(model.time, bounds=(0, AEL_Pmax),
                             initialize=0)                                          # Power from WTG to AEL
    model.P_WTG_Grid= pyo.Var(model.time, bounds=(0, max(P_gen)),
                             initialize=0)                                          # Power from WTG to Grid
    model.P_WTG_Curt= pyo.Var(model.time, bounds=(0, max(P_gen)),
                             initialize=0)                                           # Power from WTG to curtailment
    model.P_BESS_Grid= pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # Power from BESS to Grid
    model.P_BESS_AEL= pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # Power from BESS to AEL
    model.P_Grid_AEL= pyo.Var(model.time, bounds=(0, AEL_Pmax),
                             initialize=0)                                          # Power from Grid to AEL
    model.P_Grid_BESS= pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # Power from Grid to BESS

    # AEL constraints
    def P_rule_AEL(model, t):  # AEL power at each time step is the sum of the different states
        return model.P_on_AEL[t] + model.P_off_AEL[t] + model.P_idle_AEL[t] + model.P_start_AEL[t] == model.P_AEL[t]
    model.P_rule_AEL = pyo.Constraint(model.time, rule=P_rule_AEL)

    def state_rule_AEL(model, t):  # Only one state in the AEL is allowed at the same time
        return model.on_AEL[t] + model.off_AEL[t] + model.idle_AEL[t] == 1
    model.state_rule_AEL = pyo.Constraint(model.time, rule=state_rule_AEL)

    def cool_state_rule_AEL(model, t):  # Cooling state defined when off
        return model.cool_AEL[t] <= model.off_AEL[t]
    model.cool_state_rule_AEL = pyo.Constraint(model.time, rule=cool_state_rule_AEL)

    def cool_rule1_AEL(model, t):  # AEL cooling stops when fully cold
        if t == 0:
            return model.cool_AEL[t] <= AEL_state_i / AEL_t_off + 0.9
        else:
            return model.cool_AEL[t] <= model.state_AEL[t - 1] / AEL_t_off + 0.9
    model.cool_rule1_AEL = pyo.Constraint(model.time, rule=cool_rule1_AEL)

    def cool_rule2_AEL(model, t):  # AEL cooling stops when fully cold
        if t == 0:
            return model.cool_AEL[t] >= AEL_state_i / AEL_t_off - 1 + model.off_AEL[t]
        else:
            return model.cool_AEL[t] >= model.state_AEL[t - 1] / AEL_t_off - 1 + model.off_AEL[t]
    model.cool_rule2_AEL = pyo.Constraint(model.time, rule=cool_rule2_AEL)

    def state_updater_rule_AEL(model, t):  # AEL state updater
        if t == 0:
            return model.state_AEL[t] == AEL_state_i + \
                   model.on_frac_AEL[t] * AEL_t_off - model.cool_AEL[t]
        else:
            return model.state_AEL[t] == model.state_AEL[t - 1] + \
                   model.on_frac_AEL[t] * AEL_t_off - model.cool_AEL[t]
    model.state_updater_rule_AEL = pyo.Constraint(model.time, rule=state_updater_rule_AEL)

    def off_rule_AEL(model, t):  # AEL 'Off' mode power enabler
        return model.P_off_AEL[t] == AEL_P_off * model.off_AEL[t]
    model.off_rule_AEL = pyo.Constraint(model.time, rule=off_rule_AEL)

    def idle_rule_AEL(model, t):  # AEL 'Idle' power enabler
        return model.P_idle_AEL[t] == AEL_P_idle * model.idle_AEL[t]
    model.idle_rule_AEL = pyo.Constraint(model.time, rule=idle_rule_AEL)

    def on_max_rule_AEL(model, t):  # AEL maximum power at 'On' mode
        return model.P_on_AEL[t] <= AEL_Pmax * model.on_AEL[t]
    model.on_max_rule_AEL = pyo.Constraint(model.time, rule=on_max_rule_AEL)

    def on_min_rule_AEL(model, t):  # AEL minimum power for H2 production
        return model.P_on_AEL[t] >= AEL_P_on_min * model.on_AEL[t]
    model.on_min_rule_AEL = pyo.Constraint(model.time, rule=on_min_rule_AEL)

    def on_conditions_rule_AEL(model, t):  # Only can be idle or on in nominal conditions
        return model.on_AEL[t] + model.idle_AEL[t] <= (model.state_AEL[t]) / AEL_t_off
    model.on_conditions_rule_AEL = pyo.Constraint(model.time, rule=on_conditions_rule_AEL)

    def Pstart_rule_AEL(model, t):  # AEL cold start power calculation
        return model.P_start_AEL[t] == model.on_frac_AEL[t] * AEL_t_start * AEL_Pmax
    model.Pstart_rule_AEL = pyo.Constraint(model.time, rule=Pstart_rule_AEL)

    def H_max_rule_AEL(model, t):  # AEL tank capacity
        return model.H_AEL[t] <= AEL_Cap
    model.Hmax_rule_AEL = pyo.Constraint(model.time, rule=H_max_rule_AEL)

    def Q_rule_AEL(model, t):  # AEL hydrogen generation
        return model.Q_AEL[t] == model.P_on_AEL[t] * AEL_Eff / AEL_HHV
    model.Q_rule_AEL = pyo.Constraint(model.time, rule=Q_rule_AEL)

    def state_factor_rule_AEL(model, t):  # AEL starting power fraction constraint
        if t == 0:
            state_prev = AEL_state_i
        else:
            state_prev = model.state_AEL[t - 1]
        return model.on_frac_AEL[t] <= (AEL_t_off - state_prev) / AEL_t_off
    model.state_factor_rule_AEL = pyo.Constraint(model.time, rule=state_factor_rule_AEL)

    def off_mode_rule_AEL(model, t):  # Starting power factor can't be over 1
        return model.on_frac_AEL[t] <= model.off_AEL[t] + model.idle_AEL[t]
    model.off_mode_rule_AEL = pyo.Constraint(model.time, rule=off_mode_rule_AEL)

    def H_rule_AEL(model, t):  # AEL stored hydrogen rule
        if t == 0:
            H_prev = AEL_initial_H
        else:
            H_prev = model.H_AEL[t - 1]
        return model.H_AEL[t] == H_prev + model.Q_AEL[t]
    model.H_rule_AEL = pyo.Constraint(model.time, rule=H_rule_AEL)

    if RHU_Parameters['Purging']:
        def imp_state_rule(model, t):  # Impurity state update
            if t == 0:
                imp_prev = AEL_initial_imp
            else:
                imp_prev = model.imp_AEL[t - 1]
            return model.imp_AEL[t] == imp_prev + model.cont_AEL[t] - 2 * model.purg_AEL[t]
        model.imp_state_rule = pyo.Constraint(model.time, rule=imp_state_rule)

        def enable_purg_rule(model, t): # Purging when off/idle
            if t == 0:
                return model.purg_AEL[t] == 0
            else:
                return model.purg_AEL[t] <= model.off_AEL[t] + model.idle_AEL[t]
        model.enable_purg_rule = pyo.Constraint(model.time, rule=enable_purg_rule)

        def disable_purg_rule(model, t): # Stopping the purging when fully clean
            if t == 0:
                imp_prev = AEL_initial_imp
            else:
                imp_prev = model.imp_AEL[t - 1]
            return model.purg_AEL[t] <= imp_prev
        model.disable_purg_rule = pyo.Constraint(model.time, rule=disable_purg_rule)

        def shutdown_rule(model, t):  # Shutdown when impurity is dangerous
            if t == 0:
                imp_prev = AEL_initial_imp
            else:
                imp_prev = model.imp_AEL[t - 1]
            return imp_prev - model.off_AEL[t]  <= 1.9
        model.shutdown_rule = pyo.Constraint(model.time, rule=shutdown_rule)

    # O2 contamination per P_on piecewise constraint
    contamination_per_P_on = [0., 2., 1., 0.]
    P_on_index = [0., 0.3/AEL_Pmax, 0.5/ AEL_Pmax, 1]
    model.cont_rule = pyo.Piecewise(model.time, model.cont_AEL, model.P_on_AEL,  # range and domain variables
                              pw_pts=P_on_index,
                              pw_constr_type='EQ',
                              f_rule=contamination_per_P_on,
                              pw_repn='INC')

    # BESS constraints
    def P_ch_rule_BESS(model, t):  # Forces limit on charging power
        return (BESS_maxpower * model.charging_BESS[t]) >= model.P_C_BESS[t]
    model.P_ch_rule_BESS = pyo.Constraint(model.time, rule=P_ch_rule_BESS)

    def P_dis_rule_BESS(model, t):  # Forces limit on discharging power
        return (BESS_maxpower * model.discharging_BESS[t]) >= model.P_D_BESS[t]
    model.P_dis_rule_BESS = pyo.Constraint(model.time, rule=P_dis_rule_BESS)

    def states_rule_BESS(model, t):  # Prevents orders of charge and discharge simultaneously
        return (model.charging_BESS[t] + model.discharging_BESS[t]) <= 1
    model.states_rule_BESS = pyo.Constraint(model.time, rule=states_rule_BESS)

    def E_rule_BESS(model, t):  # The stored energy must be the result of (E + charge*eff - discharge/eff)
        if t == 0:
            E_prev = BESS_SOCi * BESS_capacity / 100
        else:
            E_prev = model.E_BESS[t - 1]
        return model.E_BESS[t] == E_prev + model.P_C_BESS[t] * BESS_efficiency - model.P_D_BESS[t] / BESS_efficiency
    model.E_rule_BESS = pyo.Constraint(model.time, rule=E_rule_BESS)

    def SOC_max_rule_BESS(model, t):
        return model.max_SOC_BESS >= model.E_BESS[t] * (100 // BESS_capacity)
    model.SOC_max_rule_BESS = pyo.Constraint(model.time, rule=SOC_max_rule_BESS)

    def SOC_min_rule_BESS(model, t):
        return model.min_SOC_BESS <= model.E_BESS[t] * (100 // BESS_capacity)
    model.SOC_min_rule_BESS = pyo.Constraint(model.time, rule=SOC_min_rule_BESS)

    def DOD_rule_BESS(model):
        return model.DOD_BESS == model.max_SOC_BESS - model.min_SOC_BESS
    model.DOD_rule_BESS = pyo.Constraint(rule=DOD_rule_BESS)

    # BESS degradation model
    deg_cost_per_cycle = [0., BESS_cost / 1000000., BESS_cost / 200000., BESS_cost / 60000., BESS_cost / 40000.,
                          BESS_cost / 20000., BESS_cost / 15000., BESS_cost / 11000., BESS_cost / 10000.,
                          BESS_cost / 8000., BESS_cost / 7000., BESS_cost / 6000.]

    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
    model.deg = pyo.Piecewise(model.deg_cost_BESS, model.DOD_BESS,  # range and domain variables
                              pw_pts=DOD_index,
                              pw_constr_type='EQ',
                              f_rule=deg_cost_per_cycle,
                              pw_repn='INC')

    def EN_rule_BESS(model):
        return sum((model.P_D_BESS[t1] + model.P_C_BESS[t1]) / 2. * (100 // BESS_capacity)
                   for t1 in model.time)
    model.EN_BESS = pyo.Expression(rule=EN_rule_BESS)  # Half of the total energy throughput in %
    model.DOD1_BESS = pyo.Var(bounds=(0, 100))

    def DOD1_rule_BESS(model):
        return model.DOD1_BESS >= model.EN_BESS - model.DOD_BESS
    model.DOD1_rule_BESS = pyo.Constraint(rule=DOD1_rule_BESS)
    model.deg_cost1_BESS = pyo.Var(domain=pyo.NonNegativeReals)
    model.deg1 = pyo.Piecewise(model.deg_cost1_BESS, model.DOD1_BESS,  # range and domain variables
                               pw_pts=DOD_index,
                               pw_constr_type='EQ',
                               f_rule=deg_cost_per_cycle,
                               pw_repn='INC')

    # System constraints
    def WTG_Powers_rule(model,t):             # Balance of powers of generation
       return P_gen[t] == model.P_WTG_BESS[t] + model.P_WTG_Grid[t] + model.P_WTG_AEL[t] + model.P_WTG_Curt[t]
    model.WTG_Powers_rule = pyo.Constraint(model.time, rule=WTG_Powers_rule)

    def BESS_charge_Powers_rule(model,t):     # Balance of powers of charging BESS
        return model.P_C_BESS[t] == model.P_WTG_BESS[t] + model.P_Grid_BESS[t]
    model.BESS_charge_Powers_rule = pyo.Constraint(model.time, rule=BESS_charge_Powers_rule)

    def BESS_discharge_Powers_rule(model,t):  # Balance of powers of discharging BESS
        return model.P_D_BESS[t] == model.P_BESS_Grid[t] + model.P_BESS_AEL[t]
    model.BESS_discharge_Powers_rule = pyo.Constraint(model.time, rule=BESS_discharge_Powers_rule)

    def AEL_Powers_rule(model,t):             # Balance of powers of AEL
        return model.P_AEL[t] == model.P_BESS_AEL[t] + model.P_Grid_AEL[t] + model.P_WTG_AEL[t]
    model.AEL_Powers_rule = pyo.Constraint(model.time, rule=AEL_Powers_rule)

    if not RHU_Parameters['Arbitrage']:
        def arbitrage_disabled_rule(model, t):  # Arbitrage is disabled
            return model.P_BESS_Grid[t] ==  0
        model.arbitrage_disabled_rule = pyo.Constraint(model.time, rule=arbitrage_disabled_rule)

    # Objective function
    if RHU_Parameters['Degradation']:
        model.obj = pyo.Objective(
            expr=sum((Price_El[t] * (model.P_WTG_Grid[t] + model.P_BESS_Grid[t]
                                     - (model.P_Grid_BESS[t] + model.P_Grid_AEL[t]))
                      + Price_H2 * model.Q_AEL[t]
                      - model.on_AEL[t] * AEL_cost / AEL_hours
                      -(model.P_start_AEL[t] * AEL_cost) / (AEL_Pmax * AEL_t_start * AEL_cycles))
                     for t in model.time)
                    -  model.deg_cost_BESS - model.deg_cost1_BESS,
                    sense=pyo.maximize)
    if not RHU_Parameters['Degradation']:
        model.obj = pyo.Objective(
                expr=sum((Price_El[t] * (model.P_WTG_Grid[t] + model.P_BESS_Grid[t]
                                         - (model.P_Grid_BESS[t] + model.P_Grid_AEL[t]))
                          + Price_H2 * model.Q_AEL[t]) for t in model.time), sense=pyo.maximize)
    # Apply the solver
    opt = SolverFactory('cbc')
    opt.solve(model)

    # Extract variables
    Results = {}
    Results['H_AEL'] = [model.H_AEL[t]() for t in model.time]
    Results['P_AEL'] = [model.P_AEL[t]() for t in model.time]
    Results['P_on_AEL'] = [model.P_on_AEL[t]() for t in model.time]
    Results['P_idle_AEL'] = [model.P_idle_AEL[t]() for t in model.time]
    Results['P_off_AEL'] = [model.P_off_AEL[t]() for t in model.time]
    Results['P_start_AEL'] = [model.P_start_AEL[t]() for t in model.time]
    Results['Q_AEL'] = [model.Q_AEL[t]() for t in model.time]
    Results['off_AEL'] = [model.off_AEL[t]() for t in model.time]
    Results['on_AEL'] = [model.on_AEL[t]() for t in model.time]
    Results['idle_AEL'] = [model.idle_AEL[t]() for t in model.time]
    Results['cool_AEL'] = [model.cool_AEL[t]() for t in model.time]
    Results['state_AEL'] = [model.state_AEL[t]()*100/AEL_t_off for t in model.time]
    Results['on_frac_AEL'] = [model.on_frac_AEL[t]() for t in model.time]
    Results['P_C_BESS'] = [model.P_C_BESS[t]() for t in model.time]
    Results['P_D_BESS'] = [model.P_D_BESS[t]() for t in model.time]
    Results['SOC'] = [model.E_BESS[t]()*100/BESS_capacity for t in model.time]
    Results['P_WTG_BESS'] = [model.P_WTG_BESS[t]() for t in model.time]
    Results['P_WTG_AEL'] = [model.P_WTG_AEL[t]() for t in model.time]
    Results['P_WTG_Grid'] = [model.P_WTG_Grid[t]() for t in model.time]
    Results['P_WTG_Curt'] = [model.P_WTG_Curt[t]() for t in model.time]
    Results['P_BESS_Grid'] = [model.P_BESS_Grid[t]() for t in model.time]
    Results['P_BESS_AEL'] = [model.P_BESS_AEL[t]() for t in model.time]
    Results['P_Grid_AEL'] = [model.P_Grid_AEL[t]() for t in model.time]
    Results['P_Grid_BESS'] = [model.P_Grid_BESS[t]() for t in model.time]
    Results['imp_AEL'] = [model.imp_AEL[t]() for t in model.time]
    Results['cont_AEL'] = [model.cont_AEL[t]() for t in model.time]
    Results['purg_AEL'] = [model.purg_AEL[t]() for t in model.time]

    # if sum(Results['purg_AEL']) > 1:
    #     print('PURGE DETECTED')
    # Calculate exchanged energy with the system
    P_ex = []
    E_acc = []
    E_acc_h = 0
    for t in range(len(Results['P_AEL'])):
        # Calculated hourly exchanged power
        P_ex_h = Results['P_WTG_Grid'][t] + Results['P_BESS_Grid'][t] - \
                  Results['P_Grid_BESS'][t] - Results['P_Grid_AEL'][t]
        # Calculate accumulated h
        E_acc_h = E_acc_h + P_ex_h
        # Store results
        P_ex.append(P_ex_h)
        E_acc.append(E_acc_h)
    Results['P_ex'] = P_ex
    Results['E_acc'] = E_acc
    return Results

#%%  RHU model for real-time operation
def RHU_RT(RHU_Parameters, P_Gen_real, Price_El_real, DM_commitments, Grid_AEL_DM, Grid_BESS_DM, Price_H2):
    Price_El_real = [abs(a) for a in Price_El_real]
    # Unpack parameters
    AEL_Pmax = RHU_Parameters['AEL Maximum power']
    AEL_Eff = RHU_Parameters['AEL efficiency']
    AEL_Cap = RHU_Parameters['Tank capacity']
    AEL_initial_H = RHU_Parameters['Initial hydrogen level']
    AEL_cost = RHU_Parameters['AEL investment cost'] * 1000 * RHU_Parameters['K_AEL']
    AEL_cycles = RHU_Parameters['Lifetime cycles']
    AEL_hours = RHU_Parameters['Lifetime hours']
    AEL_state_i = RHU_Parameters['Initial state']
    AEL_t_start = RHU_Parameters['Cold start time']
    AEL_t_off = RHU_Parameters['Off time']
    AEL_P_on_min = RHU_Parameters['AEL Minimum power']
    AEL_HHV = RHU_Parameters['HHV']
    AEL_P_idle = RHU_Parameters['Idle state power']
    AEL_P_off = RHU_Parameters['Off state power']
    AEL_initial_imp = RHU_Parameters['AEL initial H2 in O2']
    deg_model = RHU_Parameters['Degradation']
    BESS_SOCi = RHU_Parameters['Batt_SOCi']
    BESS_capacity = RHU_Parameters['Batt_E']
    BESS_maxpower = RHU_Parameters['Batt_P']
    BESS_efficiency = RHU_Parameters['Batt_Eff']
    BESS_cost = RHU_Parameters['Batt_Cost'] * 1000 * RHU_Parameters['K_AEL']
    Dev_coef = RHU_Parameters['Dev_coef']

    # Initialise model
    model = pyo.ConcreteModel()
    model.time = pyo.RangeSet(0, min(len(Price_El_real), len(P_Gen_real)) - 1)

    # AEL variables
    model.H_AEL = pyo.Var(model.time, bounds=(0, AEL_Cap), initialize=0)            # Stored hydrogen
    model.P_AEL = pyo.Var(model.time, bounds=(0, AEL_Pmax), initialize=0)           # AEL power
    model.P_on_AEL = pyo.Var(model.time, bounds=(0, AEL_Pmax), initialize=0)        # AEL 'On' mode power
    model.P_idle_AEL = pyo.Var(model.time, bounds=(0, AEL_P_idle), initialize=0)    # AEL 'Idle' mode power
    model.P_off_AEL = pyo.Var(model.time, bounds=(0, AEL_P_off), initialize=0)      # AEL 'Off' mode power
    model.P_start_AEL = pyo.Var(model.time, bounds=(0, AEL_Pmax), initialize=0)     # AEL cold start power
    model.Q_AEL = pyo.Var(model.time, initialize=0)                                 # AEL hydrogen production
    model.off_AEL = pyo.Var(model.time, domain=pyo.Binary)                          # AEL 'Off' mode verifier
    model.on_AEL = pyo.Var(model.time, domain=pyo.Binary)                           # AEL 'On' mode verifier
    model.idle_AEL = pyo.Var(model.time, domain=pyo.Binary)                         # AEL 'Idle' mode verifier
    model.cool_AEL = pyo.Var(model.time, domain=pyo.Binary)                         # AEL 'Cool' mode verifier
    model.state_AEL = pyo.Var(model.time, bounds=(0, AEL_t_off))                    # AEL state at the end of period
    model.on_frac_AEL = pyo.Var(model.time, bounds=(0, 1))                          # AEL start power fraction
    model.imp_AEL = pyo.Var(model.time, bounds=(0, 2), initialize=0)                # AEL O2 impurity
    model.cont_AEL = pyo.Var(model.time, bounds=(0, 2), initialize=0)               # AEL O2 contamination process
    model.purg_AEL = pyo.Var(model.time, domain=pyo.Binary)                         # AEL purging mode verifier
    # BESS variables
    model.E_BESS = pyo.Var(model.time, bounds=(0, BESS_capacity),
                        initialize=0)                                               # BESS stored energy at the end of t
    model.charging_BESS = pyo.Var(model.time, domain=pyo.Binary)                    # BESS charge verifier
    model.discharging_BESS = pyo.Var(model.time, domain=pyo.Binary)                 # BESS discharge verifier
    model.P_C_BESS = pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # BESS charging power during period
    model.P_D_BESS = pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # BESS discharging power during period
    model.DOD_BESS = pyo.Var(bounds=(0, 100))
    model.deg_cost_BESS = pyo.Var()                                                 # BESS daily degradation cost
    model.max_SOC_BESS = pyo.Var(bounds=(BESS_SOCi, 100))                           # BESS maximum SOC
    model.min_SOC_BESS = pyo.Var(bounds=(0, BESS_SOCi))                             # BESS minimum SOC
    # System variables
    model.P_WTG_BESS= pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # Power from WTG to BESS
    model.P_WTG_AEL= pyo.Var(model.time, bounds=(0, AEL_Pmax),
                             initialize=0)                                          # Power from WTG to AEL
    model.P_WTG_Grid= pyo.Var(model.time, bounds=(0, max(P_Gen_real)),
                             initialize=0)                                          # Power from WTG to Grid
    model.P_WTG_Curt= pyo.Var(model.time, bounds=(0, max(P_Gen_real)),
                             initialize=0)                                           # Power from WTG to curtailment
    model.P_BESS_Grid= pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # Power from BESS to Grid
    model.P_BESS_AEL= pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # Power from BESS to AEL
    model.P_Grid_AEL= pyo.Var(model.time, bounds=(0, AEL_Pmax),
                             initialize=0)                                          # Power from Grid to AEL
    model.P_Grid_BESS= pyo.Var(model.time, bounds=(0, BESS_maxpower),
                             initialize=0)                                          # Power from Grid to BESS
    model.Dev_costs= pyo.Var(model.time,initialize=0)                               # Hourly deviation costs

    # AEL constraints
    def P_rule_AEL(model, t):  # AEL power at each time step is the sum of the different states
        return model.P_on_AEL[t] + model.P_off_AEL[t] + model.P_idle_AEL[t] + model.P_start_AEL[t] == model.P_AEL[t]
    model.P_rule_AEL = pyo.Constraint(model.time, rule=P_rule_AEL)

    def state_rule_AEL(model, t):  # Only one state in the AEL is allowed at the same time
        return model.on_AEL[t] + model.off_AEL[t] + model.idle_AEL[t] == 1
    model.state_rule_AEL = pyo.Constraint(model.time, rule=state_rule_AEL)

    def cool_state_rule_AEL(model, t):  # Cooling state defined when off
        return model.cool_AEL[t] <= model.off_AEL[t]
    model.cool_state_rule_AEL = pyo.Constraint(model.time, rule=cool_state_rule_AEL)

    def cool_rule1_AEL(model, t):  # AEL cooling stops when fully cold
        if t == 0:
            return model.cool_AEL[t] <= AEL_state_i / AEL_t_off + 0.9
        else:
            return model.cool_AEL[t] <= model.state_AEL[t - 1] / AEL_t_off + 0.9
    model.cool_rule1_AEL = pyo.Constraint(model.time, rule=cool_rule1_AEL)

    def cool_rule2_AEL(model, t):  # AEL cooling stops when fully cold
        if t == 0:
            return model.cool_AEL[t] >= AEL_state_i / AEL_t_off - 1 + model.off_AEL[t]
        else:
            return model.cool_AEL[t] >= model.state_AEL[t - 1] / AEL_t_off - 1 + model.off_AEL[t]
    model.cool_rule2_AEL = pyo.Constraint(model.time, rule=cool_rule2_AEL)

    def state_updater_rule_AEL(model, t):  # AEL state updater
        if t == 0:
            return model.state_AEL[t] == AEL_state_i + \
                   model.on_frac_AEL[t] * AEL_t_off - model.cool_AEL[t]
        else:
            return model.state_AEL[t] == model.state_AEL[t - 1] + \
                   model.on_frac_AEL[t] * AEL_t_off - model.cool_AEL[t]
    model.state_updater_rule_AEL = pyo.Constraint(model.time, rule=state_updater_rule_AEL)

    def off_rule_AEL(model, t):  # AEL 'Off' mode power enabler
        return model.P_off_AEL[t] == AEL_P_off * model.off_AEL[t]
    model.off_rule_AEL = pyo.Constraint(model.time, rule=off_rule_AEL)

    def idle_rule_AEL(model, t):  # AEL 'Idle' power enabler
        return model.P_idle_AEL[t] == AEL_P_idle * model.idle_AEL[t]
    model.idle_rule_AEL = pyo.Constraint(model.time, rule=idle_rule_AEL)

    def on_max_rule_AEL(model, t):  # AEL maximum power at 'On' mode
        return model.P_on_AEL[t] <= AEL_Pmax * model.on_AEL[t]
    model.on_max_rule_AEL = pyo.Constraint(model.time, rule=on_max_rule_AEL)

    def on_min_rule_AEL(model, t):  # AEL minimum power for H2 production
        return model.P_on_AEL[t] >= AEL_P_on_min * model.on_AEL[t]
    model.on_min_rule_AEL = pyo.Constraint(model.time, rule=on_min_rule_AEL)

    def on_conditions_rule_AEL(model, t):  # Only can be idle or on in nominal conditions
        return model.on_AEL[t] + model.idle_AEL[t] <= (model.state_AEL[t]) / AEL_t_off
    model.on_conditions_rule_AEL = pyo.Constraint(model.time, rule=on_conditions_rule_AEL)

    def Pstart_rule_AEL(model, t):  # AEL cold start power calculation
        return model.P_start_AEL[t] == model.on_frac_AEL[t] * AEL_t_start * AEL_Pmax
    model.Pstart_rule_AEL = pyo.Constraint(model.time, rule=Pstart_rule_AEL)

    def H_max_rule_AEL(model, t):  # AEL tank capacity
        return model.H_AEL[t] <= AEL_Cap
    model.Hmax_rule_AEL = pyo.Constraint(model.time, rule=H_max_rule_AEL)

    def Q_rule_AEL(model, t):  # AEL hydrogen generation
        return model.Q_AEL[t] == model.P_on_AEL[t] * AEL_Eff / AEL_HHV
    model.Q_rule_AEL = pyo.Constraint(model.time, rule=Q_rule_AEL)

    def state_factor_rule_AEL(model, t):  # AEL starting power fraction constraint
        if t == 0:
            state_prev = AEL_state_i
        else:
            state_prev = model.state_AEL[t - 1]
        return model.on_frac_AEL[t] <= (AEL_t_off - state_prev) / AEL_t_off
    model.state_factor_rule_AEL = pyo.Constraint(model.time, rule=state_factor_rule_AEL)

    def off_mode_rule_AEL(model, t):  # Starting power factor can't be over 1
        return model.on_frac_AEL[t] <= model.off_AEL[t] + model.idle_AEL[t]
    model.off_mode_rule_AEL = pyo.Constraint(model.time, rule=off_mode_rule_AEL)

    def H_rule_AEL(model, t):  # AEL stored hydrogen rule
        if t == 0:
            H_prev = AEL_initial_H
        else:
            H_prev = model.H_AEL[t - 1]
        return model.H_AEL[t] == H_prev + model.Q_AEL[t]
    model.H_rule_AEL = pyo.Constraint(model.time, rule=H_rule_AEL)

    if RHU_Parameters['Purging']:
        def imp_state_rule(model, t):  # Impurity state update
            if t == 0:
                imp_prev = AEL_initial_imp
            else:
                imp_prev = model.imp_AEL[t - 1]
            return model.imp_AEL[t] == imp_prev + model.cont_AEL[t] - 2 * model.purg_AEL[t]
        model.imp_state_rule = pyo.Constraint(model.time, rule=imp_state_rule)

        def enable_purg_rule(model, t): # Purging when off/idle
            if t == 0:
                return model.purg_AEL[t] == 0
            else:
                return model.purg_AEL[t] <= model.off_AEL[t] + model.idle_AEL[t]
        model.enable_purg_rule = pyo.Constraint(model.time, rule=enable_purg_rule)

        def disable_purg_rule(model, t): # Stopping the purging when fully clean
            if t == 0:
                imp_prev = AEL_initial_imp
            else:
                imp_prev = model.imp_AEL[t - 1]
            return model.purg_AEL[t] <= imp_prev
        model.disable_purg_rule = pyo.Constraint(model.time, rule=disable_purg_rule)

        def shutdown_rule(model, t):  # Shutdown when impurity is dangerous
            if t == 0:
                imp_prev = AEL_initial_imp
            else:
                imp_prev = model.imp_AEL[t - 1]
            return imp_prev - model.off_AEL[t]  <= 1.9
        model.shutdown_rule = pyo.Constraint(model.time, rule=shutdown_rule)

        # O2 contamination per P_on piecewise constraint
        contamination_per_P_on = [0., 2., 1., 0.]
        P_on_index = [0., 0.3/AEL_Pmax, 0.5/ AEL_Pmax, 1]
        model.cont_rule = pyo.Piecewise(model.time, model.cont_AEL, model.P_on_AEL,  # range and domain variables
                              pw_pts=P_on_index,
                              pw_constr_type='EQ',
                              f_rule=contamination_per_P_on,
                              pw_repn='INC')

    # BESS constraints
    def P_ch_rule_BESS(model, t):  # Forces limit on charging power
        return (BESS_maxpower * model.charging_BESS[t]) >= model.P_C_BESS[t]
    model.P_ch_rule_BESS = pyo.Constraint(model.time, rule=P_ch_rule_BESS)

    def P_dis_rule_BESS(model, t):  # Forces limit on discharging power
        return (BESS_maxpower * model.discharging_BESS[t]) >= model.P_D_BESS[t]
    model.P_dis_rule_BESS = pyo.Constraint(model.time, rule=P_dis_rule_BESS)

    def states_rule_BESS(model, t):  # Prevents orders of charge and discharge simultaneously
        return (model.charging_BESS[t] + model.discharging_BESS[t]) <= 1
    model.states_rule_BESS = pyo.Constraint(model.time, rule=states_rule_BESS)

    def E_rule_BESS(model, t):  # The stored energy must be the result of (E + charge*eff - discharge/eff)
        if t == 0:
            E_prev = BESS_SOCi * BESS_capacity / 100
        else:
            E_prev = model.E_BESS[t - 1]
        return model.E_BESS[t] == E_prev + model.P_C_BESS[t] * BESS_efficiency - model.P_D_BESS[t] / BESS_efficiency
    model.E_rule_BESS = pyo.Constraint(model.time, rule=E_rule_BESS)

    def SOC_max_rule_BESS(model, t):
        return model.max_SOC_BESS >= model.E_BESS[t] * (100 // BESS_capacity)
    model.SOC_max_rule_BESS = pyo.Constraint(model.time, rule=SOC_max_rule_BESS)

    def SOC_min_rule_BESS(model, t):
        return model.min_SOC_BESS <= model.E_BESS[t] * (100 // BESS_capacity)
    model.SOC_min_rule_BESS = pyo.Constraint(model.time, rule=SOC_min_rule_BESS)

    def DOD_rule_BESS(model):
        return model.DOD_BESS == model.max_SOC_BESS - model.min_SOC_BESS
    model.DOD_rule_BESS = pyo.Constraint(rule=DOD_rule_BESS)

    # BESS degradation model
    deg_cost_per_cycle = [0., BESS_cost / 1000000., BESS_cost / 200000., BESS_cost / 60000., BESS_cost / 40000.,
                          BESS_cost / 20000., BESS_cost / 15000., BESS_cost / 11000., BESS_cost / 10000.,
                          BESS_cost / 8000., BESS_cost / 7000., BESS_cost / 6000.]

    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
    model.deg = pyo.Piecewise(model.deg_cost_BESS, model.DOD_BESS,  # range and domain variables
                              pw_pts=DOD_index,
                              pw_constr_type='EQ',
                              f_rule=deg_cost_per_cycle,
                              pw_repn='INC')

    def EN_rule_BESS(model):
        return sum((model.P_D_BESS[t1] + model.P_C_BESS[t1]) / 2. * (100 // BESS_capacity)
                   for t1 in model.time)
    model.EN_BESS = pyo.Expression(rule=EN_rule_BESS)  # Half of the total energy throughput in %
    model.DOD1_BESS = pyo.Var(bounds=(0, 100))

    def DOD1_rule_BESS(model):
        return model.DOD1_BESS >= model.EN_BESS - model.DOD_BESS
    model.DOD1_rule_BESS = pyo.Constraint(rule=DOD1_rule_BESS)
    model.deg_cost1_BESS = pyo.Var(domain=pyo.NonNegativeReals)
    model.deg1 = pyo.Piecewise(model.deg_cost1_BESS, model.DOD1_BESS,  # range and domain variables
                               pw_pts=DOD_index,
                               pw_constr_type='EQ',
                               f_rule=deg_cost_per_cycle,
                               pw_repn='INC')

    # System constraints
    def WTG_Powers_rule(model,t):             # Balance of powers of generation
       return P_Gen_real[t] == model.P_WTG_BESS[t] + model.P_WTG_Grid[t] + model.P_WTG_AEL[t] + model.P_WTG_Curt[t]
    model.WTG_Powers_rule = pyo.Constraint(model.time, rule=WTG_Powers_rule)

    def BESS_charge_Powers_rule(model,t):     # Balance of powers of charging BESS
        return model.P_C_BESS[t] == model.P_WTG_BESS[t] + Grid_BESS_DM[t]
    model.BESS_charge_Powers_rule = pyo.Constraint(model.time, rule=BESS_charge_Powers_rule)

    def BESS_discharge_Powers_rule(model,t):  # Balance of powers of discharging BESS
        return model.P_D_BESS[t] == model.P_BESS_Grid[t] + model.P_BESS_AEL[t]
    model.BESS_discharge_Powers_rule = pyo.Constraint(model.time, rule=BESS_discharge_Powers_rule)

    def AEL_Powers_rule(model,t):             # Balance of powers of AEL
        return model.P_AEL[t] == model.P_BESS_AEL[t] + model.P_WTG_AEL[t] + Grid_AEL_DM[t]
    model.AEL_Powers_rule = pyo.Constraint(model.time, rule=AEL_Powers_rule)

    if not RHU_Parameters['Arbitrage']:
        def arbitrage_disabled_rule(model, t):  # Arbitrage is disabled
            return model.P_BESS_Grid[t] ==  0
        model.arbitrage_disabled_rule = pyo.Constraint(model.time, rule=arbitrage_disabled_rule)

    def Dev_costs_rule(model,t):                # Deviations cost calculation
        return model.Dev_costs[t] == \
               (DM_commitments[t] - (model.P_WTG_Grid[t] + model.P_BESS_Grid[t]))*Dev_coef*Price_El_real[t]
    model.Dev_costs_rule = pyo.Constraint(model.time, rule=Dev_costs_rule)

    def Dev_costs_sign_rule(model,t):                # Deviations costs can't be negative
        return model.Dev_costs[t] >= 0
    model.Dev_costs_sign_rule = pyo.Constraint(model.time, rule=Dev_costs_sign_rule)
    # Objective function
    if RHU_Parameters['Degradation']:
        model.obj = pyo.Objective(
            expr=sum((Price_H2 * model.Q_AEL[t]
                    - model.Dev_costs[t]
                    - model.on_AEL[t] * AEL_cost / AEL_hours
                    -(model.P_start_AEL[t] * AEL_cost) / (AEL_Pmax * AEL_t_start * AEL_cycles))
                     for t in model.time)
                    -  model.deg_cost_BESS - model.deg_cost1_BESS,
                    sense=pyo.maximize)
    if not RHU_Parameters['Degradation']:
        model.obj = pyo.Objective(
                expr=sum((Price_H2 * model.Q_AEL[t] - model.Dev_costs[t]) for t in model.time), sense=pyo.maximize)
    # Apply the solver
    opt = SolverFactory('cbc')
    opt.solve(model)

    # Extract variables
    Results = {}
    Results['H_AEL'] = [model.H_AEL[t]() for t in model.time]
    Results['P_AEL'] = [model.P_AEL[t]() for t in model.time]
    Results['P_on_AEL'] = [model.P_on_AEL[t]() for t in model.time]
    Results['P_idle_AEL'] = [model.P_idle_AEL[t]() for t in model.time]
    Results['P_off_AEL'] = [model.P_off_AEL[t]() for t in model.time]
    Results['P_start_AEL'] = [model.P_start_AEL[t]() for t in model.time]
    Results['Q_AEL'] = [model.Q_AEL[t]() for t in model.time]
    Results['off_AEL'] = [model.off_AEL[t]() for t in model.time]
    Results['on_AEL'] = [model.on_AEL[t]() for t in model.time]
    Results['idle_AEL'] = [model.idle_AEL[t]() for t in model.time]
    Results['cool_AEL'] = [model.cool_AEL[t]() for t in model.time]
    Results['state_AEL'] = [model.state_AEL[t]()*100/AEL_t_off for t in model.time]
    Results['on_frac_AEL'] = [model.on_frac_AEL[t]() for t in model.time]
    Results['P_C_BESS'] = [model.P_C_BESS[t]() for t in model.time]
    Results['P_D_BESS'] = [model.P_D_BESS[t]() for t in model.time]
    Results['SOC'] = [model.E_BESS[t]()*100/BESS_capacity for t in model.time]
    Results['P_WTG_BESS'] = [model.P_WTG_BESS[t]() for t in model.time]
    Results['P_WTG_AEL'] = [model.P_WTG_AEL[t]() for t in model.time]
    Results['P_WTG_Grid'] = [model.P_WTG_Grid[t]() for t in model.time]
    Results['P_WTG_Curt'] = [model.P_WTG_Curt[t]() for t in model.time]
    Results['P_BESS_Grid'] = [model.P_BESS_Grid[t]() for t in model.time]
    Results['P_BESS_AEL'] = [model.P_BESS_AEL[t]() for t in model.time]
    Results['P_Grid_AEL'] = Grid_AEL_DM
    Results['P_Grid_BESS'] = Grid_BESS_DM
    Results['imp_AEL'] = [model.imp_AEL[t]() for t in model.time]
    Results['cont_AEL'] = [model.cont_AEL[t]() for t in model.time]
    Results['purg_AEL'] = [model.purg_AEL[t]() for t in model.time]
    Results['Dev_costs'] = [model.Dev_costs[t]() for t in model.time]

    # if sum(Results['purg_AEL']) > 1:
    #     print('PURGE DETECTED')
    # Calculate exchanged energy with the system
    P_ex = []
    E_acc = []
    E_acc_h = 0
    for t in range(len(Results['P_AEL'])):
        # Calculated hourly exchanged power
        P_ex_h = Results['P_WTG_Grid'][t] + Results['P_BESS_Grid'][t] - \
                  Results['P_Grid_BESS'][t] - Results['P_Grid_AEL'][t]
        # Calculate accumulated h
        E_acc_h = E_acc_h + P_ex_h
        # Store results
        P_ex.append(P_ex_h)
        E_acc.append(E_acc_h)
    Results['P_ex'] = P_ex
    Results['E_acc'] = E_acc
    return Results