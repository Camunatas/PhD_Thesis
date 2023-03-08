import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# %% Arbitrage function (without degradation)
# def arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower,
#               batt_efficiency, cost):
#     # Model initialization
#     model = pyo.ConcreteModel()
#     model.time = pyo.RangeSet(0, len(energy_price)-1)
#     model.SOC = pyo.Var(model.time, bounds=(0, batt_capacity), initialize=0)  # Battery SOC at the end of period. in energy units
#     model.charging = pyo.Var(model.time, domain=pyo.Binary)                # Charge verifier
#     model.discharging = pyo.Var(model.time, domain=pyo.Binary)             # Discharge verifier
#     model.ESS_C = pyo.Var(model.time, bounds=(0, batt_maxpower))           # Energy being charged during period
#     model.ESS_D = pyo.Var(model.time, bounds=(0, batt_maxpower))           # Energy being discharged during period
#
#
#     # Defining the optimization constraints
#     def c1_rule(model, t):  # Forces limit on charging power
#         return (batt_maxpower * model.charging[t]) >= model.ESS_C[t]
#     model.c1 = pyo.Constraint(model.time, rule=c1_rule)
#
#     def c2_rule(model, t):  # Forces limit on discharging power
#         return (batt_maxpower * model.discharging[t]) >= model.ESS_D[t]
#     model.c2 = pyo.Constraint(model.time, rule=c2_rule)
#
#     def c3_rule(model, t):  # Prevents orders of charge and discharge simultaneously
#         return (model.charging[t] + model.discharging[t]) <= 1
#     model.c3 = pyo.Constraint(model.time, rule=c3_rule)
#
#     def c4_rule(model, t):  # The SOC must be the result of (SOC + charge*eff - discharge/eff)
#         if t == 0:
#             soc_prev = initial_SOC
#         else:
#             soc_prev = model.SOC[t-1]
#         return model.SOC[t] == soc_prev + model.ESS_C[t] * batt_efficiency - model.ESS_D[t] / batt_efficiency
#     model.c4 = pyo.Constraint(model.time, rule=c4_rule)
#
#     def c5_rule(model):
#            return model.SOC[len(energy_price)-1] == 0.0
#     model.c5 = pyo.Constraint(rule=c5_rule)
#
#
#     # Objective Function: Maximize profitability
#     model.obj = pyo.Objective(
#         expr=sum((energy_price[t] * (model.ESS_D[t] - model.ESS_C[t]))
#               for t in model.time), sense=pyo.maximize)
#
#     # Applying the solver
#     opt = SolverFactory('cbc')
#     opt.solve(model)
#     # model.pprint()
#     ESS_D = [model.ESS_D[t1]() for t1 in model.time]
#     ESS_C = [model.ESS_C[t1]() for t1 in model.time]
#     for i in range(len(energy_price)):
#         if ESS_D[i] is None:
#             ESS_D[i] = 0
#         if ESS_C[i] is None:
#             ESS_C[i] = 0
#
#     # Extracting data from model
#     _SOC_E = [model.SOC[t1]() for t1 in model.time]
#     _SOC_E.insert(0,initial_SOC)
#     _SOC = [i * (100 // batt_capacity) for i in _SOC_E]
#     _P_output = [-ESS_D[t1] + ESS_C[t1] for t1 in model.time]
#
#
#     return _P_output, _SOC

# %% Arbitrage function (with degradation)
def arbitrage(initial_SOC, energy_price, batt_capacity, batt_maxpower,
              batt_efficiency, cost):
    # Model initialization
    model = pyo.ConcreteModel()
    model.time = pyo.RangeSet(0, len(energy_price) - 1)
    model.SOC = pyo.Var(model.time, bounds=(0, batt_capacity),
                        initialize=0)  # Battery SOC at the end of period. in energy units
    model.charging = pyo.Var(model.time, domain=pyo.Binary)  # Charge verifier
    model.discharging = pyo.Var(model.time, domain=pyo.Binary)  # Discharge verifier
    model.ESS_C = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being charged during period
    model.ESS_D = pyo.Var(model.time, bounds=(0, batt_maxpower))  # Energy being discharged during period
    model.DOD = pyo.Var(bounds=(0, 100))
    model.deg_cost = pyo.Var()
    model.max_SOC = pyo.Var(bounds=(initial_SOC, 100))
    model.min_SOC = pyo.Var(bounds=(0, initial_SOC))

    # Defining the optimization constraints
    def c1_rule(model, t):  # Forces limit on charging power
        return (batt_maxpower * model.charging[t]) >= model.ESS_C[t]

    model.c1 = pyo.Constraint(model.time, rule=c1_rule)

    def c2_rule(model, t):  # Forces limit on discharging power
        return (batt_maxpower * model.discharging[t]) >= model.ESS_D[t]

    model.c2 = pyo.Constraint(model.time, rule=c2_rule)

    def c3_rule(model, t):  # Prevents orders of charge and discharge simultaneously
        return (model.charging[t] + model.discharging[t]) <= 1

    model.c3 = pyo.Constraint(model.time, rule=c3_rule)

    def c4_rule(model, t):  # The SOC must be the result of (SOC + charge*eff - discharge/eff)
        if t == 0:
            soc_prev = initial_SOC
        else:
            soc_prev = model.SOC[t - 1]
        return model.SOC[t] == soc_prev + model.ESS_C[t] * batt_efficiency - model.ESS_D[t] / batt_efficiency

    model.c4 = pyo.Constraint(model.time, rule=c4_rule)

    def c5_rule(model):
        return model.SOC[len(energy_price) - 1] == 0.0

    model.c5 = pyo.Constraint(rule=c5_rule)

    def c6_rule(model, t):
        return model.max_SOC >= model.SOC[t] * (100 // batt_capacity)

    model.c6 = pyo.Constraint(model.time, rule=c6_rule)

    def c7_rule(model, t):
        return model.min_SOC <= model.SOC[t] * (100 // batt_capacity)

    model.c7 = pyo.Constraint(model.time, rule=c7_rule)

    def c8_rule(model):
        return model.DOD == model.max_SOC - model.min_SOC

    model.c8 = pyo.Constraint(rule=c8_rule)

    # Degradation model
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100]
    # Newest curve
    # deg_cost_per_cycle = [0., cost/300000., cost/100000., cost/20000., cost/10000.,
    #                       cost/6000., cost/5000., cost/3000., cost/2000.,cost/1500., cost/1100., cost/1000.]
    # Second-new curve
    deg_cost_per_cycle = [0., cost / 1000000., cost / 200000., cost / 60000., cost / 40000.,
                          cost / 20000., cost / 15000., cost / 11000., cost / 10000., cost / 8000., cost / 7000.,
                          cost / 6000.]
    # Old degradation curve
    # deg_cost_per_cycle = [0., cost / 15000., cost / 7000., cost / 3300., cost / 2050., cost / 1475., cost / 1150.,
    #                       cost / 950., cost / 760., cost / 675., cost / 580., cost / 500.]
    model.deg = pyo.Piecewise(model.deg_cost, model.DOD,  # range and domain variables
                              pw_pts=DOD_index,
                              pw_constr_type='EQ',
                              f_rule=deg_cost_per_cycle,
                              pw_repn='INC')

    def EN_rule(model):
        return sum((model.ESS_D[t1] + model.ESS_C[t1]) / 2. * (100 // batt_capacity)
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
    # Objective Function: Maximize profitability
    model.obj = pyo.Objective(
        expr=sum((energy_price[t] * (model.ESS_D[t] - model.ESS_C[t]))
                 for t in model.time) - model.deg_cost - model.deg_cost1, sense=pyo.maximize)

    # Applying the solver
    opt = SolverFactory('cbc')
    opt.solve(model)
    # model.pprint()
    ESS_D = [model.ESS_D[t1]() for t1 in model.time]
    ESS_C = [model.ESS_C[t1]() for t1 in model.time]
    DOD1 = model.DOD1()
    DOD = model.DOD()
    for i in range(len(energy_price)):
        if ESS_D[i] is None:
            ESS_D[i] = 0
        if ESS_C[i] is None:
            ESS_C[i] = 0
        if DOD1 is None:
            DOD1 = 0
        if DOD is None:
            DOD = 0

    # Extracting data from model
    _SOC_E = [model.SOC[t1]() for t1 in model.time]
    _SOC_E.insert(0, initial_SOC)
    _SOC = [i * (100 // batt_capacity) for i in _SOC_E]
    _P_output = [-ESS_D[t1] + ESS_C[t1] for t1 in model.time]
    if DOD1 > DOD:
        print()
        print(model.DOD(), model.DOD1())
        print(model.deg_cost(), model.deg_cost1())
        print(_P_output)
        print(_SOC)
        print(energy_price)
    # _P_output = np.zeros(len(energy_price))
    # for i in range(len(energy_price)):
    #     if i == 0:
    #         _P_output[i] = 0
    #     else:
    #         _P_output[i-1] = round(_SOC_E[i] - _SOC_E[i-1],4)

    return _P_output, _SOC
#%% Circulated energy function
def energy(powers):
    circulated_energy = 0
    for P in powers:
        circulated_energy = circulated_energy + abs(P)
    return circulated_energy


#%% Net benefit function
def scen_eval(powers, prices, SOC, cost, batt_capacity):
    # Degradation model
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
    deg_cost_per_cycle = [0., cost / 1000000., cost / 200000., cost / 60000., cost / 40000.,
                          cost / 20000., cost / 15000., cost / 11000., cost / 10000., cost / 8000., cost / 7000.,
                          cost / 6000.]

    benefits = []
    en100 = energy(powers)/2/batt_capacity*100
    DOD = max(SOC) - min(SOC)
    for d in range(len(DOD_index) - 1):
        if DOD >= DOD_index[d] and DOD <= DOD_index[d + 1]:
            deg_cost = deg_cost_per_cycle[d] + (deg_cost_per_cycle[d + 1] - deg_cost_per_cycle[d]) * (
                        DOD - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
            break

    DOD1 = max(en100-DOD,0)
    if DOD1>100:
        deg_cost1 = deg_cost_per_cycle[-1]
    for d in range(len(DOD_index) - 1):
        if DOD1 >= DOD_index[d] and DOD1 <= DOD_index[d + 1]:
            deg_cost1 = deg_cost_per_cycle[d] + (deg_cost_per_cycle[d + 1] - deg_cost_per_cycle[d]) * (
                        DOD1 - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
            break

    # Obtaining benefits, sales and purchases
    for i in range(min(len(powers),len(prices))):
        Benh = -powers[i] * prices[i]
        benefits.append(Benh)

    return sum(benefits), (deg_cost + deg_cost1)/cost
    # return sum(benefits) - (deg_cost + deg_cost1)   , (deg_cost + deg_cost1)/cost


#%% Daily benefits, energy & deg scaler for emulating capacity loss
def Deg_scaler(Powers, Benefits, SOCs, deg_acc, EOL_Capacity, daily_deg, cost, calendar_deg):
    DOD_max = max(SOCs)/100
    # Scaling maximum DOD
    DOD_max_new = 1-deg_acc
    # Degradation model
    DOD_index = [0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
    deg_cost_per_cycle = [0., cost / 1000000., cost / 200000., cost / 60000., cost / 40000.,
                          cost / 20000., cost / 15000., cost / 11000., cost / 10000., cost / 8000., cost / 7000.,
                          cost / 6000.]
    # Energy, degradation and benefits
    daily_energy = energy(Powers)
    if DOD_max > DOD_max_new:
        Benefits = DOD_max_new * Benefits / DOD_max
        daily_energy = daily_energy * DOD_max_new / DOD_max
        en100 = daily_energy
        DOD = DOD_max_new * 100
        for d in range(len(DOD_index) - 1):
            if DOD >= DOD_index[d] and DOD <= DOD_index[d + 1]:
                deg_cost = deg_cost_per_cycle[d] + (deg_cost_per_cycle[d + 1] - deg_cost_per_cycle[d]) * (
                        DOD - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
                break

        DOD1 = max(en100 - DOD, 0)
        if DOD1 > 100:
            deg_cost1 = deg_cost_per_cycle[-1]
        for d in range(len(DOD_index) - 1):
            if DOD1 >= DOD_index[d] and DOD1 <= DOD_index[d + 1]:
                deg_cost1 = deg_cost_per_cycle[d] + (deg_cost_per_cycle[d + 1] - deg_cost_per_cycle[d]) * (
                        DOD1 - DOD_index[d]) / (DOD_index[d + 1] - DOD_index[d])
                break
        daily_deg = ((deg_cost + deg_cost1) / cost) * (1 - EOL_Capacity)
    # Stopping ESS use if EOL is met
    if deg_acc >= EOL_Capacity:
        daily_deg = daily_energy = Benefits = 0

    return Benefits, daily_energy, daily_deg + calendar_deg





