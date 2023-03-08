#%% Circulated energy function
def energy(powers):
    circulated_energy = 0
    for P in powers:
        circulated_energy = circulated_energy + abs(P)
    return circulated_energy


#%% Net benefit function
def scen_eval(powers, prices, SOC, cost, batt_capacity):
    if cost == 0:
        cost = 0.000001
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

    return sum(benefits), sum(benefits) - (deg_cost + deg_cost1) , (deg_cost + deg_cost1)/cost