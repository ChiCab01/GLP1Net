###############################################################################
###############################################################################
########################## LIBRARIES AND TOML #################################
###############################################################################
###############################################################################



###############################################################################
# IMPORTING LIBRARIES

import toml, pandas as pd, numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import ast
import os



###############################################################################
# LOADING OF FILE TOML AND EXTRACTION OF PARAMETERS

# Loading TOML file.
T = toml.load("ODE.toml")

# Extracting elements from TOML file. 
sim = T["simulation"]
plot_cfg = T.get("plot", {})
kin_def = T["kinetics"]["defaults"]
kin_map = T["kinetics"]["rate_law_by_type"]
col = T["column_map"]
init = T.get("initial_conditions", {})
pathway = T.get("pathway", {})
color_map = plot_cfg.get("colors", {})
paths = T.get("paths", {})




###############################################################################
###############################################################################
######################## CREATION OF FUNCTIONS ################################
###############################################################################
###############################################################################



###############################################################################
# 1. FUNCTION TO CREATE LISTS FROM LISTS OR STRINGS.
def as_list(val):

    # 1) If the value is a list, return it whithout modifying it.
    if isinstance(val, list): 
        return val
    
    # 2) If the value is a string, previously it tries to read it as a 
    # python list, otherwise it considers it as a text separed by commas.
    # If any other values is read (no list or string), an empty list is returned.
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []       
        try:
            maybe = ast.literal_eval(s)
            if isinstance(maybe, list):
                return [str(x).strip() for x in maybe]
        except Exception:
            return [x.strip() for x in s.split(",") if x.strip()]
    return []



###############################################################################
# 2. FUNCTION TO SIMULATE CHEMICAL REACTIONS.
def f(t, y):
    
    # 1) Calculate derivative of y with respect to time.
    dydt = np.zeros_like(y) 

    # 2) Iterate on a list called reactions, containing stoichiometry and rate.
    for sto, rate in reactions: 
        v = rate(y) 
        # speed of reaction is calculated based on actual condition.
        for i, coeff in sto.items():
            dydt[i] += coeff * v 
            # for each reaction, speed of variation of each 
            # chemical component is updated combining speed of reaction (v) and 
            # stoichiometric coefficient (coeff).
    return dydt



###############################################################################
# 3. FUNCTION TO CREATE LISTS FROM LISTS OR STRINGS.
def get_float(row, name, default):
    
    # 1) If the name exists, try to convert it into a number.
    # If successful, turn it into a float number.
    #If unsuccessful, keep it as NaN whithout giving errors.
    if name in row and pd.notna(row[name]):
        try:
            return float(row[name])
        except Exception:
            return default
    return default




###############################################################################
###############################################################################
################### ODE (Ordinary Differential Equation) ######################
###############################################################################
###############################################################################


# 0) Paths.
PATH_REACTIONS_XLSX = paths["PATH_REACTIONS_XLSX"]
PATH_REACTIONS_CONTROL_XLSX = paths["PATH_REACTIONS_CONTROL_XLSX"]


# 1) Loading of reactions from an excel file. 
TABLE = PATH_REACTIONS_XLSX if os.path.exists(PATH_REACTIONS_XLSX) else PATH_REACTIONS_CONTROL_XLSX
df = pd.read_excel(TABLE) if TABLE.endswith(".xlsx") else pd.read_csv(TABLE)
df.columns = [str(c).strip() for c in df.columns]


# 2) Definition of useful columns from the dataframe imported.
col_id   = col["reaction_id"]
col_rt   = col["reaction_type"]
col_src  = col["sources"]
col_tgt  = col["targets"]
col_km   = col["km"]
col_kd   = col["kd"]
col_kcat = col["kcat"]


# 3) Choose whether to use all reactions or a part of them.
# If pathway contains a key equal to "mode" set to "list", 
# a part of the reactions is taken, the rows in which ids are elencated in
# pathway["reactions"];
# If pathway contains a key equal to "auto", all reactions are taken.
if pathway.get("mode","auto") == "list":
    wanted = set(pathway.get("reactions", []))
    df = df[df[col_id].isin(wanted)].copy()
else:
    df = df.copy() 


# 4) Rows without sources and target are discarded; a sum of numbers of 
# sources and targets is computed for each row: if equal to 0, no inputs or
# outputs are given for the reaction, so it is discarded from the dataframe.
df[col_src] = df[col_src].apply(as_list)
df[col_tgt] = df[col_tgt].apply(as_list)
df = df[(df[col_src].map(len) + df[col_tgt].map(len)) > 0].reset_index(drop=True)



# 5) List of all metabolites involved is created and at which initial concentration they
# are is set. 
species = sorted(set(sum(df[col_src].tolist(), []) + sum(df[col_tgt].tolist(), [])))
idx = {s:i for i,s in enumerate(species)}
y0 = np.zeros(len(species), float)
for k,v in init.items():
    if k in idx:
        y0[idx[k]] = float(v)


# 6) Kinetic constants are extracted from the TOML, in a float type.
Km_def = float(kin_def.get("Km_uM", 1.0))
Kd_def = float(kin_def.get("Kd_uM", 0.5))
kcat_def = float(kin_def.get("kcat_s", 0.1))
fin = float(kin_def.get("fin_uM_s", 0.05))
kout = float(kin_def.get("kout_s", 0.01))


# 7) Mathematical model for ODE is created.
reactions = []
for _, row in df.iterrows():
    rtype = str(row.get(col_rt, "")).strip().lower()
    srcs  = row[col_src]
    tgts  = row[col_tgt]

    # Stoichiometry is created.
    sto = {}
    for s in srcs:
        if s in idx: sto[idx[s]] = sto.get(idx[s], 0.0) - 1.0
    for p in tgts:
        if p in idx: sto[idx[p]] = sto.get(idx[p], 0.0) + 1.0
    if not sto:
        continue

    # The kinetic law (rate) is defined.
    km   = get_float(row, col_km, Km_def)
    kd   = get_float(row, col_kd, Kd_def)
    kcat = get_float(row, col_kcat, kcat_def)
    law = kin_map.get(rtype, "mm")

    # Si estraggono gli indici di TUTTI i reagenti (non solo il primo)
    src_indices = [idx[s] for s in srcs if s in idx]
    iS = src_indices[0] if src_indices else None

    # Leggi cinetiche con protezione contro valori negativi (max(y[i], 0.0))
    if law == "const_in": 
        rate = lambda y, fin=fin: fin
    elif law == "first_order_out": 
        rate = lambda y, i=iS, k=kout: (k * max(y[i], 0.0)) if i is not None else 0.0
    elif law == "kd_sat": 
        rate = lambda y, i=iS, kd=kd, kcat=kcat: (kcat * max(y[i], 0.0) / (kd + max(y[i], 0.0) + 1e-12)) if i is not None else 0.0
    elif law == "mass_action":
        # Supporta più reagenti (es. A + B -> C)
        rate = lambda y, indices=src_indices, k=kcat: k * np.prod([max(y[j], 0.0) for j in indices])
    else:  
        # Default: Michaelis-Menten
        rate = lambda y, i=iS, km=km, kcat=kcat: (kcat * max(y[i], 0.0) / (km + max(y[i], 0.0) + 1e-12)) if i is not None else 0.0

    reactions.append((sto, rate))


# 8) if no reaction selected, return the sentence below.
if not reactions:
    print("No reactions selected; pathway.mode/column_map and reaction_id should be checked.")
    exit()



# 9) All elements prepared are given to a differential equation solver 
# to calculate the evolution of the system over time.
t_eval = np.linspace(sim["t0"], sim["t_end"], sim["n_eval"])

sol = solve_ivp(f, (sim["t0"], sim["t_end"]), y0, t_eval=t_eval,
                method=sim.get("method", "BDF"),
                rtol=sim.get("rtol", 1e-6), atol=sim.get("atol", 1e-9))

# 10) Plot: on x axis there is the time, on the y axis the species concentration.
to_plot = plot_cfg.get("species_to_plot", [])
if not to_plot:  
    to_plot = species[:5]

plt.figure(figsize=(10,5))
for s in to_plot:
    if s in idx:
        c = color_map.get(s)  
        if c:
            plt.plot(sol.t, sol.y[idx[s]], label=s, color=c)
        else:
            plt.plot(sol.t, sol.y[idx[s]], label=s)
    else:
        print(f"[warning] specie '{s}' non trovata tra: {species}")
plt.xlabel(f"Time ({T['units']['time']})")
plt.ylabel(f"Concentration ({T['units']['concentration']})")
plt.yscale(plot_cfg.get("yscale","linear"))
plt.title("ODE simulation")
plt.legend()
plt.grid(True)
plt.show()