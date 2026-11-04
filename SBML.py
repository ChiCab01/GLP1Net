###############################################################################
###############################################################################
########################## LIBRARIES AND TOML #################################
###############################################################################
###############################################################################



###############################################################################
# IMPORTING LIBRARIES

import toml
import cobra
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import libsbml
import re
from cobra import Model, Reaction, Metabolite


###############################################################################
# LOADING OF FILE TOML AND EXTRACTION OF PARAMETERS

# Loading TOML file.
TOML = toml.load("SBML.toml")

# Extracting elements from TOML file. 
config = {
    "metadata": TOML.get("metadata", {}),
    "paths": TOML.get("paths", {}),
    "GREEK_LETTERS": TOML.get("GREEK_LETTERS", {}),
    "defaults": TOML.get("defaults", {}),
    "DEFAULT_TYPE_BOUNDS": TOML.get("DEFAULT_TYPE_BOUNDS", {}),
    "REACTION_PATTERNS": TOML.get("REACTION_PATTERNS", {})
}




###############################################################################
###############################################################################
######################## CREATION OF FUNCTIONS ################################
###############################################################################
###############################################################################



###############################################################################
# 1.1) FUNCTION TO CLEAN AND NORMALIZE METABOLITES AND REACTIONS NAME.
def clean_metabolite_id(met_string: str, 
                        config: dict) -> str:
    
    # 1) If the "met_string" is empty or not a string, "" is returned;
    if pd.isna(met_string) or not str(met_string).strip():
        return ""
    
    # 2) Otherwise extract the name of the metabolite, turn it into lower case and strip it.
    name = str(met_string).lower().strip()

    # 3) Extract from the TOML file the dictionary to convert greek letters
    # in their spelled form (eg. "α" = "alpha").
    GREEK_LETTERS = config.get("GREEK_LETTERS", {})

    # 4) Convert greek letters in their spelled form.
    for greek, written in GREEK_LETTERS.items():
        name = name.replace(greek, written)

    # 5) Convert +, - and spaces into "".
    name = name.replace("+", "")   
    name = name.replace("-", "")  
    name = name.replace(" ", "")   
    return name




# 1.2) FUNCTION TO CLEAN METABOLITES AND TURN THEM INTO LISTS.
def split_and_clean_metabolites(met_string: str, 
                                config: dict):
    # 1) If the "met_string" is empty or not a string, "" is returned;
    if pd.isna(met_string) or not str(met_string).strip():
        return ""
    
    # 2) The string is splitted where + and , 
    metabolites = str(met_string).split(',')

    # 3) The metabolites are cleaned using clean_metabolite_id function and
    # and are turned into a list.
    cleaned = [clean_metabolite_id(met, config) for met in metabolites if met.strip()]
    
    # 4) Metabolites duplicated are removed.
    seen, unique_cleaned = set(), []
    for m in cleaned:
        if m not in seen:
            seen.add(m)
            unique_cleaned.append(m)

    return unique_cleaned




# 1.2) FUNCTION TO CLEAN CATALYSTS AND TURN THEM INTO LISTS.
def _split_and_modify_catalysts(catalyst_string: str, 
                                config: dict) -> str:
    
    # 1) If "catalyst_string" is empty or not a string, "" is returned.
    if not isinstance(catalyst_string, str) or not catalyst_string.strip():
        return ""
    
    # 2) Split the string of catalysts where , and strip it, then save catalysts into a list.
    parts = [p.strip() for p in catalyst_string.split(',') if p.strip()]
    
    # 3) Clean catalysts using clean_metabolite_id function and 
    # to each of them add "_a" as a suffix, saving them into a list.
    modified_parts = [f"{clean_metabolite_id(p, config)}_a" for p in parts]
    
    # 4) Join the catalysts together again.
    return ",".join(modified_parts)




# 1.4) FUNCTION TO DEFINE LOWER AND UPPER BOUNDS. 
def lu_bounds(
        reaction_type: str, 
        lb_in, ub_in,
        config: dict) -> tuple[float, float]:

    # 1) Extraction of DEFAULT_TYPE_BOUNDS and defaults from TOML file.
    type_bounds_config = config.get("DEFAULT_TYPE_BOUNDS", {})
    defaults_config = config.get("defaults", {})
    default_lower_bound = defaults_config.get("lower_bound", -1000.0)
    default_upper_bound = defaults_config.get("upper_bound", 1000.0)
    bound_override = defaults_config.get("bound_override", "auto")

    # 2) Take from DEFAULT_TYPE_BOUNDS in TOML the lower and upper bounds for FBA
    # and whether the reaction is reversible or not.
    info = type_bounds_config.get(reaction_type, 
                                  {"bounds": (default_lower_bound, default_upper_bound), 
                                   "reversible": False})
    default_lb, default_ub = info["bounds"]
    reversible = info.get("reversible", False) 
    
    lb, ub = None, None

    # 3) Define rules for bound_override:
    if bound_override == "force":
        lb, ub = default_lb, default_ub
        # if the option is "force", use default lower and upper bounds previously defined.
    elif bound_override in ("auto", "preserve"):
        if pd.notna(lb_in) and pd.notna(ub_in):    
            try:
                lb, ub = float(lb_in), float(ub_in)
            except (ValueError, TypeError):
                    pass
        # if the option is "auto"/"preserve", use bounds from the file if available and valid.

    # 4) Check to be sure a value is assigned to lb and ub.
    if lb is None:
        lb = default_lb
    if ub is None:
        ub = default_ub
        # if no valid bounds are present in the file, use the generic defaults.

    # 5) Choose a fixed rule for reactions that are reversible.
    if reversible:
        U = max(abs(lb), abs(ub), default_upper_bound)
        lb, ub = -float(U), float(U)
        # if the reaction is reversible, upper and lower bound are always equals and opposites. 

    # 6) Final check to be sure that lower and upper bound are float and that they are at least equal 
    # to the default values instead of missing values.
    lb = float(lb if lb is not None else default_lower_bound)
    ub = float(ub if ub is not None else default_upper_bound)

    return lb, ub




###############################################################################
# 2.1) FUNCTION TO LOAD METABOLITES AND REACTIONS FROM CSV.
def load_initial_data(metabolites_csv_path: str, 
                      reactions_csv_path: str, 
                      config: dict) -> tuple[dict, pd.DataFrame]:

    # 1) Load raw metabolites from the path specified.
    df_met = pd.read_csv(metabolites_csv_path)
    metabolites = {}

    # 1.1)  For each row extract the "name" and the "compartment" whether present (otherwise fill
    # it with the default "c"); clean the metabolite "name" using clean_metabolite_id function, eliminate
    # empty and duplicated metabolites and save them into "clean_id"; 
    # store into Metabolite of COBRA the "id", the "name" and the "compartment" of the metabolites.
    duplicates_found = []
    empty_after_clean = []

    for _, row in df_met.iterrows():
        raw_name = row['name']
        compartment = row['compartment'] if 'compartment' in df_met.columns else 'c'
        clean_id = clean_metabolite_id(raw_name, config)
        if not clean_id:
            empty_after_clean.append(raw_name)
            continue
        if clean_id in metabolites:
            duplicates_found.append(f"'{raw_name}' (cleaned as '{clean_id}')")
        else:
            metabolites[clean_id] = Metabolite(id=clean_id, 
                                               name=raw_name, 
                                               compartment=compartment)
    
    # 1.2) To check how many metabolites have been eliminated due to empty or repeated "clean_id",
    # a report is printed.
    print(f"\n[Report Metabolites]")
    print(f"Original file contains {len(df_met)} rows")
    if empty_after_clean:
        print(f"{len(empty_after_clean)} discarded metabolites since their cleaned name was empty.")
        print("Discarded metabolites:", empty_after_clean) 
    if duplicates_found:
        print(f"{len(duplicates_found)} discarded metabolites since duplicated after their cleaning.")
        # print("Original metabolites name:", duplicates_found) 
    print(f"Unique metabolites name kept: {len(metabolites)}")    


    # 2) Load raw reactions from the path specified.
    # 2.1) Eliminate rows whose "name" value is missing.
    df_reactions_raw = pd.read_csv(reactions_csv_path)
    dropped_reactions = df_reactions_raw[df_reactions_raw['name'].isna()]
    df_reactions = df_reactions_raw.dropna(subset=["name"])

    # 2.2) To check how many reactions have been eliminated due to empty "name" values,
    # a report is printed.
    print(f"\n[Report Reactions]")
    print(f"Original file contains {len(df_reactions_raw)} rows")
    if not dropped_reactions.empty:
        print(f"{len(dropped_reactions)} discarded  reactions since 'name' column was empty.")
        print(f"Indexes of reactions discarded  {list(dropped_reactions.index)})")
    print(f"Valid reactions loaded: {len(df_reactions)}")

    print(f"Loaded {len(metabolites)} metabolites and {len(df_reactions)} initial reactions.")
    return metabolites, df_reactions




# 2.2) FUNCTION TO PREPARARE REACTION DATAFRAME FOR NEXT STEPS.
def prepare_reaction_dataframe(raw_reactions_df: pd.DataFrame, 
                               valid_ids: set, 
                               config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # 1) Creation of a copy of the df_reactions_raw imported with load_initial_data.
    df = raw_reactions_df.copy()
    
    # 2) Columns of this dataframe are turned into lower case, stripped and cleaned
    # substituting ampty spaces with "_"   
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # 3) Kinetic constants (Km, Kd, kcat) are three columns of the raw data file;
    # with a function, row by row the values are checked to be not null;
    # if not missing, they are turned into numbers;
    # if only some values are missing, they are transformed in "NaN" not to cause error;
    # if all the column is empty, it is filled with "NaN" values.

    # The function is then applied and the new dataframe df is going to be
    # characterized by the three cleaned columns "km_um", "kd_um", "kcat_s".
    def _pick_num(df, colnames):
        for cn in colnames:
            if cn in df.columns:
               return pd.to_numeric(df[cn], errors="coerce")
        return pd.Series(np.nan, index=df.index)
    df["km_um"]  = _pick_num(df, ["km(µm)", "km(um)", "km_um", "km"])
    df["kd_um"]  = _pick_num(df, ["kd(µm)", "kd(um)", "kd_um", "kd"])
    df["kcat_s"] = _pick_num(df, ["kcat(s^-1)", "kcat_s^-1", "kcat_s", "kcat"])

    # 1) Extraction of "source" and "target" name from "shared_name" column.
    # In cytoscape, "source" and target" in "shared_name" are reversed.
    if 'shared_name' in df.columns:
        extracted = df['shared_name'].str.extract(r"(.+?)\s+\(interacts with\)\s+(.+)")
        df['source_from_name'] = extracted[1].fillna("")
        df['target_from_name'] = extracted[0].fillna("")
    else:
        df['source_from_name'] = ""
        df['target_from_name'] = ""

    # 2) Fusion of "source" and "target" with "other_sources" and "other_targets" in presence of these last ones;
    # otherwise, keep "source" and "target" extracted at step 1).
    # In the end, only "source", "target", "catalyst", "shared_name" and "disease" are saved in the dataframe df.
    df['source'] = df['other_sources'].where(df['other_sources'].notna(), df['source_from_name'])
    df['target'] = df['other_targets'].where(df['other_targets'].notna(), df['target_from_name'])
    df['catalyst'] = df['catalyst'].fillna("")
    df['shared_name'] = df['shared_name'] if 'shared_name' in df.columns else ""
    df['disease'] = df['disease'] if 'disease' in df.columns else ""

    # 3) Cleaning and splitting "source" and "target" into lists; simply cleaning of "catalyst".
    df['source_list'] = df['source'].apply(lambda x: split_and_clean_metabolites(x, config))
    df['target_list'] = df['target'].apply(lambda x: split_and_clean_metabolites(x, config))
    df['catalyst_clean_str'] = df['catalyst'].apply(lambda x: _split_and_modify_catalysts(x, config))

    # 4) Identification of non valid metabolites, that are the ones of the reaction file that differ
    # from the list of metabolites imported from the metabolite file.
    invalid_rows = []
    for idx, row in df.iterrows():
        for met_list, role in [(row['source_list'], 'source'), (row['target_list'], 'target')]:
            for met in met_list:
                if met not in valid_ids:
                    invalid_rows.append({
                        "reaction_id": row['shared_name'] or f"row_{idx}",
                        "invalid_metabolite": met,
                        "role": role
                    })
    invalid_df = pd.DataFrame(invalid_rows)
    if not invalid_df.empty:
        invalid_df.to_excel("MISMATCHED_METABOLITES.xlsx", index=False)
        print(f"Found {len(invalid_df)} metabolites not valid. Report saved in MISMATCHED_METABOLITES.xlsx")
    else:
        pd.DataFrame(columns=['reaction_id','invalid_metabolite','role']).to_excel("MISMATCHED_METABOLITES.xlsx", index=False)
        print("No valid metabolites have been found.")

    # 5) Definition of the columns for the future new dataframe: 
    # "reaction_type", "shared_name", "disease", "reaction_string", "reaction_id",
    # "source_clean_str", "target_clean_str", "catalyst_clean_str", "source_modified", "target_modified".
    processed_rows = []
    for idx, row in df.iterrows():
        # 5.1) Reactions presenting "effect" as "target" are associated with the "reaction_type" == sink;
        # so they follow "sink" rules. 
        target_list = [t for t in row['target_list'] if t.lower() != 'effect'] 
        source_list = row['source_list']
        if "effect" in row['target_list']:  
            for met in source_list:  
                processed_rows.append({
                    'reaction_type': 'sink',
                    'source_clean_str': met,
                    'target_clean_str': "",
                    'catalyst_clean_str': "",
                    'shared_name': row['shared_name'],
                    'disease': row['disease'],
                    'reaction_string': f"{met} --> out",
                    'reaction_id': f"output_{met}",
                    'source_modified': [met],  
                    'target_modified': [],
                    'km_um':  row.get('km_um', np.nan),
                    'kd_um':  row.get('kd_um', np.nan),
                    'kcat_s': row.get('kcat_s', np.nan),    
                })
            continue

        # 5.2) Generical reactions rule definition. 
        processed_rows.append({
            'reaction_type': row.get('reaction_type', ''),
            'source_clean_str': ",".join(source_list),
            'target_clean_str': ",".join(target_list),
            'catalyst_clean_str': row['catalyst_clean_str'],
            'shared_name': row['shared_name'],
            'disease': row['disease'],
        })

    # 6) Creation of a dataframe for all reaction types excluded
    # "feed" and "sink" reactions.
    df_main = pd.DataFrame(processed_rows)

    # 7) Identifications of all sources and all targets so that ...
    all_sources = [m for lst in df['source_list'] for m in lst]
    all_targets = [m for lst in df['target_list'] for m in lst]

    # ... all metabolites that are only sources (nothing feed them)
    # are treated as "feed" reactions.
    feed_mets = set(all_sources) - set(all_targets)
    feed_rows = [{
        'reaction_type': 'feed',
        'source_clean_str': "",
        'target_clean_str': met,
        'catalyst_clean_str': "",
        'shared_name': '',
        'disease': '',
        'reaction_string': f"in --> {met}",
        'reaction_id': f"input_{met}",
        'source_modified': [],  # <--
        'target_modified': [met],  # <--
        'km_um':  np.nan,
        'kd_um':  np.nan,
        'kcat_s': np.nan,
    } for met in feed_mets]

    # ... all metabolites that are only targets (nothing interacts 
    # with them anymore) are treated as "sink" reactions.
    sink_mets = set(all_targets) - set(all_sources)
    sink_rows = [{
        'reaction_type': 'sink',
        'source_clean_str': met,
        'target_clean_str': "",
        'catalyst_clean_str': "",
        'shared_name': '',
        'disease': '',
        'reaction_string': f"{met} --> out",
        'reaction_id': f"output_{met}",
        'source_modified': [met],
        'target_modified': [],
        'km_um':  np.nan,
        'kd_um':  np.nan,
        'kcat_s': np.nan,
    } for met in sink_mets]

    # 8) Creation of a dataframe for "feed" and "sink" reaction types.
    df_feed_sink = pd.DataFrame(feed_rows + sink_rows)

    # 9) Application of the reaction patterns in the TOML file to the sources and targets
    # created to create "reaction_string" and "reaction_id" based on "reaction_type".
    for idx, row in df_main.iterrows():
            # 9.1) If "reaction_type" == "sink" or "feed", these rows are skipped.
            if row['reaction_type'] in ('sink', 'feed') or not row['target_clean_str']:
                continue

            # 9.2) REACTION_PATTERNS is extracted from TOML file and used to update sources and
            # targets in order to compile "reaction_string" column.
            r_type = row.get('reaction_type', '')
            pattern = config.get("REACTION_PATTERNS", {}).get(r_type, {})

            sources = row['source_clean_str'].split(',') if row['source_clean_str'] else []
            targets = row['target_clean_str'].split(',') if row['target_clean_str'] else []

            sources_mod, targets_mod = [], []
            seen_sources, seen_targets = set(), set()

            for template in pattern.get("sources", {}).keys():
                for s in sources:
                    for t in targets:
                        # 1) Creation of a string.
                        generated_string_src = (template.replace("{source}", s)
                                                        .replace("{target}", t)
                                                        .replace("{catalyst}", row['catalyst_clean_str']))
                        
                        # 2) Splitting the string into metabolites, that are also cleaned.
                        individual_sources = [met.strip() for met in generated_string_src.split(' + ')]
                        
                        # 3) Each metabolite is added to the list paying attention to duplicates.
                        for met in individual_sources:
                            if met not in seen_sources:
                                sources_mod.append(met)
                                seen_sources.add(met)

            for template in pattern.get("targets", {}).keys():
                for s in sources:
                    for t in targets:
                        # 1) Creation of a string.
                        generated_string_tgt = (template.replace("{source}", s)
                                                        .replace("{target}", t)
                                                        .replace("{catalyst}", row['catalyst_clean_str']))
                        
                        # 2) Splitting the string into metabolites, that are also cleaned.
                        individual_targets = [met.strip() for met in generated_string_tgt.split(' + ')]
                        
                        # 3) Each metabolite is added to the list paying attention to duplicates.
                        for met in individual_targets:
                            if met not in seen_targets:
                                targets_mod.append(met)
                                seen_targets.add(met)

            # 9.3) "reaction_string" and "reaction_id" are created;
            # "source_modified" and "target_modified" are created too by applying the REACTION_PATTERNS to the original sources and targets.
            symbol_template = pattern.get("symbols", {}).get("symbol", "-->") if pattern else "-->"
            catalyst_for_string = row['catalyst_clean_str'].replace(',', ' + ')
            reaction_str = f"{' + '.join(sources_mod)} {symbol_template.replace('{catalyst}', catalyst_for_string)} {' + '.join(targets_mod)}"
            df_main.at[idx, 'source_modified'] = sources_mod
            df_main.at[idx, 'target_modified'] = targets_mod
            df_main.at[idx, 'reaction_string'] = reaction_str
            df_main.at[idx, 'reaction_id'] = f"{'_'.join(sources_mod)}_to_{'_'.join(targets_mod)}"

    # 10) Union of the two dataframes, the one containing "feed" and "sink" reactions (df_feed_sink), 
    # the other containing all other reactions (df_main).
    df_final = pd.concat([df_main, df_feed_sink], ignore_index=True)

    print(f"DataFrame ready with {len(df_final)} reactions. Invalid metabolites found: {len(invalid_df)}")
    df_final['catalyst_modified'] = df_final['catalyst_clean_str'].apply(
        lambda x: x.split(',') if isinstance(x, str) and x else []
    )
    return df_final, invalid_df




# 2.3) FUNCTION TO HANDLE DISEASES AND DUPLICATES.
def add_boundary_and_map_diseases(df: pd.DataFrame, 
                                  config: dict) -> pd.DataFrame:
    
    # 1) Eliminate duplicated "reaction_id" and keep the first occurence.
    print(f"Length of the dataframe before deduplicating reaction_id: {len(df)}")
    df = df.drop_duplicates(subset=['reaction_id'], keep='first').reset_index(drop=True)  
    print(f"Length of the dataframe after deduplicating reaction_id: {len(df)}")


    # 2) Extract list of metabolites from a raw. 
    def extract_list_from_row(row, list_col, str_col, mod_col):
        # 2.1) A list_col is searched; if found, its content is already a list,
        # so it is only cleaned.
        val = row.get(list_col, None)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if str(x).strip()]

        # 2.2) If the list_col has not been found, a str_col is searched; if found, its content 
        # is a string, so the content is divided using comma as separator, it is cleaned and a 
        # list is then generated.
        val = row.get(str_col, None)
        if isinstance(val, str) and val.strip():
            parts = [p.strip() for p in val.split(',') if p.strip()]
            return parts

        # 2.3) If no list_col and no str_col are found, a last attempt is to search a mod_col
        # and treat it as a list.
        val = row.get(mod_col, None)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if str(x).strip()]

        # 2.4) If all previous attempts are failures, an empty list is retuned.
        return []


    # 3) Construct a dictionary for the disease associated to each single metabolite;
    # the function extract_list_from_row is used to extract metabolites from "source_list",
    # "source_clean_str", "source_modified", and from "target_list", "target_clean_str",
    # "target_modified"
    disease_map = {}
    for _, row in df.iterrows():
        d = row.get('disease', "")
        if pd.isna(d) or not str(d).strip():
            continue
        d = str(d).strip()
        srcs = extract_list_from_row(row, 'source_list', 'source_clean_str', 'source_modified')  
        trgs = extract_list_from_row(row, 'target_list', 'target_clean_str', 'target_modified')  
        for m in srcs + trgs:
            if m:
                if m not in disease_map:
                    disease_map[m] = d

    # 4) Diffuse the disease to "feed" and "sink" reactions rows.
    def map_row_disease(row):
        existing = row.get('disease', "")
        if pd.notna(existing) and str(existing).strip():
            return existing

        rtype = row.get('reaction_type', "")
        if str(rtype).lower() == 'feed':
            targets = extract_list_from_row(row, 'target_list', 'target_clean_str', 'target_modified')  
            for t in targets:
                if t in disease_map:
                    return disease_map[t]
            return ""  

        if str(rtype).lower() == 'sink':
            sources = extract_list_from_row(row, 'source_list', 'source_clean_str', 'source_modified') 
            for s in sources:
                if s in disease_map:
                    return disease_map[s]
            return ""  

        all_mets = extract_list_from_row(row, 'source_list', 'source_clean_str', 'source_modified') + \
                   extract_list_from_row(row, 'target_list', 'target_clean_str', 'target_modified')
        for m in all_mets:
            if m in disease_map:
                return disease_map[m]

        return ""

    df['disease'] = df.apply(map_row_disease, axis=1) 
    df = df[df['reaction_id'] != 'output_effect'].reset_index(drop=True)
    print("Mapped diseases and eliminated duplicated reaction_id.")
    return df




###############################################################################
# 3.1) FUNCTION TO LOAD METABOLITES AND REACTIONS INTO COBRA MODEL.
def build_cobra_model(final_df: pd.DataFrame, 
                      metabolites: dict, 
                      config: dict) -> tuple[cobra.Model, dict, list]:
    
    # 1) Creation of the COBRA model.
    model = cobra.Model(config.get("metadata", {}).get("model_id", "default_model"))

    # 2) All metabolites are collected from "source_modified" and "target_modified".
    all_met_ids = set()
    for col in ['source_modified', 'target_modified']:
        if col in final_df.columns:
            for lst in final_df[col].dropna():
                all_met_ids.update(lst)

    # 3) Also "catalyst_modified" are added.
    if 'catalyst_modified' in final_df.columns:
        for lst in final_df['catalyst_modified'].dropna():
            all_met_ids.update(lst)

    # 4) "Metabolite" of COBRA is updated. Previously, metabolites from the 
    # raw metabolite csv file have been inserted; now other metabolites created
    # during preparation of the reaction dataframe are added.
    for met_id in sorted(list(all_met_ids)): 
        if met_id not in metabolites:
            compartment = 'c'
            suffixes = ('_a', '_p', '_unspliced', '_spliced')
            if met_id.endswith(suffixes):
                base_met_id = None
                for suffix in suffixes:
                    if met_id.endswith(suffix):
                        base_met_id = met_id[:-len(suffix)]
                        break 
                if base_met_id and base_met_id in metabolites:
                    compartment = metabolites[base_met_id].compartment
            metabolites[met_id] = Metabolite(met_id, 
                                             name=met_id, 
                                             compartment=compartment)

    # 5) Filling "Reaction" of COBRA with "reaction_id" and creation of a stoichiometry dictionary: 
    # if numbers are present before metabolites name, use them as stoichiometric coefficients;
    # otherwise, use -1 for sources and +1 for targets.
    reactions_to_add = []

    for _, row in final_df.iterrows():
        if not row.get('reaction_id'):
            continue
        
        reaction = Reaction(row['reaction_id'])
        stoichiometry = {}

        # Stoichiometric coefficients of sources.
        for item in row.get('source_modified', []):
            parts = item.split(maxsplit=1)
            coeff = 1.0
            met_id = item
            if len(parts) == 2 and parts[0].replace('.', '', 1).isdigit():
                try:
                    coeff = float(parts[0])
                    met_id = parts[1]
                except ValueError:
                    pass 

            clean_id = clean_metabolite_id(met_id, config)
            if clean_id in metabolites:
                stoichiometry[metabolites[clean_id]] = -coeff

        # Stoichiometric coefficients of targets.
        for item in row.get('target_modified', []):
            parts = item.split(maxsplit=1)
            coeff = 1.0
            met_id = item
            if len(parts) == 2 and parts[0].replace('.', '', 1).isdigit():
                try:
                    coeff = float(parts[0])
                    met_id = parts[1]
                except ValueError:
                    pass
            
            clean_id = clean_metabolite_id(met_id, config)
            if clean_id in metabolites:
                stoichiometry[metabolites[clean_id]] = coeff

        # Stoichiometry is added to COBRA, lowe and upper bounds for an hypothetic 
        # FBA are added too.
        if stoichiometry:   
            reaction.add_metabolites(stoichiometry)
            lb_in = row.get('lower_bound', None)
            ub_in = row.get('upper_bound', None)
            lb, ub = lu_bounds(row.get('reaction_type', ''), lb_in, ub_in, config)
            reaction.lower_bound = lb
            reaction.upper_bound = ub
            reactions_to_add.append(reaction)

    model.add_reactions(reactions_to_add)

    print(f"COBRA model constructed with {len(model.metabolites)} metabolites and {len(model.reactions)} reactions.")
    return model, metabolites, reactions_to_add





###############################################################################
# 4.1) FUNCTION TO SAVE METABOLITES UPDATED IN AN EXCEL FILE.
def save_metabolites_to_excel(metabolites_dict,
                             model = None,
                             output_path = None,
                             config=None):
   
   # 1) Metabolites id cleaning and creation of the list of metabolites.
    met_data = []

    for m in metabolites_dict.values():
        final_id = clean_metabolite_id(m.id, config) 


        if final_id:
            met_data.append({
                "name": m.name,
                "id": final_id,
                "compartment": m.compartment
            })

    met_df = pd.DataFrame(met_data)


    # 2) Based on the metabolites "id", deduplication and sorting is performed.
    met_df = met_df.sort_values(by="id", kind="mergesort").reset_index(drop=True)
    before_csv = len(met_df)
    met_df = met_df.drop_duplicates(subset=["id"], keep="first")
    removed_inside_csv = before_csv - len(met_df)
   
    # 3) Checking whether some metabolites has not been added to Metabolite in COBRA.
    if model:
        existing_ids = {m.id.lower() for m in model.metabolites}
        met_df['status'] = met_df['id'].str.lower().apply(
            lambda x: 'existing' if x in existing_ids else 'new'
        )
    else:
        met_df['status'] = 'new'

    # 4) Add metabolites to model if provided.
    new_mets = []
    if model:
        to_add_df = met_df[met_df['status'] == 'new']
        for _, row in to_add_df.iterrows():
            met_id = row["id"]
            name = row["name"]
            compartment = row["compartment"]
            new_mets.append(Metabolite(id=met_id, name=name, compartment=compartment))
        if new_mets:
            model.add_metabolites(new_mets)


   # 5) Saving cleaned Excel containing metabolites.
    if output_path:
        met_df.to_excel(output_path, index=False)


    # 6) Summary
    new_met_count = (met_df['status'] == 'new').sum()
    skipped_existing = len(met_df) - new_met_count
    print(f"[metabolites] total in DataFrame: {len(met_df)}")
    print(f"[metabolites] removed duplicates inside dataframe: {removed_inside_csv}")
    print(f"[metabolites] already in model (skipped): {skipped_existing}")
    print(f"[metabolites] added to model: {len(new_mets)}")

    return met_df, new_mets




def save_reactions_to_excel(reactions_df: pd.DataFrame, 
                            model: cobra.Model = None, 
                            output_path: str = None) -> pd.DataFrame:

    # 1) Deduplication of reactions based on "reaction_id"
    reactions_df = reactions_df.drop_duplicates(subset=['reaction_id']).copy()
    
    # 2) Identification of reactions already present in the model; if there is 
    # a new reaction, in the new clumn "status" created the value assumed is "new";
    # otherwise, it is "existing".
    if model:
        existing_ids = {r.id for r in model.reactions}
        reactions_df['status'] = reactions_df['reaction_id'].apply(
            lambda x: 'existing' if x in existing_ids else 'new'
        )
        new_reactions_count = (reactions_df['status'] == 'new').sum()
        skipped_existing = len(reactions_df) - new_reactions_count
    else:
        reactions_df['status'] = 'new'
        new_reactions_count = len(reactions_df)
        skipped_existing = 0

    # 3) Saving cleaned Excel containing reactions sorted by "reaction_id".
    if output_path:
        reactions_df_sorted = reactions_df.sort_values(by='reaction_id', ascending=True)
        reactions_df_sorted.to_excel(output_path, index=False)

    # 4) Summary
    print(f"\n[reactions] total in DataFrame: {len(reactions_df)}")
    print(f"[reactions] already in model (skipped): {skipped_existing}")
    print(f"[reactions] new (to be added): {new_reactions_count}")

    return reactions_df





###############################################################################
###############################################################################
##################### IMPORTING and PREPARING FOR COBRA #######################
###############################################################################
###############################################################################

# 0) Paths.
MET_CSV_PATH = config["paths"]["MET_CSV_PATH"]
REACTIONS_CSV_PATH = config["paths"]["REACTIONS_CSV_PATH"]
REACTIONS_XLSX_PATH_prepare = config["paths"]["REACTIONS_XLSX_PATH_prepare"]
REACTIONS_XLSX_PATH_disease = config["paths"]["REACTIONS_XLSX_PATH_disease"]
METABOLITE_XLSX_PATH = config["paths"]["METABOLITE_XLSX_PATH"]
REACTIONS_XLSX_PATH = config["paths"]["REACTIONS_XLSX_PATH"]


# 1) Loading of metabolite and reactions data; definition of valid_ids. 
metabolites_raw, reactions_raw = load_initial_data(metabolites_csv_path = MET_CSV_PATH, 
                                                   reactions_csv_path = REACTIONS_CSV_PATH,
                                                   config = TOML)
valid_ids = set(metabolites_raw.keys())


# 2) Creation of the reaction dafaframe cleaned.
reactions_prepared_df, invalid_metabolites_df = prepare_reaction_dataframe(
    reactions_raw,
    valid_ids,
    TOML
)
reactions_prepared_df.to_excel(REACTIONS_XLSX_PATH_disease, index=False)


# 3) Adding missing diseases and deduplicate.
reactions_with_boundaries_df = add_boundary_and_map_diseases(
    reactions_prepared_df,
    TOML)
reactions_with_boundaries_df.to_excel(REACTIONS_XLSX_PATH_prepare, index=False)


# 4) Adding metabolites and reactions to cobra model.
final_model, final_metabolites, final_reactions = build_cobra_model(
    reactions_with_boundaries_df, 
    metabolites_raw, 
    TOML
)



# 5) Final metaboliets are saved.
save_metabolites_to_excel(final_metabolites,
                          model = final_model,
                          output_path = METABOLITE_XLSX_PATH,
                          config=TOML)

save_reactions_to_excel(reactions_with_boundaries_df,
                        model = final_model,
                        output_path = REACTIONS_XLSX_PATH)


print("Pipeline completed.")



###############################################################################
###############################################################################
###################### CREATION OF SBML FILE FROM COBRA #######################
###############################################################################
###############################################################################

###############################################################################
###############################################################################
###################### CREATION OF SBML FILE FROM COBRA #######################
###############################################################################
###############################################################################

# 1) Substitute NaNs in kinetic constants columns with default values from TOML (not zeros).
default_km = float(TOML.get("kinetics", {}).get("defaults", {}).get("Km_uM", 1.0))
default_kd = float(TOML.get("kinetics", {}).get("defaults", {}).get("Kd_uM", 0.5))
default_kcat = float(TOML.get("kinetics", {}).get("defaults", {}).get("kcat_s", 0.1))

reactions_with_boundaries_df['km_um'] = reactions_with_boundaries_df['km_um'].fillna(default_km)
reactions_with_boundaries_df['kd_um'] = reactions_with_boundaries_df['kd_um'].fillna(default_kd)
reactions_with_boundaries_df['kcat_s'] = reactions_with_boundaries_df['kcat_s'].fillna(default_kcat)


# 2) Save the final model in cobra in SBML format.
output_filename = "GLP1_MODEL_without_kinetics.xml"
cobra.io.write_sbml_model(final_model, output_filename)
print(f"Cobra model saved in the SBML file: {output_filename}")


# 3) Add kinetic constants to the file "GLP1_MODEL_without_kinetics.xml" created.
sbml_reader = libsbml.SBMLReader()
sbml_document = sbml_reader.readSBML(output_filename)
model = sbml_document.getModel()

# Iterrows returns (index, row_series), so we unpack into idx and params
for idx, params in reactions_with_boundaries_df.iterrows():
    reaction_id = params["reaction_id"]
    reaction = model.getReaction(reaction_id)
    
    if reaction is None:
        print(f"Reaction '{reaction_id}' not found in SBML model.")
        continue
    
    kinetic_law = reaction.createKineticLaw()
    
    km_param = kinetic_law.createLocalParameter()
    km_param.setId("km")
    km_param.setValue(params["km_um"])

    kd_param = kinetic_law.createLocalParameter()
    kd_param.setId("kd")
    kd_param.setValue(params["kd_um"])

    kcat_param = kinetic_law.createLocalParameter()
    kcat_param.setId("kcat")
    kcat_param.setValue(params["kcat_s"])

    rtype = str(params.get('reaction_type', '')).lower()

    # Assign math string depending on the reaction type (feed, sink, or standard)
    if rtype == "feed":
        math_string = "kcat"
        ast_node = libsbml.parseL3Formula(math_string)
        kinetic_law.setMath(ast_node)

    elif rtype == "sink" and reaction.getNumReactants() > 0:
        substrate = reaction.getReactant(0).getSpecies()
        math_string = f"kcat * {substrate}"
        ast_node = libsbml.parseL3Formula(math_string)
        kinetic_law.setMath(ast_node)

    elif reaction.getNumReactants() > 0:
        substrate = reaction.getReactant(0).getSpecies()
        math_string = f"kcat * {substrate} / (km + {substrate})"
        ast_node = libsbml.parseL3Formula(math_string)
        kinetic_law.setMath(ast_node)

# 4) Final SBML model with kinetic laws added is saved.
output_kinetic_filename = "GLP1_MODEL_with_kinetics.xml"
sbml_writer = libsbml.SBMLWriter()
sbml_writer.writeSBML(sbml_document, output_kinetic_filename)

print(f"SBML model with kinetic parameters saved in: {output_kinetic_filename}")