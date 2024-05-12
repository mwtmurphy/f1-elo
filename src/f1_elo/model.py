import itertools
import json
import typing
import yaml

import bayes_opt
import numpy as np
import pandas as pd


# global variables
with open("params.yaml") as conf_file:
    CONFIG = yaml.safe_load(conf_file)

MOD_DF = pd.read_csv(CONFIG["data"]["features_path"])
MOD_DF[["constructorScore", "driverScore", "expected", "actual"]] = None
IX_CHUNKS = MOD_DF.reset_index().groupby(["year", "round"])["index"].agg(["min", "max"]).values
MOD_MAT = MOD_DF.values

DRI_RTG = {dri: CONFIG["model"]["start_score"] for dri in set(MOD_DF["driverId"])}
CON_RTG = {con: CONFIG["model"]["start_score"] for con in set(MOD_DF["constructorId"])}

def model_data(k: float, c: float, w: float, export: bool = False) -> typing.Union[float, None]:
    '''If export == False, returns negative RMSEE based on params. 
    If export == True, exports modelled data to 'interim' data folder 
    for data reporting.'''

    dri_scores = DRI_RTG.copy()
    con_scores = CON_RTG.copy()
    exp, out = [], []

    for start_ix, end_ix in IX_CHUNKS:
        yr_mat = MOD_MAT[start_ix:end_ix]
        rnd_dri_scores = {dri: {"diff": 0, "n": 0, "exp": 0, "act": 0} for dri in yr_mat[:, 4]}
        rnd_con_scores = {con: {"diff": 0, "n": 0, "exp": 0, "act": 0} for con in yr_mat[:, 3]}

        for ix_1, ix_2 in itertools.combinations(range(yr_mat.shape[0]), 2):
            con_a, dri_a, pos_a, st_a = yr_mat[ix_1, [3, 4, 5, 7]]
            con_b, dri_b, pos_b, st_b = yr_mat[ix_2, [3, 4, 5, 7]]
    
            # continue if drivers in same car or a driver does not finish for misc reason
            if pos_a == pos_b or "misc retirement" in [st_a, st_b]:
                continue

            # get current rating
            elo_a = dri_scores[dri_a] + (w * con_scores[con_a])
            elo_b = dri_scores[dri_b] + (w * con_scores[con_b])
            
            # calculate position influence
            q_a = 10 ** (elo_a / c)
            q_b = 10 ** (elo_b / c)
    
            e_a = q_a / (q_a + q_b)        
            e_b = q_b / (q_a + q_b)

            # score outcome
            if pos_a < pos_b:
                o_a = 1
                o_b = 0
            else:
                o_a = 0
                o_b = 1
                
            # calculate score change and update round scores
            diff_a = k * (o_a - e_a)
            diff_b = k * (o_b - e_b)

            # log driver results and changes if neither retire due to car failure (not attributable to drivers)
            if "constructor retirement" not in [st_a, st_b]:
                rnd_dri_scores[dri_a]["exp"] += e_a
                rnd_dri_scores[dri_a]["act"] += o_a
                rnd_dri_scores[dri_a]["diff"] += diff_a
                rnd_dri_scores[dri_a]["n"] += 1
            
                rnd_dri_scores[dri_b]["exp"] += e_a
                rnd_dri_scores[dri_b]["act"] += o_a
                rnd_dri_scores[dri_b]["diff"] += diff_b
                rnd_dri_scores[dri_b]["n"] += 1
            
            # log constructor changes if diff constructors and neither driver retires due to driver error (not attributable to constructors)
            if con_a != con_b and "driver_retirement" not in [st_a, st_b]:
                rnd_con_scores[con_a]["diff"] += diff_a
                rnd_con_scores[con_a]["n"] += 1
                rnd_con_scores[con_b]["diff"] += diff_b
                rnd_con_scores[con_b]["n"] += 1
                
            # store expected and final values for error analysis
            exp += [e_a, e_b]
            out += [o_a, o_b]
        
        # update driver values for finishing drivers and driver-caused retirements
        for dri in rnd_dri_scores.keys():
            if rnd_dri_scores[dri]["n"] != 0: # more than 1 car on grid
                dri_scores[dri] += (rnd_dri_scores[dri]["diff"] / rnd_dri_scores[dri]["n"])
                
        yr_mat[:, 9] = list(map(lambda el: dri_scores[el], yr_mat[:, 4])) # driver score
        yr_mat[:, 10] = list(map(lambda el: rnd_dri_scores[el]["exp"], yr_mat[:, 4])) # expected outcome
        yr_mat[:, 11] = list(map(lambda el: rnd_dri_scores[el]["act"], yr_mat[:, 4])) # actual outcome

        # update constructor values for finishing drivers
        for con in rnd_con_scores.keys():
            if rnd_con_scores[con]["n"] != 0: # more than 1 car on grid
                con_scores[con] += (rnd_con_scores[con]["diff"] / rnd_con_scores[con]["n"])
        
        yr_mat[:, 8] = list(map(lambda el: con_scores[el], yr_mat[:, 3]))

    if export == False:
        err_df = pd.DataFrame({"pred": exp, "true": out})
        err_df["squared_error"] = (err_df["true"] - err_df["pred"]) ** 2
        neg_rmse = -(pow(err_df["squared_error"].sum() / err_df.shape[0], 0.5))
        
        return neg_rmse
    
    else:
        RES_DF = pd.DataFrame(MOD_MAT, columns=MOD_DF.columns)
        RES_DF.to_csv(CONFIG["data"]["modelled_path"], index=False)
        

if __name__=="__main__":

    # find optimal parameters
    opt_params = CONFIG["model"]["opt_params"]
    optimiser = bayes_opt.BayesianOptimization(
        f=model_data,
        pbounds=opt_params["pbounds"],
        random_state=opt_params["random_state"],
        allow_duplicate_points=True
    )

    optimiser.maximize(
        init_points=opt_params["init_points"], 
        n_iter=opt_params["n_iter"]
    )
    results = optimiser.max
    params_log = {
        "k": float(results["params"]["k"]),
        "c": float(results["params"]["c"]),
        "w": float(results["params"]["w"])
    }
    metrics_log = {
        "RMSE": float(-results["target"])
    }

    #log metrics and params locally
    with open(CONFIG["data"]["metrics_path"], "w") as out:
        json.dump(metrics_log, out)

    with open(CONFIG["data"]["params_path"], "w") as out:
        yaml.dump(params_log, out)

    # create final model data
    model_data(k=params_log["k"], c=params_log["c"], w=params_log["w"], export=True)
