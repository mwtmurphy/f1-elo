import itertools
import json
import typing
import yaml

import bayes_opt
import numpy as np
import pandas as pd
from sklearn import linear_model

# global variables
with open("params.yaml") as conf_file:
    CONFIG = yaml.safe_load(conf_file)

MOD_DF = pd.read_csv(CONFIG["data"]["features_path"])
MOD_DF[["constructorScore", "driverScore", "expected", "actual"]] = None
IX_CHUNKS = MOD_DF.reset_index().groupby(["year", "round"])["index"].agg(["min", "max"]).values
MOD_MAT = MOD_DF.values

DRI_RTG = {dri: CONFIG["model"]["start_score"] for dri in set(MOD_DF["driverId"])}
CON_RTG = {con: CONFIG["model"]["start_score"] for con in set(MOD_DF["constructorId"])}

def model_data(k: float, s: float, w: float, prob_model = None) ->pd.DataFrame:
    '''If export == False, returns negative RMSEE based on params. 
    If export == True, exports modelled data to 'interim' data folder 
    for data reporting.'''

    dri_scores = DRI_RTG.copy()
    con_scores = CON_RTG.copy()
    rating_diffs = []
    true_outcomes = []
    log_losses = []
    pred_errors = []

    for start_ix, end_ix in IX_CHUNKS:
        yr_mat = MOD_MAT[start_ix:end_ix+1]
        rnd_dri_scores = {dri: {"diff": 0, "n": 0, "exp": 0, "act": 0} for dri in yr_mat[:, 4]}
        rnd_con_scores = {con: {"diff": 0, "n": 0, "exp": 0, "act": 0} for con in yr_mat[:, 3]}

        for ix_1, ix_2 in itertools.combinations(range(yr_mat.shape[0]), 2):
            con_a, dri_a, pos_a, st_a = yr_mat[ix_1, [3, 4, 5, 7]]
            con_b, dri_b, pos_b, st_b = yr_mat[ix_2, [3, 4, 5, 7]]
    
            # continue if drivers in same car or a driver does not finish for misc reason
            if pos_a == pos_b or st_a != "finished" or st_b != "finished":
                continue

            # get expected score
            r_a = dri_scores[dri_a] + (w * con_scores[con_a])
            r_b = dri_scores[dri_b] + (w * con_scores[con_b])
            diff_r = r_a - r_b

            e_a = float(prob_model.predict_proba(np.array(diff_r).reshape(1, -1))[0, 1]) if prob_model else 1 / (1 + 10 ** (-diff_r / s))
            e_b = 1 - e_a

            # get outcome score
            if pos_a < pos_b:
                o_a = 1
                o_b = 0
                log_losses.append(-np.log(e_a))

            else:
                o_a = 0
                o_b = 1
                log_losses.append(-np.log(e_b))
                
            pred_errors.append(o_a - e_a)
            
            # store outcomes for log reg model
            rating_diffs += [diff_r, -diff_r]
            true_outcomes += [o_a, o_b]

            # calculate score change and update round scores
            diff_a = k * (o_a - e_a)
            diff_b = k * (o_b - e_b)

            # log driver results and changes if neither retire due to car failure (not attributable to drivers)
            if "constructor retirement" not in [st_a, st_b]:
                rnd_dri_scores[dri_a]["exp"] += e_a
                rnd_dri_scores[dri_a]["act"] += o_a
                rnd_dri_scores[dri_a]["diff"] += diff_a
                rnd_dri_scores[dri_a]["n"] += 1

                rnd_dri_scores[dri_b]["exp"] += e_b
                rnd_dri_scores[dri_b]["act"] += o_b
                rnd_dri_scores[dri_b]["diff"] += diff_b
                rnd_dri_scores[dri_b]["n"] += 1
            
            # log constructor changes if diff constructors and neither driver retires due to driver error (not attributable to constructors)
            if con_a != con_b and "driver retirement" not in [st_a, st_b]:
                rnd_con_scores[con_a]["diff"] += w * diff_a
                rnd_con_scores[con_a]["n"] += 1
    
                rnd_con_scores[con_b]["diff"] += w * diff_b
                rnd_con_scores[con_b]["n"] += 1
        
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
        
    # print summary errors
    rmse = np.sqrt(np.mean(np.square(pred_errors)))
    mean_loss = np.mean(log_losses)

    return rmse, mean_loss, rating_diffs, true_outcomes
        

if __name__=="__main__":
    K = 32
    S = 400
    W = 1.25

    for K in [22, 32, 42]:
        for S in [300, 400, 500]:
            for W in [0.9, 1, 1.1]:

                _, mean_loss, x, y = model_data(K, S, W)
                prev_mean_loss = 1
                n = 2
                prob_model = linear_model.LogisticRegression()

                while n <= 40 and prev_mean_loss - mean_loss >= 0.0001:
                    prev_mean_loss = mean_loss
                    x = np.array(x).reshape(-1, 1)
                    prob_model.fit(x, y)
                    rmse, mean_loss, x, y = model_data(K, S, W, prob_model=prob_model)
                    n += 1

                print(f"[*] K={K}, S={S}, W={W}, RMSE={round(rmse, 4)}, MLOSS={round(mean_loss, 4)}")


    RES_DF = pd.DataFrame(MOD_MAT, columns=MOD_DF.columns)
    RES_DF.to_csv(CONFIG["data"]["modelled_path"], index=False)

    params_log = {
        "k": K,
        "s": S,
        "w": W
    }
    metrics_log = {
        "RMSE": rmse,
        "mean_loss": mean_loss
    }

    #log metrics and params locally
    with open(CONFIG["data"]["metrics_path"], "w") as out:
        json.dump(metrics_log, out)

    with open(CONFIG["data"]["params_path"], "w") as out:
        yaml.dump(params_log, out)
