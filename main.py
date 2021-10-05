import pandas as pd
import random
import pickle
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import *
from matplotlib import pyplot as plt

from preferences import notas_pref
from ahp import ahp
from data_preparation import create_subsample
from fine_tunning import fine_tunning
from data_preparation import merge_matrices
from tau_distance import normalised_kendall_tau_distance

len_Q = 5  # n_samples to be evaluated
CV = 5  # number of cross-validation
test_size = 0.2  # 80% train and 20% test
accepted_error = .05  # max tau distance accepted between current ranking and the predicted one

df_var = pd.read_csv("dec_5obj_p2.csv", header=None)  # decision variables
# df_var = df_var.iloc[0:55, :].round(5)
df_obj = pd.read_csv('obj_5obj_p2.csv', header=None)  # values in Pareto front
# df_obj = df_obj.iloc[0:55, :].round(5)

npop, nvar = df_var.shape
nobj = df_obj.shape[1]

# Generate the preferences
df_obj = df_obj.to_numpy()
df_pref = notas_pref(df_obj)

# AHP from the original alternatives
rank_ahp = ahp(df_pref).index

# Generate the index to be evaluated
index = list(df_var.index)

# Aleatory ranking
aleatory = index.copy()
random.shuffle(aleatory)

# Start an aleatory ranking
rank_aleatory = aleatory.copy()

# Distances
current_previous = []
current_ahp = []

# Metrics
mse = []
rmse = []
r2 = []
mape = []

# Iterations
iteration = []
cont = 0

temp = 1
for aux in tqdm(range(len_Q, npop, len_Q)):
    cont += 1

    # Define Q and N-Q indexes
    Q_index = aleatory[0:aux]
    N_Q_index = [x for x in index if x not in Q_index]

    # Train
    df_Q = create_subsample(df_var=df_var, df_pref=df_pref, nobj=nobj, index=Q_index)
    X_train = df_Q.iloc[:, :-nobj]  # to predict
    y_train = df_Q.iloc[:, -nobj:]  # real targets
    # Test
    df_N_Q = create_subsample(df_var=df_var, df_pref=df_pref, nobj=nobj, index=N_Q_index)
    X_test = df_N_Q.iloc[:, :-nobj]  # to predict
    y_test = df_N_Q.iloc[:, -nobj:]  # real targets

    # Model training
    if temp > accepted_error:
        tuned_model = fine_tunning(CV, X_train, y_train)
        with open("tuned_model_cbic_5obj.pkl", 'wb') as arq:  # Save best model
            pickle.dump(tuned_model, arq)
        tuned_model.fit(X_train, y_train)
    else:
        with open("tuned_model_cbic_5obj.pkl", "rb") as fp:  # Load trained model
            tuned_model = pickle.load(fp)

    # Model evaluation
    y_pred = tuned_model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)

    # Metrics
    mse.append(mean_squared_error(pd.DataFrame(y_test.values), pd.DataFrame(y_pred.values), squared=True))
    rmse.append(mean_squared_error(pd.DataFrame(y_test.values), pd.DataFrame(y_pred.values), squared=False))
    r2.append(r2_score(pd.DataFrame(y_test.values), pd.DataFrame(y_pred.values)))
    mape.append(mean_absolute_percentage_error(pd.DataFrame(y_test.values), pd.DataFrame(y_pred.values)))

    # Merge the predictions of the df train and df test
    df_merged = merge_matrices(N_Q_index, df_pref, y_pred)

    # Employ AHP in the predicted (mixed with preferences) dataset
    rank_predicted = ahp(df_merged).index

    # Calculate distances
    temp = normalised_kendall_tau_distance(r1=rank_aleatory, r2=rank_predicted)
    current_previous.append(temp)
    current_ahp.append(normalised_kendall_tau_distance(r1=rank_ahp, r2=rank_predicted))

    # df_obj = pd.DataFrame(df_obj)
    # plt.scatter(df_obj.loc[:, 0], df_obj.loc[:, 1], color='b')  # available
    # plt.scatter(df_obj.loc[rank_predicted[0:aux], 0], df_obj.loc[rank_predicted[0:aux], 1], color='r',
    #             marker='^')  # top ranked
    # plt.scatter(df_obj.loc[rank_ahp[0:aux], 0], df_obj.loc[rank_ahp[0:aux], 1], color='g', marker='*')  # ahp
    # plt.legend(["Available", "Top ranked", 'AHP'])
    # plt.show()

    # Update the ranking
    rank_aleatory = rank_predicted

    # Storage the iterations
    iteration.append(cont)

    # if cont == 10:
    #     break

# Merge the results
results = pd.DataFrame({'Iteration': iteration,
                        'MSE': mse,
                        'RMSE': rmse,
                        'R2': r2,
                        'MAPE': mape,
                        'Current_Previous': current_previous,
                        'Current_AHP': current_ahp})

results_metrics = results[['Iteration', 'MSE', 'RMSE', 'R2', 'MAPE']]
results_tau = results[['Iteration', 'Current_Previous', 'Current_AHP']]

fig, ax = plt.subplots()
sns.lineplot(x='Iteration',
             y='value',
             hue='variable',
             data=pd.melt(results_metrics, "Iteration"))
ax.legend(["MSE", "RMSE", 'R2', 'MAPE'])
plt.ylabel("Error metric")
# plt.show()
plt.savefig('error_metric_cbic_5obj__.png')
plt.close(fig)    # close the figure window

fig, ax = plt.subplots()
sns.lineplot(x='Iteration',
             y='value',
             hue='variable',
             data=pd.melt(results_tau, "Iteration"))
ax.legend(["Current vs Previous", 'Current vs AHP'])
ax.set_ylim(0, 1)
plt.ylabel("Tau similarity")
plt.axhline(y=.05, ls=':', color='red', marker='*')
# plt.show()
plt.savefig('similarity_cbic_5obj__.png')
plt.close(fig)    # close the figure window

# Select top10 to plot
top10_pred = rank_predicted[0:10]
select_top10 = pd.DataFrame(df_obj)
select_top10 = select_top10.iloc[top10_pred, :]
select_top10.round(4)
select_top10.to_csv("selected_top10_cbic_5obj_.csv")
