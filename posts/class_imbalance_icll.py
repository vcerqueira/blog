import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier as RFC

from src.icll import ICLL

X, y = make_classification(n_samples=500, n_features=5, n_informative=3)
X = pd.DataFrame(X)

icll = ICLL(model_l1=RFC(), model_l2=RFC())
icll.fit(X, y)

probs = icll.predict_proba(X)

# resampling alternative

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN

X_train, y_train = make_classification(n_samples=500, n_features=5, n_informative=3)

X_res, y_res = SMOTE().fit_resample(X_train, y_train)

# comparisons

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from plotnine import *


data = pd.read_csv('data/pima.csv')

X, y = data.drop('target', axis=1), data['target']
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_res, y_res = SMOTE().fit_resample(X_train, y_train)

rf = RFC()
smote = RFC()
icll = ICLL(model_l1=RFC(), model_l2=RFC())

rf.fit(X_train, y_train)
smote.fit(X_res, y_res)
icll.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)
smote_probs = smote.predict_proba(X_test)
icll_probs = icll.predict_proba(X_test)

print(roc_auc_score(y_test, rf_probs[:, 1]))
print(roc_auc_score(y_test, smote_probs[:, 1]))
print(roc_auc_score(y_test, icll_probs))

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs[:, 1])
fpr_sm, tpr_sm, _ = roc_curve(y_test, smote_probs[:, 1])
fpr_ic, tpr_ic, _ = roc_curve(y_test, icll_probs)

roc_rf = pd.DataFrame({'fpr': fpr_rf, 'tpr': tpr_rf})
roc_sm = pd.DataFrame({'fpr': fpr_sm, 'tpr': tpr_sm})
roc_icll = pd.DataFrame({'fpr': fpr_ic, 'tpr': tpr_ic})
roc_rf['Model'] = 'RF'
roc_sm['Model'] = 'SMOTE'
roc_icll['Model'] = 'ICLL'

df = pd.concat([roc_sm, roc_icll], axis=0)

roc_plt = ggplot(df) + \
          aes(x='fpr', y='tpr', group='Model', color='Model') + \
          theme_classic(base_family='Palatino', base_size=12) + \
          theme(plot_margin=.125,
                axis_text=element_text(size=10),
                legend_title=element_blank(),
                legend_position='top') + \
          geom_line(size=1.7) + \
          xlab('False Positive Rate') + \
          ylab('True Positive Rate') + \
          ylim(0, 1) + xlim(0, 1) + \
          ggtitle('') + \
          geom_abline(intercept=0,
                      slope=1,
                      size=1,
                      color='black',
                      linetype='dashed')

print(roc_plt)

# roc_plt.save(f'{output_dir}/dist_plot.pdf', height=5, width=8)
