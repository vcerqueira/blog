# getting the importance of each feature in each horizon
avg_imp = pd.DataFrame([x.feature_importances_
                        for x in model_w_fe.estimators_]).mean()

# getting the top 100 features
n_top_features = 100

importance_scores = pd.Series(dict(zip(X_tr.columns, avg_imp)))
top_features = importance_scores.sort_values(ascending=False)[:n_top_features]
top_features_nm = top_features.index

# subsetting training and testing sets by those features
X_tr_top = X_tr[top_features_nm]
X_ts_top = X_ts[top_features_nm]

# re-fitting the lgbm model
model_top_features = MultiOutputRegressor(LGBMRegressor())
model_top_features.fit(X_tr_top, Y_tr)

# getting forecasts for the test set
preds_top_feats = model_top_features.predict(X_ts_top)

# computing MAE error
print(mape(Y_ts, preds_top_feats))
# 0.230

######### Plotting feature importpance

from plotnine import *

imp_df = importance_scores.sort_values(ascending=True)[-20:].reset_index()
imp_df.columns = ['Feature', 'Importance']
imp_df['Feature'] = pd.Categorical(imp_df['Feature'], categories=imp_df['Feature'])

plot = ggplot(imp_df, aes(x='Feature', y='Importance')) + \
       geom_bar(fill='#58a63e', stat='identity', position='dodge') + \
       theme_classic(
           base_family='Palatino',
           base_size=12) + \
       theme(
           plot_margin=.25,
           axis_text=element_text(size=8),
           axis_text_x=element_text(size=8),
           axis_title=element_text(size=8),
           legend_text=element_text(size=8),
           legend_title=element_text(size=8),
           legend_position='top') + \
       xlab('') + \
       ylab('Importance') + coord_flip()

plot.save('feature_importance.pdf', height=7, width=5)
