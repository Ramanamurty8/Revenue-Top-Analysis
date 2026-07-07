import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


## Load Data 
df=pd.read_csv('testdata.csv')

print(df.shape) 
print(df.head(10))
print(df.dtypes)
df['site']=df['site'].astype('category')
print(df.isna().sum())
print(df.describe())
df.corr(numeric_only=True)

df.revenue.describe

plt.hist(df['revenue'], bins=30)

plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.title("Distribution of Revenue")
plt.show()

plt.boxplot(df['revenue'])

plt.xlabel("Revenue")
plt.show()

df.top.describe()

plt.hist(df['top'], bins=30)

plt.xlabel("TOP")
plt.ylabel("Frequency")
plt.title("Distribution of TOP")
plt.show()

df['browser'].value_counts(normalize=True)*100
df['platform'].value_counts(normalize=True)*100
df['site'].value_counts(normalize=True)*100
k=df['site'].value_counts(normalize=True)*100
plt.bar(k.index,k.values)
plt.xlabel("SITE")
plt.ylabel("%")
plt.title("Distribution of SITE")
plt.show()

k1=df['platform'].value_counts(normalize=True)*100
plt.bar(k1.index,k1.values)
plt.xlabel("PLATFORM")
plt.ylabel("%")
plt.title("Distribution of PLATFORM")
plt.show()

k2=df['browser'].value_counts(normalize=True)*100
plt.bar(k2.index,k2.values)
plt.xlabel("BROWSER")
plt.ylabel("%")
plt.title("Distribution of BROWSER")
plt.show()

plt.scatter(df.top,df.revenue)
plt.xlabel('TOP')
plt.ylabel('REVENUE')
plt.title('REV VS TOP')
plt.show()

df['top_bin'] = pd.qcut(df['top'], q=10)

summary = df.groupby('top_bin').agg(med_top=('top','median'),med_revenue=('revenue','median'),count=('revenue','size'))


summary

plt.plot(summary['med_top'],summary['med_revenue'],marker='o')

plt.xlabel("Median TOP")
plt.ylabel("Median Revenue")
plt.title("Revenue vs Time on Page (Deciles)")
plt.show()

summary['pcr']=(summary.med_revenue.pct_change()*100).round(2)
summary

X = df[['top']]      
y = df['revenue']


X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

df.groupby('browser')[['top','revenue']].describe()

df.groupby('browser')[['top','revenue']].quantile([0.10,0.25,0.50,0.75,0.90])

df.boxplot(column='revenue', by='browser')

df.boxplot(column='top', by='browser')

df.groupby('browser')[['top','revenue']].corr()

sns.scatterplot( data=df, x='top', y='revenue', hue='browser' )

browser_summary = df.groupby(['browser',pd.qcut(df['top'],10)]).agg(med_top=('top','median'),med_revenue=('revenue','median'))
browser_summary

df.groupby('platform')[['top','revenue']].describe()
df.groupby('platform')[['top','revenue']].quantile([0.10,0.25,0.50,0.75,0.90])
df.boxplot(column='revenue', by='platform')
df.boxplot(column='top', by='platform')
df.groupby('platform')[['top','revenue']].corr()
sns.scatterplot( data=df, x='top', y='revenue', hue='platform')
platform_summary = df.groupby(['platform',pd.qcut(df['top'],10)]).agg(med_top=('top','median'),med_revenue=('revenue','median'))
platform_summary

df.groupby('site')[['top','revenue']].describe()
df.groupby('site')[['top','revenue']].quantile([0.10,0.25,0.50,0.75,0.90])

df.boxplot(column='revenue', by='site')
df.boxplot(column='top', by='site')
df.groupby('site')[['top','revenue']].corr()
sns.scatterplot( data=df, x='top', y='revenue', hue='site')
site_summary = df.groupby(['site',pd.qcut(df['top'],10)]).agg(med_top=('top','median'),med_revenue=('revenue','median'))

site_summary

df.groupby(['browser', 'platform'])[['top', 'revenue']].corr()
df.groupby(['browser', 'platform','site'])[['top', 'revenue']].corr()

X_cat=df[['browser','platform','site']]
X_cat1=pd.get_dummies(X_cat,drop_first=True)

X=pd.concat([df['top'],X_cat1],axis=1)
     
y = df['revenue']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

df["pred"] = model.fittedvalues
df["resid"] = model.resid

plt.scatter(df["pred"], df["resid"])
plt.axhline(0, color="red")
plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")
plt.title("Residual vs Fitted")
plt.show()

model = smf.ols(
    "revenue ~ top*C(browser)+ top*C(platform) + top*C(site)",
    data=df
).fit(cov_type="HC3")

print(model.summary())

df["pred"] = model.fittedvalues
df["resid"] = model.resid

plt.scatter(df["pred"], df["resid"])
plt.axhline(0, color="red")
plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")
plt.title("Residual vs Fitted")
plt.show()

model = smf.ols("revenue ~ top*C(browser) + top*C(platform) + top*C(site)",data=df).fit(cov_type="HC3")

print(model.summary())
df["pred"] = model.fittedvalues
df["resid"] = model.resid

plt.scatter(df["pred"], df["resid"])
plt.axhline(0, color="red")
plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")
plt.title("Residual vs Fitted")
plt.show()

model = smf.ols("revenue ~ (top)*C(browser)*C(platform) + C(site)",data=df).fit(cov_type="HC3")

print(model.summary())

df["pred"] = model.fittedvalues
df["resid"] = model.resid

plt.scatter(df["pred"], df["resid"])
plt.axhline(0, color="red")
plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")
plt.title("Residual vs Fitted")
plt.show()

model = smf.ols("revenue ~ top*C(browser)*C(platform)*C(site)",data=df).fit(cov_type="HC3")

print(model.summary())

df["pred"] = model.fittedvalues
df["resid"] = model.resid

plt.scatter(df["pred"], df["resid"])
plt.axhline(0, color="red")
plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")
plt.title("Residual vs Fitted")
plt.show()

