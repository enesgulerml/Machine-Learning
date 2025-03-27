import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from warnings import filterwarnings
filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

customer = pd.read_csv("Projects/ML_Project/Why Customer Churn/Customer_Info.csv")
customer.head()
customer.info()
customer.describe().T
customer.isnull().sum()
customer.columns


with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(
        data=customer, x='age', hue='gender', multiple='stack',
        shrink=0.9, alpha=0.85, ax=axes[0], palette="viridis"
    )
    axes[0].set_xlabel('Customer Age', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribution of Customers by Age and Gender', fontsize=12, fontweight='bold')

    age_group = customer[['under_30', 'senior_citizen']].replace({'Yes': 1, 'No': 0})
    age_group['30-65'] = 1 - (age_group.under_30 + age_group.senior_citizen)
    age_group = age_group.sum().reset_index(name='count')

    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    axes[1].pie(
        age_group['count'], labels=age_group['index'], autopct='%1.1f%%', colors=colors, startangle=90,
        wedgeprops={'edgecolor': 'black'}
    )
    axes[1].set_title('Distribution of Customers by Age Groups', fontsize=12, fontweight='bold')

fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4)

plt.show()

################

location = pd.read_csv("Projects/ML_Project/Why Customer Churn/Location_Data.csv")

location.head()
location.info()
location.isnull().sum()
location.columns
location.describe().T

###################

service = pd.read_csv("Projects/ML_Project/Why Customer Churn/Online_Services.csv")

service.head()
service.info()
service.isnull().sum()
service.columns
service.describe().T
service = service.replace({'Yes':1,'No':0})

#### Encoding

service_matrix = service.replace({'Yes':1, 'No':0})
service_matrix = pd.get_dummies(service_matrix,columns=['internet_type'],dtype = int)
corr_matrix = service_matrix.drop(['customer_id'],axis =1).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix, cmap="mako", annot=True, fmt=".2f", linewidths=0.5,
    vmin=-1, vmax=1, cbar=True, square=True, annot_kws={"size": 8}
)
plt.title("Correlation Matrix of Service Subscriptions", fontsize=12, fontweight="bold", pad=12)
plt.xticks(rotation=45, fontsize=9, ha="right")
plt.yticks(fontsize=9)
plt.show()

##########

payment = pd.read_csv("Projects/ML_Project/Why Customer Churn/Payment_Info.csv")

payment.head()
payment.info()
payment.isnull().sum()
payment.columns
payment.describe().T
payment = payment.rename(columns = {'monthly_ charges':'monthly_charges'})

###############

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    cols = [
        'monthly_charges', 'total_charges', 'total_revenue',
        'avg_monthly_long_distance_charges', 'total_long_distance_charges', 'total_extra_data_charges'
    ]
    titles = [
        'Distribution of Monthly Wages', 'Distribution of Total Wages', 'Distribution of Total Income',
        'Monthly Long Distance Fees', 'Total Long Distance Charges', 'Extra Data Charges'
    ]

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']

    for ax, col, title, color in zip(axes.flat, cols, titles, colors):
        sns.histplot(payment[col], ax=ax, color=color, alpha=0.85, edgecolor='black')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)
    plt.show()

service_price = pd.merge(service, payment[["customer_id","monthly_charges"]], on = "customer_id", how = "inner")
service_price.head()

###################

option = pd.read_csv("Projects/ML_Project/Why Customer Churn/Service_Options.csv")

option.head()
option.isnull().sum()
option.columns
option.describe().T
option = option.replace({'Yes':1,'No':0})

#######################

status = pd.read_csv("Projects/ML_Project/Why Customer Churn/Status_Analysis.csv")

status.head()
status.info()
status.isnull().sum()
status.columns

status.groupby(['customer_status']).agg({'satisfaction_score': 'describe', 'cltv': 'mean'})

status_group = status.groupby('customer_status').size().reset_index(name='count')
label_order = ['Churned', 'Joined', 'Stayed']

fig, ax = plt.subplots(1, 3, figsize=(14, 5))

colors = ['#ff9999', '#66b3ff', '#99ff99']
ax[0].pie(
    status_group['count'], labels=status_group['customer_status'], autopct='%1.1f%%',
    colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'}
)
ax[0].set_title("Customer Status Distribution (%)", fontsize=11, fontweight="bold")

sns.boxplot(data=status, x='customer_status', y='satisfaction_score', order=label_order, ax=ax[1], palette='coolwarm')
ax[1].set_title("Customer Status vs. Satisfaction Score", fontsize=11, fontweight="bold")
ax[1].set_xlabel("")
ax[1].set_ylabel("Satisfaction Score", fontsize=10)

sns.boxplot(data=status, x='customer_status', y='cltv', order=label_order, ax=ax[2], palette='viridis')
ax[2].set_title("Customer Status vs. CLTV", fontsize=11, fontweight="bold")
ax[2].set_xlabel("")
ax[2].set_ylabel("CLTV", fontsize=10)

plt.tight_layout()
plt.show()

churn = status[status.customer_status == 'Churned'].drop(columns=['customer_status', 'churn_value'])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']  # Renk paleti
churn_cat = churn.groupby('churn_category').size().reset_index(name='count')
ax[0].pie(
    x=churn_cat['count'], labels=churn_cat['churn_category'], autopct='%1.1f%%',
    colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'}
)
ax[0].set_title("Churn Categories Distribution (%)", fontsize=11, fontweight="bold")

category_order = ['Attitude', 'Competitor', 'Dissatisfaction', 'Other', 'Price']
sns.boxplot(data=churn, x='churn_category', y='satisfaction_score', order=category_order, ax=ax[1], palette='coolwarm')
ax[1].set_xticklabels(labels=category_order, fontsize=9, rotation=20)
ax[1].set_title("Churn Category vs. Satisfaction Score", fontsize=11, fontweight="bold")
ax[1].set_xlabel("")
ax[1].set_ylabel("Satisfaction Score", fontsize=10)

plt.tight_layout()
plt.show()

why_churn = churn.groupby(['churn_category','churn_reason']).size().reset_index(name = 'count')
why_churn

wide_customer = pd.merge(customer, service,on = 'customer_id',how ='inner')
wide_customer = pd.merge(wide_customer, payment,on = 'customer_id',how ='inner')
wide_customer = pd.merge(wide_customer, option,on = 'customer_id',how ='inner')
wide_customer = pd.merge(wide_customer, status,on = 'customer_id',how ='inner')

wide_customer.head()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(
    data = wide_customer, x='age', hue='customer_status', multiple='stack', ax=ax[0],
    palette="viridis", edgecolor="black", alpha=0.9
)
ax[0].set_title('Customer Status and Age Distribution', fontsize=11, fontweight="bold")
ax[0].set_xlabel("Age", fontsize=10)
ax[0].set_ylabel("Number of Customers", fontsize=10)

sns.countplot(
    data = wide_customer, x='gender', hue='customer_status', ax=ax[1], alpha=0.85,
    palette="Set2", edgecolor="black"
)
ax[1].set_title('Customer Status and Age Distribution', fontsize=11, fontweight="bold")
ax[1].set_xlabel("Sex", fontsize=10)
ax[1].set_ylabel("Number of Customers", fontsize=10)

plt.tight_layout()
plt.show()

## Modeling

x = wide_customer.copy()

x['tech_service'] = x.online_security + x.online_backup + x.device_protection + x.premium_tech_support
x['streaming_service'] = x.streaming_tv + x.streaming_movies + x.streaming_music

x = x[['age','number_of_dependents','internet_type','phone_service_x','streaming_service','tech_service','contract',
       'unlimited_data','number_of_referrals','satisfaction_score']]

x = pd.get_dummies(x, columns = ['contract','internet_type'], dtype = int, drop_first=True)


y = wide_customer[['churn_value']].astype(int)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 23)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train.values.ravel())

y_pred2 = model.predict(x_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred2))

#########

cm2 = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm2, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

coef = model.coef_
predictor = x.columns
predictor_coef = pd.DataFrame({'predictor':predictor, 'coef': coef[0]})
predictor_coef.sort_values(by='coef', ascending=False)
predictor_coef

