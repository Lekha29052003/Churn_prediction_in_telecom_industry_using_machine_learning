#importing the required packages
import pandas as pd
import numpy as np

#import visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

#mount the drive
from google.colab import drive
drive.mount('/content/drive')

#insert the data file
working_dir_path='/content/drive/MyDrive/CAPSTONE PROJECT - ALMA BETTER/'
telecom_df= pd.read_csv(working_dir_path + 'Telecom Churn.csv')

# Viewing the data of top 5 rows to look the glimps of the data
telecom_df.head(5)

# View the data of bottom 5 rows to look the glimps of the data
telecom_df.tail(5)

#Getting the shape of dataset with rows and columns
print(telecom_df.shape)

#Getting the data types of all the columns
telecom_df.dtypes

#check details about the data set
telecom_df.info()
telecom_df.nunique()

#Looking for the description of the dataset to get insights of the data
telecom_df.describe(include='all')

#Printing the count of true and false in 'churn' feature
print(telecom_df.Churn.value_counts())

#check for count of missing values in each column.
telecom_df.isna().sum()
telecom_df.isnull().sum()

missing = pd.DataFrame((telecom_df.isnull().sum())*100/telecom_df.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()

# Checking Duplicate Values
len(telecom_df[telecom_df.duplicated()])

#Printing the unique value inside "churn" column
telecom_df["Churn"].unique()

#Printing the count of true and false in 'churn' feature
print(telecom_df.Churn.value_counts())

#Printing the count of true and false in 'churn' feature
print(telecom_df.Churn.value_counts())

#To get the Donut Plot to analyze churn
data = telecom_df['Churn'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',shadow=True,radius = 2.0, labels =
['Not churned customer','Churned customer'],colors=['royalblue' ,'lime'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for Churn')
plt.show()

#let's see churn by using countplot
sns.countplot(x=telecom_df.Churn)

#printing the unique value of sate column
telecom_df['State'].nunique()

#Comparison churn with state by using countplot
sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
ax = sns.countplot(x='State', hue="Churn", data=telecom_df)
plt.show()

s1=telecom_df['State'].unique()
s2=telecom_df.groupby(['State'])['Churn'].mean()
plt.rcParams['figure.figsize'] = (18, 7)
plt.plot(s1,s2,color='r', marker='o', linewidth=2, markersize=12)
plt.title(" States churn rate", fontsize = 20)
plt.xlabel('state', fontsize = 15)
plt.ylabel('churn rate', fontsize = 15)
plt.show()

plt.rcParams['figure.figsize'] = (12, 7)
color = plt.cm.copper(np.linspace(0, 0.5, 20))
((telecom_df.groupby(['State'])['Churn'].mean())*100).sort_values(ascending =
False).head(6).plot.bar(color = ['violet','indigo','b','g','y','orange','r'])
plt.title(" State with most churn percentage", fontsize = 20)
plt.xlabel('state', fontsize = 15)
plt.ylabel('percentage', fontsize = 15)
plt.show()

#calculate State vs Churn percentage
State_data = pd.crosstab(telecom_df["State"],telecom_df["Churn"])
State_data['Percentage_Churn'] = State_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(State_data)

#show the most churn state of top 10 by ascending the above list
telecom_df.groupby(['State'])['Churn'].mean().sort_values(ascending = False).head(10)

#calculate Area code vs Churn percentage
Area_code_data = pd.crosstab(telecom_df["Area code"],telecom_df["Churn"])
Area_code_data['Percentage_Churn'] = Area_code_data.apply(lambda x :
x[1]*100/(x[0]+x[1]),axis = 1)
print(Area_code_data)

sns.set(style="darkgrid")
ax = sns.countplot(x='Area code', hue="Churn", data=telecom_df)
plt.show()

#Separating churn and non churn customers
churn_df = telecom_df[telecom_df["Churn"] == bool(True)]
not_churn_df = telecom_df[telecom_df["Churn"] == bool(False)]

#Account length vs Churn
sns.distplot(telecom_df['Account length'])

#comparison of churned account length and not churned account length
sns.distplot(telecom_df['Account length'],color = 'yellow',label="All")
sns.distplot(churn_df['Account length'],color = "red",hist=False,label="Churned")
sns.distplot(not_churn_df['Account length'],color = 'green',hist= False,label="Not churned")
plt.legend()

#Show count value of 'yes','no'
telecom_df['International plan'].value_counts()

#Show the unique data of "International plan"
telecom_df["International plan"].unique()

#Calculate the International Plan vs Churn percentage
International_plan_data = pd.crosstab(telecom_df["International plan"],telecom_df["Churn"])
International_plan_data['Percentage Churn'] = International_plan_data.apply(lambda x :
x[1]*100/(x[0]+x[1]),axis = 1)
print(International_plan_data)

#To get the Donut Plot to analyze International Plan
data = telecom_df['International plan'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',shadow=True,radius = 2.0, labels =
['No','Yes'],colors=['skyblue' ,'orange'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for International plan')
plt.show()


#Analysing by using countplot
sns.countplot(x='International plan',hue="Churn",data = telecom_df)

#show the unique value of the "Voice mail plan" column
telecom_df["Voice mail plan"].unique()

#Calculate the Voice Mail Plan vs Churn percentage
Voice_mail_plan_data = pd.crosstab(telecom_df["Voice mail plan"],telecom_df["Churn"])
Voice_mail_plan_data['Percentage Churn'] = Voice_mail_plan_data.apply(lambda x :
x[1]*100/(x[0]+x[1]),axis = 1)
print(Voice_mail_plan_data)

#To get the Donut Plot to analyze Voice mail plan
data = telecom_df['Voice mail plan'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',startangle=90,shadow=True,radius =
2.0, labels = ['NO','YES'],colors=['skyblue','red'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for Voice mail plan')
plt.show()

#Analysing by using countplot
sns.countplot(x='Voice mail plan',hue="Churn",data = telecom_df)

#show the data of 'Number vmail messages'
telecom_df['Number vmail messages'].unique()

#Printing the data of 'Number vmail messages'
telecom_df['Number vmail messages'].value_counts()

#Show the details of 'Number vmail messages' data
telecom_df['Number vmail messages'].describe()

#Analysing by using displot diagram
sns.distplot(telecom_df['Number vmail messages'])

#Analysing by using boxplot diagram between 'number vmail messages' and 'churn'
fig = plt.figure(figsize =(10, 8))
telecom_df.boxplot(column='Number vmail messages', by='Churn')
fig.suptitle('Number vmail message', fontsize=14, fontweight='bold')
plt.show()


#Printing the data of customer service calls
telecom_df['Customer service calls'].value_counts()

#Calculating the Customer service calls vs Churn percentage
Customer_service_calls_data = pd.crosstab(telecom_df['Customer service calls'],telecom_df["Churn"])
Customer_service_calls_data['Percentage_Churn'] = Customer_service_calls_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Customer_service_calls_data)

#Analysing using countplot
sns.countplot(x='Customer service calls',hue="Churn",data = telecom_df)


#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total day calls'].mean())

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total day minutes'].mean())

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total day charge'].mean())

#show the relation using scatter plot
sns.scatterplot(x="Total day minutes", y="Total day charge", hue="Churn",
data=telecom_df,palette='hls')

#show the relation using box plot plot
sns.boxplot(x="Total day minutes", y="Total day charge", hue="Churn",
data=telecom_df,palette='hls')

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total eve calls'].mean())

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total eve minutes'].mean())

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total eve charge'].mean())

#show the relation using scatter plot
sns.scatterplot(x="Total eve minutes", y="Total eve charge", hue="Churn",
data=telecom_df,palette='hls')

#show the relation using box plot plot
sns.boxplot(x="Total eve minutes", y="Total eve charge", hue="Churn",
data=telecom_df,palette='hls')

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total night calls'].mean())

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total night charge'].mean())

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total night minutes'].mean())

#show the relation using scatter plot
sns.scatterplot(x="Total night minutes", y="Total night charge", hue="Churn",
data=telecom_df,palette='hls')

#show the relation using box plot
sns.boxplot(x="Total night minutes", y="Total night charge", hue="Churn",
data=telecom_df,palette='hls')

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total intl minutes'].mean())

#Print the mean value of churned and not churned customer
print(telecom_df.groupby(["Churn"])['Total intl minutes'].mean())

#show the relation using scatter plot
sns.scatterplot(x="Total intl minutes", y="Total intl charge", hue="Churn",
data=telecom_df,palette='hls')

#show the relation using box plot
sns.boxplot(x="Total intl minutes", y="Total intl charge", hue="Churn",
data=telecom_df,palette='hls')

#Deriving a relation between overall call charge and overall call minutes
day_charge_perm = telecom_df['Total day charge'].mean()/telecom_df['Total day minutes'].mean()
eve_charge_perm = telecom_df['Total eve charge'].mean()/telecom_df['Total eve minutes'].mean()
night_charge_perm = telecom_df['Total night charge'].mean()/telecom_df['Total night minutes'].mean()
int_charge_perm= telecom_df['Total intl charge'].mean()/telecom_df['Total intl minutes'].mean()

print([day_charge_perm,eve_charge_perm,night_charge_perm,int_charge_perm])

sns.barplot(x=['Day','Evening','Night','International'],y=[day_charge_perm,eve_charge_per
m,night_charge_perm,int_charge_perm])

#Printing boxplot for each numerical column present in the data set
df1=telecom_df.select_dtypes(exclude=['object','bool'])
for column in df1:
    plt.figure(figsize=(17,1))
    sns.boxplot(data=df1, x=column)
    plt.show()

#Printing displot for each numerical column present in the data set
df1=telecom_df.select_dtypes(exclude=['object','bool'])
for column in df1:
    plt.figure(figsize=(17,1))
    sns.displot(data=df1, x=column)
    plt.show()

#Printing strip plot for each numerical column present in the data set
df1=telecom_df.select_dtypes(exclude=['object','bool'])
for column in df1:
    plt.figure(figsize=(17,1))
    sns.stripplot(data=df1, x=column)
    plt.show()

# Plot a boxplot for churn column by each numerical feature present in the data set
df2= telecom_df.describe().columns
for col in df2:
    fig=plt.figure(figsize=(17,3))
    ax=fig.gca()
    feature=telecom_df[col]
    label=telecom_df['Churn']
    correlation= feature.corr(label)
    plt.scatter(x=feature,y=label)
    plt.xlabel(col)
    plt.ylabel('Churn')
    plt.show()

#Plot the box plot for churn vs all numerical column
for col in df2:
    fig=plt.figure(figsize=(17,10))
    ax=fig.gca()
#feature=telecom_df[col]


#label=telecom_df['Churn']
telecom_df.boxplot(column = 'Churn', by = col, ax = ax)
plt.xlabel(col)
plt.ylabel('Churn')
plt.show()


# visualization using correlation plot
plt.figure(figsize=(19,8))
telecom_df.corr()['Churn'].sort_values(ascending = False).plot(kind='bar',color =
['red','blue','yellow','indigo','orange','brown','pink'])

## plot the Correlation matrix
plt.figure(figsize=(17,8))
correlation=telecom_df.corr()
sns.heatmap(abs(correlation), annot=True, cmap='coolwarm')

#create a correlation heatmap
#Assigning true=1 and False=0 to churn variable
telecom_df['Churn'] = telecom_df['Churn'].replace({bool(True):1,bool(False):0})
plt.figure(figsize=(17,9))
sns.heatmap(telecom_df.corr(), cmap="Paired",annot=False)
plt.title("Correlation Heatmap", fontsize=20)

#plot the pair plot for all coloumn
sns.pairplot(telecom_df, height=3)