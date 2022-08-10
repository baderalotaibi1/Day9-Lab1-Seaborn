import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/bader/OneDrive/gitlesson/Day9-Lab1-Seaborn/titanic.csv")
#Q1: Print first and last rows from the dataset
print(df.head(),"\n")
print(df.tail(),"\n")
#Q2: Select only survived people
print(df[df["survived"]>0])
#Q3: Select sex, fare, survived columns
print(df[["sex","fare","survived"]])
#Q4: Add a new_column to a DataFrame that combines class and embark_town
df["new_column"] = df['class'] +"-"+ df["embark_town"]
print(df)
#Q5: Remove new_column from the DataFrame
df=df.drop("new_column",axis=1)
print(df)
#Q6: Filter DataFrame for rows of survived Males only
sur=df["survived"]>0
sex=df["sex"]=="male"
filter_male=df[sex & sur]
print(filter_male)

#Q7: The total number of males who survived
print(filter_male["survived"].value_counts())

#Q8: How many values in each class?
print(df['class'].value_counts())

#Q9: Draw barplot represents survived people based on sex
sns.barplot(x="sex",y="survived",data=df)
plt.show()
#Q10: Draw catplot represents survived people based on embarked
sns.catplot(data=df,kind="count",x='survived',col='embarked')
plt.show()
#Q11: Draw boxplot represents distribution of male and female based on age and pclass
sns.boxplot(data=df,x='sex',y='age',hue='pclass')
plt.show()

#Q12: Draw heatmap represents correlations between sibsp, parch, age, fare, and survived columns
df_corr=df[['sibsp','parch','age','fare','survived']].corr()
sns.heatmap(data=df_corr,annot=True)
plt.show()

#Q13: Draw factorplot represents the relation between sibsp and survived columns
sns.factorplot(x="sibsp",y="survived" ,data=df)
plt.show()


