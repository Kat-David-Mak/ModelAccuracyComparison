import pandas as pd

#retrieving an excel file called “zoo_dataset.xlsx” from the end assessment folder
read_file = pd.read_excel('end assessment/zoo_dataset.xlsx',sheet_name='Sheet1')
#converting the file into a csv file
read_file.to_csv('end assessment/zoo_dataset.csv', index=None, header=True)
zoo_data = pd.read_csv('end assessment/zoo_dataset.csv')

#displaying the first five rows 
print(zoo_data.head()) 