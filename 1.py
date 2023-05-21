from os import walk
import os
import pandas as pd


csv_filenames = os.listdir('C:\\Users\\Гамид\\Desktop\\train_dataset_Росатом\\Anonimized')
participants = pd.read_csv('C:\\Users\\Гамид\\Desktop\\train_dataset_Росатом\\Участники anonimized.csv')

def filter_str(s):
	return ''.join(filter(lambda x: x.isalpha() or x == ' ', s))

#os.makedirs('C:\\Users\\Гамид\\Desktop\\train_dataset_Росатом\\augmented_csv')
participants['Список компетенций'] = participants['Список компетенций'].str.replace(';', '')

komp = participants['Список компетенций'].values.tolist()

for el in csv_filenames:
	data = pd.read_csv('C:\\Users\\Гамид\\Desktop\\train_dataset_Росатом\\Anonimized\\' + el)
	if 'Название команды' in data.columns:
		row2delete = []
		data.insert(1, "ФИО", ['temp']*len(data.index), True)
		for index, row in data.iterrows():
			names = row['ФИО участников'].split('; ')
			for name in names:
				temp = row
				temp['ФИО'] = name
				data.loc[len(data.index)] = temp
			row2delete.append(index)
		for index in row2delete:
			data = data.drop(index)
		data = data.drop(columns=['Название команды', 'ФИО участников'])
		#print(data.columns)
	data.to_csv('C:\\Users\\Гамид\\Desktop\\train_dataset_Росатом\\augmented_csv\\' + el, index=False)

for el in csv_filenames:
	data = pd.read_csv('C:\\Users\\Гамид\\Desktop\\train_dataset_Росатом\\augmented_csv\\' + el)
	data['Пол'] = ['Не указано']*len(data.index)
	data['Должность'] = ['Не указано']*len(data.index)
	data['Начало трудового стажа'] = ['Не указано']*len(data.index)
	data['Место работы'] = ['Не указано']*len(data.index)
	data['Профессия'] = ['Не указано']*len(data.index)
	data['Начало трудовой деятельности в РОСАТОМ'] = ['Не указано']*len(data.index)
	data['Образование'] = ['Не указано']*len(data.index)
	data['Место образования'] = ['Не указано']*len(data.index)
	data['Год оканчания'] = ['Не указано']*len(data.index)
	data['Специальность'] = ['Не указано']*len(data.index)

	for index, row in data.iterrows():
		if row['ФИО'] in participants['ФИО'].values.tolist():
			row['Пол'] = participants.loc[participants['ФИО'] == row['ФИО']]['Пол'].item()
			row['Должность'] = participants.loc[participants['ФИО'] == row['ФИО']]['Должность'].item()
			row['Начало трудового стажа'] = participants.loc[participants['ФИО'] == row['ФИО']]['Начало трудового стажа'].item()
			row['Место работы'] = participants.loc[participants['ФИО'] == row['ФИО']]['Место работы'].item()
			row['Профессия'] = participants.loc[participants['ФИО'] == row['ФИО']]['Профессия'].item()
			row['Начало трудовой деятельности в РОСАТОМ'] = participants.loc[participants['ФИО'] == row['ФИО']]['Начало трудовой деятельности в РОСАТОМ'].item()
			row['Образование'] = participants.loc[participants['ФИО'] == row['ФИО']]['Образование'].item()
			row['Место образования'] = participants.loc[participants['ФИО'] == row['ФИО']]['Место образования'].item()
			row['Год оканчания'] = participants.loc[participants['ФИО'] == row['ФИО']]['Год оканчания'].item()
			row['Специальность'] = participants.loc[participants['ФИО'] == row['ФИО']]['Специальность'].item()
			data.loc[index] = row

	data.to_csv('C:\\Users\\Гамид\\Desktop\\train_dataset_Росатом\\augmented_csv\\' + el, index=False)

#Вывод из эксплуатации объектов использования атомной энергии.csv