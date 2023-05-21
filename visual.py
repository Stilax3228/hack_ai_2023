import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
from datetime import datetime
import os
from sklearn.cluster import KMeans, AffinityPropagation
import matplotlib.pyplot as plt
import statistics
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4
import io
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import defaultPageSize
import plotly.graph_objs as go


# Формируем глобальные списки для сохранения рисунков при генерации отчетных материалов
report_imgs_page1 = []
report_imgs_page2 = []
report_imgs_page3 = []
# Служебные функции
###############################################

# Пробразование объекта плот в пдф-объект
def fig2img_pdf(plt, arr):
	
	buf = io.BytesIO()
	plt.savefig(buf)
	buf.seek(0)
	img = Image.open(buf)
	newsize = (400, 200)
	img = img.resize(newsize)
	side_im_data = io.BytesIO()
	img.save(side_im_data, format='png')

	side_im_data.seek(0)
	side_out = ImageReader(side_im_data)
	arr.append(side_out)
	
	return side_out

# Формирование отчетных материалов
def report(page):
	canvas = Canvas("C:\\Users\\Гамид\\Desktop\\report\\hello.pdf", pagesize=(500, 700))
	PAGE_WIDTH  = defaultPageSize[0]
	PAGE_HEIGHT = defaultPageSize[1]
	if page == 1:
		for el in report_imgs_page1:
			canvas.drawImage(el, PAGE_WIDTH//2 - 200, PAGE_HEIGHT - 400)
	elif page == 2:
		for el in report_imgs_page2:
			canvas.drawImage(el, PAGE_WIDTH//2 - 200, PAGE_HEIGHT - 400)
	elif page == 3:
		for el in report_imgs_page3:
			canvas.drawImage(el, PAGE_WIDTH//2 - 200, PAGE_HEIGHT - 400)
	
	canvas.save()

def get_data_list(path) -> list:
	res = [x[:-4] for x in os.listdir(path)]
	res.sort()
	return res

# получение списка колонок float в таблице
def get_float_columns_names(df) -> list:
	columns = list(df.columns)
	res =[]
	for col in columns:
		if str(df[col].dtype) == 'float64':
			res.append(col)
	return res

# получение списка колонок category в таблице
def get_category_columns_names(df) -> list:
	columns = list(df.columns)
	res =[]
	for col in columns:
		if str(df[col].dtype) != 'float64':
			res.append(col)
	return res

# предобработка столбца перевод во float того, что можно...
def column_preprocess(df, col_name):
	# попытка перевeсти строки в число
	# если не получается, то оставляем строку как есть
	def try_convert_to_float(arg):
		s = arg
		if type(s) == str:
			s = s.replace(',', '.')
		try:
			return float(s)
		except:
			return str(arg)
	# если аргумент не число, то возвращяем 0.0
	# иначе оставляем как есть
	def convert_str_to_0(arg):
		if type(arg) != float:
			return 0.0
		else:
			return arg
	# переводим данные в список и работаем с ним
	rab = list(df[col_name])
	n = 0
	for i in range(len(rab)):
		rab[i] = try_convert_to_float(rab[i])
		if type(rab[i]) == float:
			n += 1
	if n > 0:
		for i in range(len(rab)):
			rab[i] = convert_str_to_0(rab[i])
	# переводим обработанный список вновь в столбец
	df[col_name] = rab 

# предобработка таблицы:
# удаление столбцов, корректировка данных, пропусков, дат...
def dataframe_preprocess(df):
	# столбец в возраст
	def date_to_age(arg):
		try:
			d = datetime.now() - datetime.strptime(arg, '%Y-%m-%d')
		except:
			try:
				d = datetime.now() - datetime.strptime(arg, '%d.%m.%Y')
			except:
				return 0.0
		return round(d.days/365.2425, 1)
	# удаление неинформативных столбцов
	del df['№'], df['Компетенция'], df['Баллы, ед.'], df['Баллы, %']
	# заполнение пропусков
	df.fillna('Не указано', inplace=True)
	# предобработка всех столбцов
	for col in list(df.columns):
		column_preprocess(df, col)
	# замена столбца о трудовом стаже
	df['Трудовой стаж, лет'] = df['Начало трудового стажа'].apply(date_to_age)
	del df['Начало трудового стажа']
	# замена столбца о стаже в Росатоме
	df['Трудовой стаж в РОСАТОМ, лет'] = df['Начало трудовой деятельности в РОСАТОМ'].apply(date_to_age)
	del df['Начало трудовой деятельности в РОСАТОМ']
	# удаление столбцов, у которых все пропуски
	del_col_list = []
	for col in list(df.columns): 
		rab = list(dict(df[col].value_counts()).keys())
		if (len(rab) == 1) and rab[0] == 'Не указано':
			del_col_list.append(col)
	df.drop(del_col_list, axis=1, inplace=True)

	return df

# генерация данных для кластеризации
# df - таблица pandas
# col_list - список колонок для кластеризации
def gen_data_for_clust(df, col_list):
	try:
		res = df[col_list].to_numpy().astype('float32') 
	except:
		print('Ошибка извлечения данных длля кластеризации')
		res = None
	return res

# кластеризация K-средних в таблице
# cl_count - количество кластеров
# data - numpy двумерный массив данных
# возвращает список индексов кластеризации
def k_means_clustering(cl_count, data) -> list:
	try:
		kmean = KMeans(cl_count)
		kmean.fit(data)
		return list(kmean.labels_)
	except Exception as e:
		print('Ошибка кластеризации')
		print(e)
		return []

# получение списка ФИО для кластера
# df - таблица pandas
# cluster - номер кластера
def get_cluster_fio_list(df, cluster) -> list:
	return list(df[df.clust == cluster]['ФИО'])

# получение словаря распределения категориальных значений для кластера
# df - таблица pandas
# cluster - номер кластера
# col_name - название столбца
def get_cluster_category_values(df, cluster, col_name) -> dict:
	return dict(df[df.clust == cluster][col_name].value_counts())

# получение суммы значений столбца для кластера
# df - таблица pandas
# cluster - номер кластера
# col_name - название столбца
def get_cluster_sum(df, cluster, col_name) -> dict:
	try:
		return round(sum(df[df.clust == cluster][col_name]), 3)
	except:
		print('Невозможно получить сумму для столбца:', col_name)
		return None

# получение среднего значения столбца для кластера
# df - таблица pandas
# cluster - номер кластера
# col_name - название столбца
def get_cluster_mean(df, cluster, col_name) -> dict:
	try:
		values = df[df.clust == cluster][col_name]
		return round(sum(values)/len(values), 3) 
	except:
		print('Невозможно получить среднее для столбца:', col_name)
		return None 

# функция визуализации облаков - кластеров
# name - имя датасета
# x_col - название столбца для X координаты
# y_col - название столбца для Y координаты
# cl_count - количество кластеров
def clusters_show_2d(name, x_col, y_col, cl_count):

	df = get_clust_csv(name)

	markers = ['ro', 'go', 'bo', 'yo', 'co', 'mo', 'ko']    
	plt.figure(figsize=(7, 7))
	for i in range(cl_count):        
		x = np.array(df[(df['clust'] == i)][x_col])
		y = np.array(df[(df['clust'] == i)][y_col])
		for j in range(len(x)): 
			plt.plot(x[j], y[j], markers[i])
	plt.xlabel(x_col)
	plt.ylabel(y_col)
	fig2img_pdf(plt, report_imgs_page2)

	markers = ['ro', 'go', 'bo', 'yo', 'co', 'mo', 'ko']
	dfs = []

	for i in range(cl_count):    

		x = np.array(df[(df['clust'] == i)][x_col])
		y = np.array(df[(df['clust'] == i)][y_col])

		dfs.append(pd.DataFrame({'name': 'Кластер ' + str(i + 1), 'color': [markers[i] for j in range(len(x))], 'x': x, 'y': y}))

	_df = pd.concat(dfs)
	fig = px.scatter(_df, x='x', y='y', color='name', labels={'x': x_col, 'y': y_col})

	return fig

###############################################
# Производим кластеризацию датафрейма по имени
def get_clust_csv(name):
	df = pd.read_csv('train_dataset_Росатом/augmented_csv/' + name)
	data = dataframe_preprocess(df)
	col_for_clust = ['Баллы по ключевым навыкам', 'Результат']
	# генерация массива для кластеризации
	data_for_clust = gen_data_for_clust(data, col_for_clust)
	cl_count = 3
	labels = k_means_clustering(cl_count, data_for_clust)
	data['clust'] = labels

	return data

# Чтение кластеризованных данных
participants_2022 = pd.read_csv('train_dataset_Росатом/Участники anonimized.csv')
participants_2021 = pd.read_csv('train_dataset_Росатом/Участники anonimized_2021.csv')
participants_2023 = pd.read_csv('train_dataset_Росатом/Участники anonimized_2023.csv')

csv_filenames = os.listdir('train_dataset_Росатом/augmented_csv')


# Инициализация веб-сервера
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True


# Универсальная функция визуализации кластеров по пользовательским параметрам
'''def clusters_show_2d(x_col, y_col, clusters_count=4, scale=0.9):
	markers = ['ro', 'go', 'bo', 'yo', 'co', 'mo', 'ko']
	dfs = []

	for i in range(clusters_count):    

		x = np.array(regions_4clust_df[(regions_4clust_df['k_means_clust'] == i)][x_col])
		y = np.array(regions_4clust_df[(regions_4clust_df['k_means_clust'] == i)][y_col])
		#print(x, y)

		#_, x, _, y = train_test_split(x, y, test_size=scale, random_state=1)
		#print(i)

		dfs.append(pd.DataFrame({'name': 'Кластер ' + str(i + 1), 'color': [markers[i] for j in range(len(x))], 'x': x, 'y': y}))

	_df = pd.concat(dfs)
	fig = px.scatter(_df, x='x', y='y', color='name', labels={'x': x_col, 'y': y_col})

	return fig'''


# метаданные
SIDESTYLE = {
	'position': 'fixed',
	'top': 0,
	'left': 0,
	'bottom': 0,
	'width': '16rem',
	'padding': '2rem 1rem',
	'background-color': '#222222',
}


CONTSTYLE = {
	'margin-left': '18rem',
	'margin-right': '2rem',
	'padding': '2rem 1rem',
}


# Фронтэнд. Структура
app.layout = html.Div([
	dcc.Location(id='url'),
	html.Div(
		[
			html.H2('AI4H', className='display-3', style={'color': 'white'}),
			html.Hr(style={'color': 'white'}),
			dbc.Nav(
				[
					dbc.NavLink('Информация о чемпионате', href='/page1', active='exact'),
					dbc.NavLink('Информация о компетенциях', href='/page2', active='exact'),
					dbc.NavLink('Информация об участниках', href='/page3', active='exact'),
				],
				vertical=True,pills=True),
		],
		style=SIDESTYLE,
	),
	html.Div(id='page-content', children=[], style=CONTSTYLE)
])


@app.callback(
	Output('page-content', 'children'),
	[Input('url', 'pathname')])

# Верстка первой страницы
def pagecontent(pathname):
	if pathname == '/page1':

		return [

			html.Div(
				children=[
					html.H1(children='Общая характеристика чемпионата', className='header-title'),
				], className='header'),
			html.P("Год:"),
			dcc.Dropdown(id='page1_year',
				options=['2021', '2022', '2023'],
				value='2021', className='dropdown', style= {'margin-bottom':'16px'}
			),
			dcc.Dropdown(id='page1_info',
				options=[{'label': 'Место работы', 'value': 'prof'},
						{'label': 'Возраст', 'value': 'age'},
						{'label': 'Пол', 'value': 'sex'},
						{'label': 'Образование', 'value': 'graduate'},
						{'label': 'Участие в компетенциях', 'value': 'comp_amount'},
						{'label': 'Общее число участников чемпионата', 'value': 'part_amount'}],
				value='prof', className='dropdown', style= {'margin-bottom':'16px'}
			), html.P("Тип диаграммы:"),
			dcc.Dropdown(id='page1_type_graph',
				options=[{'label': 'Круговая диаграмма', 'value': 'pie'},
						{'label': 'Столбчатая диаграмма', 'value': 'vertical'}],
				value='pie', style= {'margin-bottom':'16px'}
			), dcc.Graph(id='output_graph_page1', style= {'margin-bottom':'64px'}),
			
			]


# Верстка второй страницы	
	elif pathname == '/page2':
		return [
			html.Div(
				children=[
					html.H1(children='Частная характеристика компетенции', className='header-title'),

					html.P(children='')
				], className='header'),
			dcc.Dropdown(id='page2_comp_choice',
					options=csv_filenames,
					value='Инженер-конструктор.csv', className='dropdown', style= {'margin-bottom':'16px'}
			),
			html.Div(id='output2_custom_div'),
			dcc.Dropdown(id='page2_info',
				options=[{'label': 'Место работы', 'value': 'prof'},
						{'label': 'Пол', 'value': 'sex'},
						{'label': 'Образование', 'value': 'graduate'},
						{'label': 'Облачное представление кластеров', 'value': 'clouds'}],
				value='clouds', className='dropdown', style= {'margin-bottom':'16px'}
			),
			dcc.Dropdown(id='page2_type_graph',
				options=[{'label': 'Круговая диаграмма', 'value': 'pie'},
						{'label': 'Столбчатая диаграмма', 'value': 'vertical'}],
				value='pie', style= {'margin-bottom':'16px'}
			),
			dcc.Graph(id='output_graph_page2'),
			html.Div(
			    [
			        dbc.Button(
			            "Сформировать отчет", id="report_but_page2", n_clicks=0
			        ),
			        html.Span(id="save_label_page2", style={"verticalAlign": "middle"}),
			    ])
			#dcc.Dropdown(id='page2_col_choice'),
			#dcc.Graph(id='output2_graph_comp'),
				]


# Верстка третьей страницы
	elif pathname == '/page3':
		drop_opt = participants_2021['ФИО'].tolist()
		return [
				html.Div([
					dcc.Dropdown(id='page3_year',
						options=['2021', '2022', '2023'],
						value='2022', className='dropdown', style= {'margin-bottom':'16px'}
					),
					dcc.Dropdown(id='page3_fio',
						options=drop_opt,
						value='ФИО_5', className='dropdown', style= {'margin-bottom':'16px'}
					),
					html.Div(id='output3_custom_div2'),
					dcc.Dropdown(id='page3_info',
						options=[{'label': 'Навыки', 'value': 'skills'},
								 {'label': 'Динамика результатов', 'value': 'dinamic'}],
						value='dinamic', className='dropdown', style= {'margin-bottom':'16px'}
					),
					dcc.Graph(id='output_graph_page3'),
					])
		]


# Формируем выходной график на 3 странице
@app.callback(
	Output(component_id='output_graph_page3', component_property='figure'),
	[Input(component_id='page3_fio', component_property='value'), 
	Input("page3_year", "value"),
	Input("page3_info", "value")]
)

def update_output(fio, year, info):
	if info == 'skills':
		comp = participants_2021[participants_2021['ФИО'] == fio]['Список компетенций'].values[0].replace(';', '.csv').strip()
		data = pd.read_csv('train_dataset_Росатом/augmented_csv/' + comp)
		data = data[data['ФИО'] == fio][get_float_columns_names(data)[0]].values.tolist()[0]
	elif info == 'dinamic':
		res1 = participants_2021[participants_2021['ФИО'] == fio]['Результат'].values[0]
		res2 = participants_2022[participants_2022['ФИО'] == fio]['Результат'].values[0]
		res3 = participants_2023[participants_2023['ФИО'] == fio]['Результат'].values[0]
		figure = go.Figure(data=[go.Scatter(x=['2021', '2022', '2023'], y=[res1, res2, res3])])

		return figure

# Формируем динамический Див на 3 странице
@app.callback(
	Output(component_id='output3_custom_div2', component_property='children'),
	[Input(component_id='page3_fio', component_property='value')]
)

def update_output(fio):

	data = participants_2022

	data = data[data['ФИО'] == fio][['ФИО','Пол', 'Должность', 'Образование', 'Результат', 'Место работы', 'Список компетенций']].values.tolist()[0]
	temp = []
	name = data[0]
	sex = 'Мужской' if not(data[1]) else 'Женский'
	prof = data[2] if data[2] else 'Не указано'
	graduate = data[3] if data[3] else 'Не указано'
	result = data[4]
	prof_place = data[5]
	comp = data[6]
	temp.append(html.P(children=("ФИО: " + name)))
	temp.append(html.P(children="Пол: " + str(sex)))
	temp.append(html.P(children="Список компетенций: " + str(comp)))
	temp.append(html.P(children="Специальность: " + str(prof)))
	temp.append(html.P(children="Место работы: " + str(prof_place)))
	temp.append(html.P(children="Образование: " + str(graduate)))
	temp.append(html.P(children="Результат: " + str(result)))


	return html.Div(children=temp)


# Ивент кнопки сохранить отчет
@app.callback(
    Output("save_label_page2", "children"), [Input("report_but_page2", "n_clicks")]
)

def on_button_click(n):
    if n:
    	report(2)
    	return "Отчет успешно сформирован"
    else:
    	return ''


# Формируем Див по кластеру (список ФИО в кластере)
def generate_div_tag_by_clust(data, clust):
	clust_inf_members = []
	data = ', '.join(get_cluster_fio_list(data, clust))

	'''for el in data:
		data = data[data['clust'] == clust][['ФИО','Пол', 'Должность', 'Образование', 'Результат']].values.tolist()
		temp = []
		name = el[0]
		sex = el[1]
		prof = el[2]
		graduate = el[3]
		result = el[4]
		temp.append(html.P(children=("ФИО: " + name)))
		temp.append(html.P(children="Пол: " + str(sex)))
		temp.append(html.P(children="Специальность: " + prof))
		temp.append(html.P(children="Образование: " + graduate))
		temp.append(html.P(children="Результат: " + str(result)))

		clust_inf_members.append(html.Div(children=temp))'''

	temp = html.P(children=('Кластер ' + str(clust + 1) + ': ' + data))

	return temp


# Генерация динамических Див'ов на 2 странице
@app.callback(
	Output(component_id='output2_custom_div', component_property='children'),
	[Input(component_id='page2_comp_choice', component_property='value')]
)

def update_output(value):

	data = get_clust_csv(value)

	m_score = statistics.median(sorted(data['Результат'].tolist()))

	return html.Div(
				children=[
			html.P(children=('Всего участников по данной компетенции - ' + str(len(data.index)))),
			html.P(children=('Медиана по результатам участников - ' + str(m_score))),
			html.Div(children=[generate_div_tag_by_clust(data, i) for i in range(3)], className='header'),
			])


# !!!Доделать!!!

'''@app.callback(
	Output(component_id='output2_graph_comp', component_property='children'),
	[Input(component_id='page2_comp_choice', component_property='value')]
)

def update_output(value):
	data = pd.read_csv('train_dataset_Росатом/augmented_csv/' + value)
	cols = get_float_columns_names(data)
	options = {}
	for el in cols:
		options[el] = el

		custom_drop = dcc.Dropdown(id='page2_col_choice',
			options=options,
			multi=True, value=[cols[0], cols[1]], className='dropdown', style={'margin-bottom':'32px'}
		)

	return custom_drop
'''


# Выходной график на 1 странице
@app.callback(
	Output(component_id='output_graph_page1', component_property='figure'),
	[Input(component_id='page1_year', component_property='value'), 
	Input("page1_type_graph", "value"),
	Input("page1_info", "value")]
)

def update_output(year, type_gpaph, info):
	if year == '2021':
		data = participants_2021
	elif year == '2022':
		data = participants_2022
	elif year == '2023':
		data = participants_2023

	if info == 'prof':
		data = data['Место работы'].tolist()
		x = []
		y = []
		data = sorted(Counter(data).items(), key=lambda x: x[1], reverse=True)[:10]
		for name, count in data:
			x.append(name)
			y.append(count)
		
		if type_gpaph == 'vertical':
			figure = {
					'data': [
						{'x': x, 'y': y, 'type': 'bar', 'name': 'Кластер №1'},],
					'layout': {
						'title': ''
					}
				}
		elif type_gpaph == 'pie':
			figure = px.pie(data, values=y, names=x, hole=.3)

		return figure

	elif info == 'part_amount':
		data_male = []
		data_female = []

		data_male.append(len(participants_2021.loc[participants_2021['Пол'] == 0].index))
		data_male.append(len(participants_2022.loc[participants_2022['Пол'] == 0].index))
		data_male.append(len(participants_2023.loc[participants_2023['Пол'] == 0].index))

		data_female.append(len(participants_2021.loc[participants_2021['Пол'] == 1].index))
		data_female.append(len(participants_2022.loc[participants_2022['Пол'] == 1].index))
		data_female.append(len(participants_2023.loc[participants_2023['Пол'] == 1].index))
		
		return {
					'data': [
						{'x': ['2021', '2022', '2023'], 'y': data_male, 'type': 'bar', 'name': 'Мужской пол'},
						{'x': ['2021', '2022', '2023'], 'y': data_female, 'type': 'bar', 'name': 'Женский пол'}],
					'layout': {
						'title': ''
					}
				}

	elif info == 'age':
		data = data['Дата рождения'].tolist()

		for i in range(len(data)):
			try:
				data[i] = int((datetime.now() - datetime.strptime(data[i], '%Y-%m-%d')).days / 365)
			except:
				data[i] = 'Не указано'
		x = []
		y = []
		data = sorted(Counter(data).items(), key=lambda x: x[1], reverse=True)
		for name, count in data:
			x.append(name)
			y.append(count)

		res_y_30 = []
		res_y_35 = []
		res_y_45 = []
		res_y_50 = []
		res_y_NN = []
		res_y = []

		for el in data:
			if el[0] == 'Не указано':
				res_y_NN.append(el[0])
			elif el[0] <= 30:
				res_y_30.append(el[0])
			elif 31 <= el[0] <= 35:
				res_y_35.append(el[0])
			elif 36 <= el[0] <= 45:
				res_y_45.append(el[0])
			elif el[0] >= 46:
				res_y_50.append(el[0])

		res_y.append(len(res_y_30))
		res_y.append(len(res_y_35))
		res_y.append(len(res_y_45))
		res_y.append(len(res_y_50))
		res_y.append(len(res_y_NN))
		
		if type_gpaph == 'vertical':
			figure = {
					'data': [
						{'x': ['до 30 лет', '31-35 лет', '36-45 лет', 'от 46 лет', 'Не указано'], 'y': res_y, 'type': 'bar', 'name': 'Кластер №1'},],
					'layout': {
						'title': ''
					}
				}
		elif type_gpaph == 'pie':
			figure = px.pie(data, values=res_y, names=['до 30 лет', '31-35 лет', '36-45 лет', 'от 46 лет', 'Не указано'], hole=.3)

		return figure


	elif info == 'sex':
		data = data['Пол'].tolist()
		x = []
		y = []
		data = sorted(Counter(data).items(), key=lambda x: x[1], reverse=True)
		for name, count in data:
			x.append(name)
			y.append(count)
		
		if type_gpaph == 'vertical':
			figure = {
					'data': [
						{'x': ['Мужской', 'Женский'], 'y': y, 'type': 'bar', 'name': 'Кластер №1'},],
					'layout': {
						'title': ''
					}
				}
		elif type_gpaph == 'pie':
			figure = px.pie(data, values=y, names=['Мужской', 'Женский'], hole=.3)

		return figure

	elif info == 'comp_amount':
		data = data['Список компетенций'].tolist()
		x = []
		y = []
		data = sorted(Counter(data).items(), key=lambda x: x[1], reverse=True)[:15]
		for name, count in data:
			x.append(name)
			y.append(count)
		
		if type_gpaph == 'vertical':
			figure = {
					'data': [
						{'x': x, 'y': y, 'type': 'bar', 'name': 'Кластер №1'},],
					'layout': {
						'title': ''
					}
				}
		elif type_gpaph == 'pie':
			figure = px.pie(data, values=y, names=x, hole=.3)

		return figure


	elif info == 'graduate':
		data = data['Образование'].tolist()
		x = []
		y = []
		res_y_SPO = []
		res_y_VUZ = []
		res_y = []
		data = sorted(Counter(data).items(), key=lambda x: x[1], reverse=True)
		for name, count in data:
			try:
				if 'высш' in name.lower():
					res_y_VUZ.append(name)
				elif 'сред' in name.lower() or 'спо' in name.lower() or 'проф' in name.lower():
					res_y_SPO.append(name)
			except:
				pass
		res_y.append(len(res_y_VUZ))
		res_y.append(len(res_y_SPO))
		if type_gpaph == 'vertical':
			figure = {
					'data': [
						{'x': ['Высшее образование', 'СПО'], 'y': res_y, 'type': 'bar', 'name': 'Кластер №1'},],
					'layout': {
						'title': ''
					}
				}
		elif type_gpaph == 'pie':
			figure = px.pie(res_y, values=res_y, names=['Высшее образование', 'СПО'], hole=.3)

		return figure


# Выходной график на 2 странице
@app.callback(
	Output(component_id='output_graph_page2', component_property='figure'),
	[Input(component_id='page2_comp_choice', component_property='value'), 
	Input("page2_type_graph", "value"),
	Input("page2_info", "value")]
)

def update_output(name, type_gpaph, info):
	data = get_clust_csv(name)
	#data = data[data['clust'] == clust][['ФИО','Пол', 'Должность', 'Образование', 'Результат']].values.tolist() 

	if info == 'prof':
		data = data['Место работы'].tolist()
		x = []
		y = []
		data = sorted(Counter(data).items(), key=lambda x: x[1], reverse=True)[:10]
		for name, count in data:
			x.append(name)
			y.append(count)
		
		if type_gpaph == 'vertical':
			figure = {
					'data': [
						{'x': x, 'y': y, 'type': 'bar', 'name': 'Кластер №1'},],
					'layout': {
						'title': ''
					}
				}
		elif type_gpaph == 'pie':
			figure = px.pie(data, values=y, names=x, hole=.3)

		return figure

	elif info == 'sex':
		data = data['Пол'].tolist()
		x = []
		y = []

		data = sorted(Counter(data).items(), key=lambda x: x[1], reverse=True)[:10]
		for name, count in data:
			x.append(name)
			y.append(count)
		
		if type_gpaph == 'vertical':
			figure = {
					'data': [
						{'x': ['Мужской', 'Женский'], 'y': y, 'type': 'bar', 'name': 'Кластер №1'},],
					'layout': {
						'title': ''
					}
				}
		elif type_gpaph == 'pie':
			figure = px.pie(data, values=y, names=['Мужской', 'Женский'], hole=.3)

		return figure

	elif info == 'graduate':
		data = data['Образование'].tolist()
		x = []
		y = []
		res_y_SPO = []
		res_y_VUZ = []
		res_y = []
		data = sorted(Counter(data).items(), key=lambda x: x[1], reverse=True)
		for name, count in data:
			try:
				if 'высш' in name.lower():
					res_y_VUZ.append(name)
				elif 'сред' in name.lower() or 'спо' in name.lower() or 'проф' in name.lower():
					res_y_SPO.append(name)
			except:
				pass
		res_y.append(len(res_y_VUZ))
		res_y.append(len(res_y_SPO))
		if type_gpaph == 'vertical':
			figure = {
					'data': [
						{'x': ['Высшее образование', 'СПО'], 'y': res_y, 'type': 'bar', 'name': 'Кластер №1'},],
					'layout': {
						'title': ''
					}
				}
		elif type_gpaph == 'pie':
			figure = px.pie(res_y, values=res_y, names=['Высшее образование', 'СПО'], hole=.3)

		return figure


	elif info == 'clouds':
	
		return clusters_show_2d(name, 'Результат','Баллы по ключевым навыкам', 3)


# Запуск веб-сервера на 3000 порту в режиме дебага (при сохранении кода не требуется перезапуск программы)
app.run_server(debug=True, port=3000)




































