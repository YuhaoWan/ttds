import pandas

num = {
	1: ['10001', '123', 'Mike', '2017-5-6'],
	2: ['10002', '123', 'Sam', '2018-5-6'],
	3: ['10003', '123', 'hong', '2019-5-6'],
	4: ['10004', '123', 'xiao', '2019-5-6']
}
data_pd = pandas.DataFrame.from_dict(num,orient='index',columns=['Songs','Albums','Artists','date'])
data_dict = data_pd.to_dict(orient = 'records')

print(len(data_dict))