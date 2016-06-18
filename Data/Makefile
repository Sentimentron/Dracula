all: testing_data.txt training_data.txt

testing_data.txt: testdata.manual.2009.06.14.csv
	python3 process_training_data.py testdata.manual.2009.06.14.csv testing_data.txt

training_data.txt: training.1600000.processed.noemoticon.csv
	python3 process_training_data.py training.1600000.processed.noemoticon.csv training_data.txt

training.1600000.processed.noemoticon.csv: trainingandtestdata.zip
	unzip -p trainingandtestdata.zip training.1600000.processed.noemoticon.csv > training.1600000.processed.noemoticon.csv

testdata.manual.2009.06.14.csv: trainingandtestdata.zip
	unzip -p trainingandtestdata.zip testdata.manual.2009.06.14.csv > testdata.manual.2009.06.14.csv

trainingandtestdata.zip:
	wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

