cp -r ../encode.txt raw_data
rm -r result/*
rm -r acc/*

python main.py

rm -r result_final/bagging/*

python get_final_result.py

cp -r result_final/bagging result

python 2_statistics.py

python 3_getFinalReport.py

python 4_result.py

python statistic_analysis.py
python tsne.py
