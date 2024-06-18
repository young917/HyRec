cuda=0

cd ../
# email-Enron-half
python main.py eval --dataset email-Enron-half --load_path ./saved_model/email-Enron-half/3_4_5_0.010_2_lt0_0.00005_1.000_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1
# email-Eu-half
python main.py eval --dataset email-Eu-half --load_path ./saved_model/email-Eu-half/4_8_5_0.010_2_lt0_0.00005_1.500_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1
# contact-high-school-half
python main.py eval --dataset contact-high-school-half --load_path ./saved_model/contact-high-school-half/4_12_4_0.010_2_lt0_0.00005_1.000_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1
# contact-primary-school-half
python main.py eval --dataset contact-primary-school-half --load_path ./saved_model/contact-primary-school-half/4_18_4_0.010_2_lt0_0.00005_1.000_dl0.000/ --save_path ./result/ --device ${cuda} --save_iter 1
# NDC-classes-half
python main.py eval --dataset NDC-classes-half --load_path ./saved_model/NDC-classes-half/5_9_4_0.001_2_lt0_0.00005_1.0000_dl0.0001/ --save_path ./result/ --device ${cuda} --save_iter 1
# NDC-substances-half
python main.py eval --dataset NDC-substances-half --load_path ./saved_model/NDC-substances-half/5_8_5_0.005_2_lt0_0.00005_0.010_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1
# tags-ask-ubuntu-half
python main.py eval --dataset tags-ask-ubuntu-half --load_path ./saved_model/tags-ask-ubuntu-half/3_4_7_0.010_2_lt0_0.00005_1.000_dl0.010/ --save_path ./result/ --device ${cuda} --save_iter 1
# tags-math-sx-half
python main.py eval --dataset tags-math-sx-half --load_path ./saved_model/tags-math-sx-half/4_8_5_0.010_2_lt0_0.00005_1.000_dl0.010/ --save_path ./result/ --device ${cuda} --save_iter 1
# threads-ask-ubuntu-half
python main.py eval --dataset threads-ask-ubuntu-half --load_path ./saved_model/threads-ask-ubuntu-half/5_5_7_0.001_2_lt0_0.00005_0.500_dl2.000/ --save_path ./result/ --device ${cuda} --save_iter 1
# threads-math-sx-half
python main.py eval --dataset threads-math-sx-half --load_path ./saved_model/threads-math-sx-half/18_22_4_0.010_2_lt0_0.00005_1.000_dl0.500/ --save_path ./result/ --device ${cuda} --save_iter 1
# coauth-MAG-Geology-half
python main.py eval --dataset coauth-MAG-Geology-half --load_path ./saved_model/coauth-MAG-Geology-half/7_7_7_0.100_2_lt0_0.00005_0.000_dl0.000/ --save_path ./result/ --device ${cuda} --save_iter 1