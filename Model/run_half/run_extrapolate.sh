cuda=0

cd ../
# email-Enron-full
python main.py eval --dataset email-Enron-full --load_path ./saved_model/email-Enron-half/3_4_5_0.010_2_lt0_0.00005_1.000_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# email-Eu-full
python main.py eval --dataset email-Eu-full --load_path ./saved_model/email-Eu-half/4_8_5_0.010_2_lt0_0.00005_1.500_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# contact-high-school
python main.py eval --dataset contact-high-school --load_path ./saved_model/contact-high-school-half/4_12_4_0.010_2_lt0_0.00005_1.000_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# contact-primary-school
python main.py eval --dataset contact-primary-school --load_path ./saved_model/contact-primary-school-half/4_18_4_0.010_2_lt0_0.00005_1.000_dl0.000/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# NDC-classes-full
python main.py eval --dataset NDC-classes-full --load_path ./saved_model/NDC-classes-half/5_9_4_0.001_2_lt0_0.00005_1.0000_dl0.0001/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# NDC-substances-full
python main.py eval --dataset NDC-substances-full --load_path ./saved_model/NDC-substances-half/5_8_5_0.005_2_lt0_0.00005_0.010_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# tags-ask-ubuntu
python main.py eval --dataset tags-ask-ubuntu --load_path ./saved_model/tags-ask-ubuntu-half/3_4_7_0.010_2_lt0_0.00005_1.000_dl0.010/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# tags-math-sx
python main.py eval --dataset tags-math-sx --load_path ./saved_model/tags-math-sx-half/4_8_5_0.010_2_lt0_0.00005_1.000_dl0.010/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# threads-ask-ubuntu
python main.py eval --dataset threads-ask-ubuntu --load_path ./saved_model/threads-ask-ubuntu-half/5_5_7_0.001_2_lt0_0.00005_0.500_dl2.000/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# threads-math-sx
python main.py eval --dataset threads-math-sx --load_path ./saved_model/threads-math-sx-half/18_22_4_0.010_2_lt0_0.00005_1.000_dl0.500/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag
# coauth-MAG-Geology-full
python main.py eval --dataset coauth-MAG-Geology-full --load_path ./saved_model/coauth-MAG-Geology-half/7_7_7_0.100_2_lt0_0.00005_0.000_dl0.000/ --save_path ./result/ --device ${cuda} --save_iter 1 --extflag