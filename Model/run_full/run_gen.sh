cuda=0

cd ../
# email-Enron-full
python main.py eval --dataset email-Enron-full --load_path ./saved_model/email-Enron-full/3_5_6_0.010_2_lt0_0.00005_0.010_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1
# email-Eu-full
python main.py eval --dataset email-Eu-full --load_path ./saved_model/email-Eu-full/4_8_6_0.008_3_lt0_0.00005_0.000_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1
# contact-high-school
python main.py eval --dataset contact-high-school --load_path ./saved_model/contact-high-school/4_12_5_0.010_2_lt0_0.00005_1.500_dl0.000/ --save_path ./result/ --device ${cuda} --save_iter 1
# contact-primary-school
python main.py eval --dataset contact-primary-school --load_path ./saved_model/contact-primary-school/4_19_4_0.010_2_lt0_0.00005_0.6000_dl0.0001/ --save_path ./result/ --device ${cuda} --save_iter 1
# NDC-classes-full
python main.py eval --dataset NDC-classes-full --load_path ./saved_model/NDC-classes-full/5_9_5_0.008_2_lt0_0.00005_0.100_dl0.000/ --save_path ./result/ --device ${cuda} --save_iter 1
# NDC-substances-full
python main.py eval --dataset NDC-substances-full --load_path ./saved_model/NDC-substances-full/18_49_3_0.005_2_lt0_0.00005_0.500_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1
# tags-ask-ubuntu
python main.py eval --dataset tags-ask-ubuntu --load_path ./saved_model/tags-ask-ubuntu/8_23_4_0.010_2_lt0_0.00005_2.000_dl0.010/ --save_path ./result/ --device ${cuda} --save_iter 1
# tags-math-sx
python main.py eval --dataset tags-math-sx --load_path ./saved_model/tags-math-sx/7_31_4_0.005_2_lt0_0.00005_1.000_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1
# threads-ask-ubuntu
python main.py eval --dataset threads-ask-ubuntu --load_path ./saved_model/threads-ask-ubuntu/8_8_6_0.010_2_lt0_0.00005_0.500_dl0.100/ --save_path ./result/ --device ${cuda} --save_iter 1
# threads-math-sx
python main.py eval --dataset threads-math-sx --load_path ./saved_model/threads-math-sx/12_15_5_0.003_2_lt0_0.00005_1.000_dl0.010/ --save_path ./result/ --device ${cuda} --save_iter 1
# coauth-MAG-Geology-full
python main.py eval --dataset coauth-MAG-Geology-full --load_path ./saved_model/coauth-MAG-Geology-full/6_6_8_0.005_2_lt0_0.00005_0.000_dl0.001/ --save_path ./result/ --device ${cuda} --save_iter 1