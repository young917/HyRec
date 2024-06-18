cd ..
./bin/Evaluation --inputpath ./results/HyperK/email-Eu-full/0/hypergraph --outputdir ./results/HyperK/email-Eu-full/0/ --dupflag
cd src
python calculation_helper.py --inputpath ../results/HyperK/email-Eu-full/0/hypergraph --outputdir ../results/HyperK/email-Eu-full/0/ --sv --effdiam --dupflag
cd ../
