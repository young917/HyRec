cd ..
./bin/Evaluation --inputpath ./results/HyRec/email-Eu-full/0/hypergraph --outputdir ./results/HyRec/email-Eu-full/0/ --dupflag
cd src
python calculation_helper.py --inputpath ../results/HyRec/email-Eu-full/0/hypergraph --outputdir ../results/HyRec/email-Eu-full/0/ --sv --effdiam --dupflag
cd ../
