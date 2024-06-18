cd ..
./bin/Evaluation --inputpath ./results/ext_HyperK/email-Eu-half/0/hypergraph --outputdir ./results/ext_HyperK/email-Eu-half/0/ --dupflag
cd src
python calculation_helper.py --inputpath ../results/ext_HyperK/email-Eu-half/0/hypergraph --outputdir ../results/ext_HyperK/email-Eu-half/0/ --sv --effdiam --dupflag --appxflag
cd ../
