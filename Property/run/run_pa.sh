dataset=("email-Eu-full")

cd ../
for data in ${dataset[@]}
do
    ./bin/Evaluation --inputpath ./dataset/${data}_pa --outputdir ./results/hyperpa/${data}/ --dupflag
    cd src
    python calculation_helper.py --inputpath ../dataset/${data}_pa --outputdir ../results/hyperpa/${data}/ --sv --effdiam --dupflag
    cd ../
done