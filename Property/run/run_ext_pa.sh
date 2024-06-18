dataset=("email-Eu-half")

cd ../
for data in ${dataset[@]}
do
    ./bin/Evaluation --inputpath ./dataset/${data}_pa --outputdir results/hyperpa/${data}/ --dupflag
    cd src
    python calculation_helper.py --inputpath ../dataset/${data}_pa --outputdir ../results/hyperpa/${data}/ --sv --effdiam --dupflag
    cd ../
    # ext
    ./bin/Evaluation --inputpath ./dataset/${data}_ext_pa --outputdir results/ext_hyperpa/${data}/ --dupflag
    cd src
    python calculation_helper.py --inputpath ../dataset/${data}_ext_pa --outputdir ../results/ext_hyperpa/${data}/ --sv --effdiam --dupflag
    cd ../
done