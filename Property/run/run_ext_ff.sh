dataset=("email-Eu-half")
paramset=("0.45_0.2" "0.45_0.3" "0.48_0.2" "0.48_0.3" "0.51_0.2" "0.51_0.3")

cd ..
for param in ${paramset[@]}
do
    for data in ${dataset[@]}
    do
        ./bin/Evaluation --inputpath ./dataset/${data}_ff_${param} --outputdir ./results/hyperff/${data}/${param}/ --dupflag
        cd src
        python calculation_helper.py --inputpath ../dataset/${data}_ff_${param} --outputdir ../results/hyperff/${data}/${param}/ --sv --effdiam --dupflag
        cd ..
        # ext
        ./bin/Evaluation --inputpath ./dataset/${data}_ext_ff_${param} --outputdir ./results/ext_hyperff/${data}/${param}/ --dupflag
        cd src
        python calculation_helper.py --inputpath ../dataset/${data}_ext_ff_${param} --outputdir ../results/ext_hyperff/${data}/${param}/ --sv --effdiam --dupflag
        cd ..
    done
done