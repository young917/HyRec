dataset=("email-Eu-full" "email-Eu-half")

# answer
cd ../
for data in ${dataset[@]}
do
    ./bin/Evaluation --inputpath ./dataset/${data} --outputdir ./results/answer/${data}/ --dupflag
    cd src
    python calculation_helper.py --inputpath ../dataset/${data} --outputdir ../results/answer/${data}/ --sv --effdiam --dupflag
    cd ../
done