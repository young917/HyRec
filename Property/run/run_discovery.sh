dataset=("email-Enron-full" "contact-primary-school" "contact-high-school" "NDC-classes-full" "email-Eu-full" "NDC-substances-full" "tags-ask-ubuntu" "tags-math-sx" "threads-ask-ubuntu" "threads-math-sx" "coauth-MAG-Geology-full" "coauth-MAG-History-full")

cd ../
for data in ${dataset[@]}
do
    ./bin/Sampling --inputpath ../dataset/${data} --outputdir ./results/answer/${data}/
    cd src
    python calculation_helper.py --inputpath ../../dataset/${data} --outputdir ../results/answer/${data}/ --sv --effdiam
    cd ../
done