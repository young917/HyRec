dataset=("email-Eu-half")

bset=("8" "12" "15")
pset=("0.5" "0.7" "0.9")
cset=("2.0" "6.0" "10.0")

cd ../
for b in ${bset[@]}
do
    for p in ${pset[@]}
    do 
        for c in ${cset[@]}
        do
            for data in ${dataset[@]}
            do
                ./bin/Evaluation --inputpath ./dataset/tr/${data}_tr_${b}_${p}_${c} --outputdir results/thera/${data}/${b}_${p}_${c}/ --dupflag
                cd src
                python calculation_helper.py --inputpath ../dataset/tr/${data}_tr_${b}_${p}_${c} --outputdir ../results/thera/${data}/${b}_${p}_${c}/ --sv --effdiam --dupflag
                cd ../
                # ext
                ./bin/Evaluation --inputpath ./dataset/tr/${data}_ext_tr_${b}_${p}_${c} --outputdir results/ext_thera/${data}/${b}_${p}_${c}/ --dupflag
                cd src
                python calculation_helper.py --inputpath ../dataset/tr/${data}_ext_tr_${b}_${p}_${c} --outputdir ../results/ext_thera/${data}/${b}_${p}_${c}/ --sv --effdiam --dupflag
                cd ../
            done
        done
    done
done