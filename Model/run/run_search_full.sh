dataset=("email-Enron-full" "email-Eu-full" "contact-high-school" "contact-primary-school" "NDC-classes-full" "NDC-substances-full" "tags-ask-ubuntu" "tags-math-sx" "threads-ask-ubuntu" "threads-math-sx" "coauth-MAG-Geology-full")
anset=("0.00005")
lrset=("0.001" "0.003" "0.005" "0.008" "0.01")
szlambdaset=("0.0" "0.01" "0.1" "0.5" "0.6" "1.0" "1.5" "2.0")
deglambdaset=("0.0" "0.0001" "0.001" "0.01" "0.1")
paramset=("50" "100" "1000")
unitset=("2" "3" "4")
cuda=0


cd ..
for an in ${anset[@]}
do
    for lr in ${lrset[@]}
    do
        for unit in ${unitset[@]}
        do
            for sld in ${szlambdaset[@]}
            do
                for dld in ${deglambdaset[@]}
                do
                    for dataname in ${dataset[@]}
                    do
                        for pr in ${paramset[@]}
                        do
                            python main.py train --device ${cuda} --dataset ${dataname} --numparam ${pr} --lr ${lr} --annealrate ${an} --num_unit ${unit} --sizelambda ${sld} --deglambda ${dld} --noeval
                        done
                    done
                done
            done
        done
    done
done
