dataset=("email-Enron-full" "email-Eu-full" "contact-high-school" "contact-primary-school" "NDC-classes-full" "NDC-substances-full")
lrset=("0.1" "0.05" "0.01")
anset=("0.00005" "0.0001" "0.0005" "0.001" "0.005" "0.01")
apxset=("1" "2" "3")
cuda="0"

cd ../
for lr in ${lrset[@]}
do
    for an in ${anset[@]}
    do
        for apx in ${apxset[@]}
        do
            for data in ${dataset[@]}
            do
                python main_sv.py train --dataset ${data} --numparam 200 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an}
                python main_sv.py train --dataset ${data} --numparam 100 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an}
                python main_sv.py train --dataset ${data} --numparam 50 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an}
                python main_sv.py train --dataset ${data} --numparam 25 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an}
            done
        done
    done
done