halfdataset=("email-Enron-half" "email-Eu-half" "contact-high-school-half" "contact-primary-school-half" "NDC-classes-half" "NDC-substances-half")
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
            for halfdata in ${halfdataset[@]}
            do
                python main_sv.py train --dataset ${halfdata} --numparam 200 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an}
                python main_sv.py train --dataset ${halfdata} --numparam 100 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an}
                python main_sv.py train --dataset ${halfdata} --numparam 50 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an}
                python main_sv.py train --dataset ${halfdata} --numparam 25 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an}
            done
        done
    done
done

# For the best model, generate (predict) full hypergraphs
# python main.py eval --dataset {full data name} --load_path {trained_model_path} --gen_at_once
# python main.py eval --dataset contact-high-school --device ${cuda} --load_path ./result/contact-high-school-half/5_8_4_0.100_1_1_1_0.00050_1.00/ --gen_at_once
