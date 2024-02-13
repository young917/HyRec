dataset=("tags-ask-ubuntu" "tags-math-sx" "threads-ask-ubuntu" "threads-math-sx")
lrset=("0.1" "0.05" "0.01")
anset=("0.00005" "0.0001" "0.0005" "0.001" "0.005" "0.01")
apxset=("2" "3")
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
                python main.py train --dataset ${data} --numparam 200 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an} --sizelambda 1.0
                python main.py train --dataset ${data} --numparam 100 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an} --sizelambda 1.0
                python main.py train --dataset ${data} --numparam 50 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an} --sizelambda 1.0
                python main.py train --dataset ${data} --numparam 25 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an} --sizelambda 1.0
            done
        done
    done
done