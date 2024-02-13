halfdataset=("tags-ask-ubuntu-half" "tags-math-sx-half" "threads-ask-ubuntu-half" "threads-math-sx-half")
lrset=("0.1" "0.05" "0.01")
anset=("0.00005" "0.0001" "0.0005" "0.001" "0.005" "0.01")
apxset=("2" "3")
cuda="1"

cd ../
for lr in ${lrset[@]}
do
    for an in ${anset[@]}
    do
        for apx in ${apxset[@]}
        do
            for halfdata in ${halfdataset[@]}
            do
                python main.py train --dataset ${halfdata} --numparam 200 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an} --sizelambda 1.0
                python main.py train --dataset ${halfdata} --numparam 100 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an} --sizelambda 1.0
                python main.py train --dataset ${halfdata} --numparam 50 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an} --sizelambda 1.0
                python main.py train --dataset ${halfdata} --numparam 25 --device ${cuda} --lr ${lr} --approx ${apx} --evalapprox ${apx} --gen_at_once --annealrate ${an} --sizelambda 1.0
            done
        done
    done
done

# For the best model, generate (predict) full hypergraphs
# python main.py eval --dataset {full data name} --load_path {trained_model_path} --gen_at_once
# e.g. python main.py eval --dataset tags-math-sx --device $cuda --load_path ./result/tags-math-sx-half/5_10_5_0.100_2_2_1_0.00005_1.00/ --gen_at_once