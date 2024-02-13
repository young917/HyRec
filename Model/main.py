import sys
import torch
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import sys

from graph import hyperGraph
from model import *

def check_end(args):
    flag = False
    if os.path.isfile(args.save_path + "log.txt"):
        loss_min = -1
        cur_patience = 0
        with open(args.save_path + "log.txt", "r") as f:
            for line in f.readlines():
                ep_str, loss_str = line.rstrip().split(", ")
                loss = float(loss_str.split(":")[-1])
                if loss_min == -1 or loss_min > loss:
                    loss_min = loss
                    cur_patience = 0
                else:
                    cur_patience += 1
                if cur_patience >= 300:
                    flag = True
                    break
    return flag

def evaluate_model(hyperk_model, args, hgraph, answer_svds, answer_sizes):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")
    checkpoint = torch.load(args.load_path + "min.pt", map_location=device)
    hyperk_model.load_state_dict(checkpoint['model_state_dict'])
    
    eval_loss = 0
    hyperk_model.eval()
    with torch.no_grad():
        for i in range(args.save_iter):
            cur_loss, cur_svds = hyperk_model.write_matrix(hgraph, args.batch_size, args.save_path + "sampled_" + str(i) +  ".txt", atonce_flag=args.gen_at_once)
            eval_loss += cur_loss
            if i == 0:
                with open(args.save_path + "final_eval_log_svs.txt", '+a') as logfile:   
                    logfile.write(f'epoch:{args.max_epochs}\n') 
                    cur_svds = cur_svds.numpy()
                    for cur_sv in cur_svds:
                        logfile.write(str(cur_sv) + "\n")
    eval_loss /= args.save_iter
    
    print("[END] Eval Loss: %f" % (eval_loss))
    with open(args.save_path + "final_eval_log.txt", '+a') as lossfile:   
        lossfile.write(f'epoch:{args.max_epochs}, loss:{eval_loss}\n')

### SINGFIT ### -------------------------------------------------------
def train_model(hyperk_model, args, hgraph, answer, answer_sizes):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(hyperk_model.parameters(), lr=args.lr)
    
    train_loss = 1e+12
    train_loss_min = 1e+12
    train_patience = 0
    save_cnt = 1
    temperature = args.temp
    temp_min = 0.5
    
    for epoch in range(args.max_epochs):
        train_start_time = time.time()
        hyperk_model.train()
        optimizer.zero_grad()
        expected_sv, expected_sz = hyperk_model(temperature)
        min_len = min(expected_sv.shape[0], answer.shape[0])
        loss_sz = (args.sizelambda * criterion(expected_sz, answer_sizes))
        loss = criterion(expected_sv[:min_len], answer[:min_len]) + (args.sizelambda * criterion(expected_sz, answer_sizes))
        print("Size Loss:", loss_sz.item())
        print(expected_sv[:5], "\t", answer[:5])
        
        loss.backward()
        optimizer.step()
        train_end_time = time.time()
        with open(args.save_path + "train_time.txt", "w") as f:
            f.write(str(train_end_time - train_start_time) + " sec\n")
        
        if epoch > 100 and epoch % 100 == 1:
            temperature = np.maximum(temperature * np.exp(-args.annealrate * epoch), 0.1) #, 0.5)
            with open(args.save_path + "temp_log.txt", '+a') as file:   
                file.write(f'epoch:{epoch}, temp:{temperature}\n')  
        
        with open(args.save_path + "log.txt", '+a') as lossfile:   
            lossfile.write(f'epoch:{epoch}, loss:{loss.detach().cpu().item()}\n')
        with open(args.save_path + "size_log.txt", '+a') as lossfile:   
            lossfile.write(f'epoch:{epoch}, loss:{loss_sz.detach().cpu().item()}\n')  
        if loss.detach().cpu().item() < train_loss_min:
            train_loss_min = loss.detach().cpu().item()
            train_patience = 0
        elif loss.detach().cpu().item() > train_loss_min:
            train_patience += 1
        if train_patience >= 300:
            break
        print(f'epoch:{epoch}, loss:{loss.detach().cpu().item()}\n')
        
        if loss.detach().cpu().item() < train_loss:
            train_loss = loss.detach().cpu().item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': hyperk_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, args.save_path + "min.pt") 
                    
        if (epoch > (args.save_epoch * save_cnt)) or (epoch == (args.max_epochs - 1)):
            print("Sample ...")
            hyperk_model.eval()
            eval_loss = 0
            with torch.no_grad():
                for i in range(args.save_iter):
                    cur_loss, cur_svds = hyperk_model.write_matrix(hgraph, args.batch_size, args.save_path + "sampled_" + str(i) +  ".txt", atonce_flag=args.gen_at_once, nosave_flag=True)
                    eval_loss += cur_loss
                    if i == 0:
                        with open(args.save_path + "eval_log_svs.txt", '+a') as logfile:   
                            logfile.write(f'epoch:{epoch}\n') 
                            for svi in range(10):
                                logfile.write(str(cur_svds[svi]) + "\n")
                            
            eval_loss /= args.save_iter
            with open(args.save_path + "eval_log.txt", '+a') as lossfile:   
                lossfile.write(f'epoch:{epoch}, loss:{eval_loss}\n')  
            print("Eval Loss: %f" % (eval_loss))
            save_cnt += 1
                
    for i in range(5):
        print(str(answer[i]) + "\t" + str(expected_sv[i]))
        print()
                    
    
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.set_num_threads(8)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train, eval')
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-r", "--init_row", action="store", default=0, type=int)
    parser.add_argument("-c", "--init_col", action="store", default=0, type=int)
    parser.add_argument("-param", "--numparam", action="store", default=100, type=int)
    parser.add_argument("-de", "--device", action="store", nargs='+', type=int)
#     parser.add_argument("-de", "--device", action="store", type=int)
    
    parser.add_argument("-b", "--batch_size", action="store", default=10**6, type=int) # 2**18 for generation
    parser.add_argument("-e", "--max_epochs", action="store", default=10000, type=int) # 10**5
    parser.add_argument("-se", "--save_epoch", action="store", default=100, type=int) # 10**5, 5000
    parser.add_argument("-si", "--save_iter", action="store", default=1, type=int) # 10**5, 5000
    parser.add_argument("-lr", "--lr", action="store", default=1e-2, type=float)
    parser.add_argument("-sld", "--sizelambda", action="store", default=0.0, type=float)
    parser.add_argument("--recalculateflag", action="store_true") 
    parser.add_argument("-lp", "--load_path", action="store", default="", type=str)
    parser.add_argument("-sp", "--save_path", action="store", default="", type=str)
    parser.add_argument("-ap", "--approx", action="store", default=1, type=int)
    parser.add_argument("-eap", "--evalapprox", action="store", default=1, type=int)
    parser.add_argument("-gat", "--gen_at_once", action="store_true")
    parser.add_argument("-gb", "--gumbel", action="store", default=1, type=int)
    parser.add_argument("-tp", "--temp", action="store", default=1.0, type=float)
    parser.add_argument("-ar", "--annealrate", action="store", default=0.00003, type=float)
    args = parser.parse_args()
    
    # initialize_device
#     device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:" + str(args.device[0]) if torch.cuda.is_available() else "cpu")
            
    # Load graph    
    data_file = "./input/" + args.dataset + ".txt"    
    hgraph = hyperGraph(data_file)
    print("Calculate SVs ...")
    hgraph.calculate_sv()
    print(f'rows: {hgraph.num_row}, columns:{hgraph.num_col}, nnz:{hgraph.real_num_nonzero}')
    
    # Initialize model
    if len(args.load_path) > 0:
        tmp = args.load_path.split("/")[-2].split("_")
        assert len(tmp) >= 8
        args.init_row = int(tmp[0])
        args.init_col = int(tmp[1])
        args.approx = int(tmp[4])
        args.evalapprox = int(tmp[5])
        prev_k = int(tmp[2])
        args.k = max(math.ceil(math.log(hgraph.num_row) / math.log(args.init_row)), math.ceil(math.log(hgraph.num_col) / math.log(args.init_col)))
        if "half" in args.load_path and prev_k == args.k:
            sys.exit("No Size Change")
        print(prev_k, args.k)
        assert args.init_row ** args.k >= hgraph.num_row
        assert args.init_col ** args.k >= hgraph.num_col
    elif args.init_row + args.init_col == 0:
        diff = 1e+12
        for _k in range(3, 10):
            init_row = math.ceil(hgraph.num_row ** (1./_k))
            init_col = math.ceil(hgraph.num_col ** (1./_k))
            diff_row = (init_row ** _k) - hgraph.num_row
            diff_col = (init_col ** _k) - hgraph.num_col
            assert diff_row >= 0.0
            assert diff_col >= 0.0
            if (diff_row + diff_col < diff) and (init_row * init_col < args.numparam):
                diff = diff_row + diff_col
                args.k = _k
        args.init_row = int(math.ceil(hgraph.num_row ** (1./args.k)))
        args.init_col = int(math.ceil(hgraph.num_col ** (1./args.k)))
    else:
        args.k = max(math.ceil(math.log(hgraph.num_row) / math.log(args.init_row)), math.ceil(math.log(hgraph.num_col) / math.log(args.init_col)))
        assert args.init_row ** args.k >= hgraph.num_row
        assert args.init_col ** args.k >= hgraph.num_col
    
    if args.k < args.approx:
        sys.exit("Not Valid K and Approx")
        
    print(f'k: {args.k}, init row: {args.init_row}, init col: {args.init_col}')
    args.init_row_k = args.init_row ** args.k
    args.init_col_k = args.init_col ** args.k
    
    if len(args.save_path) == 0 and len(args.load_path) == 0:
        if args.action == "train":
            args.save_path = f"./result/{args.dataset}/"
        args.save_path += f"{args.init_row}_{args.init_col}_{args.k}_" + "%.3f_%d_%d_%d_%.5f_%.2f/" % (args.lr, args.approx, args.evalapprox, args.gumbel, args.annealrate, args.sizelambda)
        args.load_path = args.save_path
        
        if os.path.isdir(args.save_path) is False:
            if args.action == "train":
                os.makedirs(args.save_path)
            else:
                sys.exit("No Exist")
    elif len(args.load_path) > 0:
        if "half" in args.load_path:
            args.save_path = args.load_path + "full/"
            if os.path.isdir(args.save_path) is False:
                os.makedirs(args.save_path)
        else:
            args.save_path = args.load_path
    # ---------------------------------------------------------------------------------
    hyperk_model = HyperK(args.init_row, args.init_col, args.k, device, hgraph.sq_sum, args.gumbel, args.approx, args.evalapprox, args.save_path)
    if len(args.device) > 1:
        hyperk_model = nn.DataParallel(hyperk_model, device_ids = args.device)
    hyperk_model = hyperk_model.to(device)
    
    print("Answer svds")
    answer_svds = hgraph.svds.to(device)
    for i, sv in enumerate(hgraph.svds):
        print(sv, end=", ")
        if i > 20:
            break
    answer_sizes = hgraph.size_dist
    answer_sizes = torch.concat([answer_sizes, torch.zeros((args.init_col ** args.k) - hgraph.num_col)])
    answer_sizes = answer_sizes.to(device)
    print()
        
    # --------------------------------------------------
    if args.recalculateflag is False and check_end(args):
        print(args.save_path)
        evaluate_model(hyperk_model, args, hgraph, answer_svds, answer_sizes)
        sys.exit("Already exist")
    elif args.action == "train":
        if os.path.isfile(args.save_path + "log.txt"):
            os.remove(args.save_path + "log.txt")
        if os.path.isfile(args.save_path + "eval_log.txt"):
            os.remove(args.save_path + "eval_log.txt")
        if os.path.isfile(args.save_path + "temp_log.txt"):
            os.remove(args.save_path + "temp_log.txt")
        if os.path.isfile(args.save_path + "train_gen_log.txt"):
            os.remove(args.save_path + "train_gen_log.txt")
        train_model(hyperk_model, args, hgraph, answer_svds, answer_sizes)
    
    evaluate_model(hyperk_model, args, hgraph, answer_svds, answer_sizes)
