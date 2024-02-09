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
from model_sv import *
# from dataset import *


def train_model(k_model, args, hgraph, answer):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(k_model.parameters(), lr=args.lr)
    
    train_loss = 1e+12
    train_loss_min = 1e+12
    train_patience = 0
    save_cnt = 1
    temperature = args.temp
    temp_min = 0.5
    
    for epoch in range(args.max_epochs):
        k_model.train()
        optimizer.zero_grad()
        expected_sv = k_model(temperature)
        min_len = min(expected_sv.shape[0], answer.shape[0])
        loss = criterion(expected_sv[:min_len], answer[:min_len])
        print(expected_sv[:5], "\t", answer[:5])
        
        loss.backward()
        # print(k_model.sos)
        # print(k_model.sos.grad)
        optimizer.step()
        
        if epoch > 100 and epoch % 100 == 1:
            temperature = np.maximum(temperature * np.exp(-args.annealrate * epoch), 0.1) #, 0.5)
            with open(args.save_path + "temp_log.txt", '+a') as file:   
                file.write(f'epoch:{epoch}, temp:{temperature}\n')  
        
        with open(args.save_path + "log.txt", '+a') as lossfile:   
            lossfile.write(f'epoch:{epoch}, loss:{loss.detach().cpu().item()}\n')  
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
                'model_state_dict': k_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, args.save_path + "min.pt") 
                    
        if (epoch > (args.save_epoch * save_cnt)) or (epoch == (args.max_epochs - 1)):
            print("Sample ...")
            k_model.eval()
            eval_loss = 0
            with torch.no_grad():
                for i in range(args.save_iter):
                    cur_loss, cur_svds = k_model.write_matrix(hgraph, args.batch_size, args.save_path + "sampled_" + str(i) +  ".txt", atonce_flag=args.gen_at_once)
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
    parser.add_argument("-td", "--test_data", action="store", default="none", type=str)
    parser.add_argument("-p", "--perm_file", action="store", default=False, type=bool)
    parser.add_argument("-r", "--init_row", action="store", default=0, type=int)
    parser.add_argument("-c", "--init_col", action="store", default=0, type=int)
    parser.add_argument("-param", "--numparam", action="store", default=100, type=int)
    
    parser.add_argument("-de", "--device", action="store", nargs='+', type=int)
#     parser.add_argument("-de", "--device", action="store", type=int)
    
    parser.add_argument("-b", "--batch_size", action="store", default=10**6, type=int) # 2**18
    parser.add_argument("-e", "--max_epochs", action="store", default=10000, type=int) # 10**5
    parser.add_argument("-se", "--save_epoch", action="store", default=100, type=int) # 10**5, 5000
    parser.add_argument("-si", "--save_iter", action="store", default=1, type=int) # 10**5, 5000
    parser.add_argument("-lr", "--lr", action="store", default=1e-2, type=float)
    parser.add_argument("-ppu", "--perm_per_update", action="store", default=2, type=int)
    parser.add_argument("-sw", "--sample_weight", action="store", default=10, type=float)
    parser.add_argument("-dt", "--data_type", action="store", default="double", type=str)
    
    parser.add_argument("--saveflag", action="store_true")
    parser.add_argument("--checkflag", action="store_true")
    parser.add_argument("--recalculateflag", action="store_true")
    parser.add_argument("--norm_flag", action="store_true")
    parser.add_argument("-tst", "--test", action="store_true")
    
    parser.add_argument("-lp", "--load_path", action="store", default="", type=str)
    parser.add_argument("-sp", "--save_path", action="store", default="", type=str)
    parser.add_argument("-ap", "--approx", action="store", default=1, type=int)
    parser.add_argument("-eap", "--evalapprox", action="store", default=1, type=int)
    parser.add_argument("-gat", "--gen_at_once", action="store_true")
    parser.add_argument("-rt", "--retrain", action="store", default=False, type=bool)
    parser.add_argument("-perm", "--load_perm", action="store", default="False", type=str)
    parser.add_argument("-bin", "--binarize", action="store", default=1, type=int)
    parser.add_argument("-gb", "--gumbel", action="store", default=1, type=int)
    parser.add_argument("-tp", "--temp", action="store", default=1.0, type=float)
    parser.add_argument("-ar", "--annealrate", action="store", default=0.00003, type=float)
    args = parser.parse_args()
    
    # initialize_device
#     device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:" + str(args.device[0]) if torch.cuda.is_available() else "cpu")
            
    # Load graph    
    data_file = "../kronfit_neukron/input/" + args.dataset + ".txt"    
    hgraph = hyperGraph(data_file, args.batch_size, args.binarize)
    print("Calculate SVs ...")
    hgraph.calculate_sv()
    print(f'rows: {hgraph.num_row}, columns:{hgraph.num_col}, nnz:{hgraph.real_num_nonzero}')
    
    # Initialize model
    if args.init_row + args.init_col == 0:
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
    print(f'k: {args.k}, init row: {args.init_row}, init col: {args.init_col}')
    args.init_row_k = args.init_row ** args.k
    args.init_col_k = args.init_col ** args.k
    
    if len(args.save_path) == 0 and len(args.load_path) == 0:
        args.save_path = f"./result/{args.dataset}/"
        args.save_path += f"{args.init_row}_{args.init_col}_{args.k}_" + "%.3f_%d_%d_%d_%.5f/" % (args.lr, args.approx, args.evalapprox, args.gumbel, args.annealrate)
        args.load_path = args.save_path
        
        if os.path.isdir(args.save_path) is False:
            os.makedirs(args.save_path)
        elif args.recalculateflag is False:
            if os.path.isfile(args.save_path + "log.txt"):
                eplist = []
                with open(args.save_path + "log.txt", "r") as f:
                    for line in f.readlines():
                        ep_str, loss_str = line.rstrip().split(", ")
                        ep = int(ep_str.split(":")[-1])
                        eplist.append(ep)
                if len(eplist) > 500:
                    sys.exit("Already exist")
        if os.path.isfile(args.save_path + "log.txt"):
            os.remove(args.save_path + "log.txt")
        if os.path.isfile(args.save_path + "eval_log.txt"):
            os.remove(args.save_path + "eval_log.txt")
        if os.path.isfile(args.save_path + "temp_log.txt"):
            os.remove(args.save_path + "temp_log.txt")
        if os.path.isfile(args.save_path + "train_gen_log.txt"):
            os.remove(args.save_path + "train_gen_log.txt")

    # ---------------------------------------------------------------------------------
    k_model = KroneckerSVFitting(args.init_row, args.init_col, args.k, device, hgraph.sq_sum, args.binarize, args.gumbel, args.approx, args.evalapprox, args.checkflag, args.save_path)
    if len(args.device) > 1:
        k_model = nn.DataParallel(k_model, device_ids = args.device)
    k_model = k_model.to(device)
    
    print("Answer svds")
    answer_svds = hgraph.svds.to(device)
    for i, sv in enumerate(hgraph.svds):
        print(sv, end=", ")
        if i > 20:
            break
    print()
    
    train_model(k_model, args, hgraph, answer_svds)
    
    eval_loss = 0
    k_model.eval()
    for i in range(3):
        cur_loss, cur_svds = k_model.write_matrix(hgraph, args.batch_size, args.save_path + "sampled_" + str(i) +  ".txt", atonce_flag=args.gen_at_once)
        eval_loss += cur_loss
        if i == 0:
            with open(args.save_path + "eval_log_svs.txt", '+a') as logfile:   
                logfile.write(f'epoch:{args.max_epochs}\n') 
                for svi in range(10):
                    logfile.write(str(cur_svds[svi]) + "\n")
    eval_loss /= 3
    
    print("Eval Loss: %f" % (eval_loss))
    with open(args.save_path + "eval_log.txt", '+a') as lossfile:   
        lossfile.write(f'epoch:{args.max_epochs}, loss:{eval_loss}\n')  