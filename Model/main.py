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
import shutil

from graph import hyperGraph
from model import *

def check_end(args):
    flag = False
    if os.path.isfile(args.save_path + "log.txt"):
        # lossfile.write(f'epoch:{epoch}, loss_sv:{loss_sv.detach().cpu().item()}, loss_deg:{loss_deg.detach().cpu().item()}, loss_sz:{loss_sz.detach().cpu().item()}\n')
        loss_min = -1
        cur_patience = 0
        with open(args.save_path + "log.txt", "r") as f:
            for line in f.readlines():
                ep_str, svloss_str, szloss_str, degloss_str = line.rstrip().split(", ")
                svloss = float(svloss_str.split(":")[-1])
                szloss = float(szloss_str.split(":")[-1])
                degloss = float(degloss_str.split(":")[-1])
                loss = svloss + (args.sizelambda * szloss) + (args.deglambda * degloss)
                epoch = int(ep_str.split(":")[-1])
                if loss_min == -1 or loss_min > loss:
                    loss_min = loss
                    cur_patience = 0
                else:
                    cur_patience += 1
                if cur_patience >= args.patience:
                    flag = True
                    break
                elif epoch >= (args.max_epochs-1):
                    flag = True
                    break
    return flag

def evaluate_model(model, args, hgraph, answer_svds):
    print("Evaluate")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")
    checkpoint = torch.load(args.load_path + "model.pt", map_location=device)
    model.load_state_dict(checkpoint)
    
    eval_svloss_list = []
    eval_szloss_list = []
    eval_degloss_list = []
    eval_min_svloss = (1e+12, 1e+12, 1e+12)
    eval_min_szloss = (1e+12, 1e+12, 1e+12)
    eval_min_degloss = (1e+12, 1e+12, 1e+12)
    model.eval()
    with torch.no_grad():
        for i in range(args.save_iter):
            cur_svloss, cur_szloss, cur_degloss, cur_svds = model.write_matrix(hgraph, args.batch_size, args.save_path + "sampled_{}.txt".format(i))
            eval_svloss_list.append(cur_svloss)
            eval_szloss_list.append(cur_szloss)
            eval_degloss_list.append(cur_degloss)
            print("[Eval Loss] sv loss: %.4f\t size loss: %.4f\t deg loss: %.4f" % (cur_svloss, cur_szloss, cur_degloss))
            
    print(f"[END] Eval: svloss:{np.mean(eval_svloss_list)}, szloss:{np.mean(eval_szloss_list)}, degloss:{np.mean(eval_degloss_list)}")
    with open(args.save_path + "final_eval_log.txt", '+a') as lossfile:   
        lossfile.write(f'epoch:{args.max_epochs}, svloss:{np.mean(eval_svloss_list)}, szloss:{np.mean(eval_szloss_list)}, degloss:{np.mean(eval_degloss_list)}\n')
    with open(args.save_path + "final_eval_log_dev.txt", '+a') as lossfile:   
        lossfile.write(f'epoch:{args.max_epochs}, svloss:{np.std(eval_svloss_list)}, szloss:{np.std(eval_szloss_list)}, degloss:{np.std(eval_degloss_list)}\n')
        
def train_model(model, args, hgraph, answer, answer_sizes, answer_degrees, device):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_loss = 1e+12
    train_loss_min = 1e+12
    train_patience = 0
    save_cnt = 1
    temperature = args.temp
    temp_min = 0.5
    
    for epoch in range(args.max_epochs):
        train_start_time = time.time()
        model.train()
        optimizer.zero_grad()
        expected_sv, expected_sz, expected_deg = model(temperature)
        # sz
        loss_sz = criterion(expected_sz, answer_sizes)
        # deg
        loss_deg = criterion(expected_deg, answer_degrees)
        # sv
        min_len = min(expected_sv.shape[0], answer.shape[0])
        loss_sv = criterion(expected_sv[:min_len], answer[:min_len])
       
        loss = loss_sv + (args.sizelambda * loss_sz) + (args.deglambda * loss_deg)

        loss.backward()
        optimizer.step()
        
        train_end_time = time.time()
        with open(args.save_path + "train_time.txt", "w") as f:
            f.write(str(train_end_time - train_start_time) + " sec\n")
        
        if epoch > 100 and epoch % 100 == 1:
            temperature = np.maximum(temperature * np.exp(-args.annealrate * epoch), 0.1)
            with open(args.save_path + "temp_log.txt", '+a') as file:   
                file.write(f'epoch:{epoch}, temp:{temperature}\n')  
        
        if loss.detach().cpu().item() < train_loss_min:
            train_loss_min = loss.detach().cpu().item()
            train_patience = 0
            torch.save(model.state_dict(), args.save_path + "model.pt")
        elif loss.detach().cpu().item() > train_loss_min:
            train_patience += 1
        if train_patience >= args.patience:
            break
            
        with open(args.save_path + "log.txt", '+a') as lossfile:   
            lossfile.write(f'epoch:{epoch}, loss_sv:{loss_sv.detach().cpu().item()}, loss_deg:{loss_deg.detach().cpu().item()}, loss_sz:{loss_sz.detach().cpu().item()}\n')
        print(f'epoch:{epoch}, loss:{loss.detach().cpu().item()}, loss_sv:{loss_sv.detach().cpu().item()}, loss_deg:{loss_deg.detach().cpu().item()}, loss_sz:{loss_sz.detach().cpu().item()}\n')
        
        if (epoch > (args.save_epoch * save_cnt)) or (epoch == (args.max_epochs - 1)):
            print("Sample ...")
            model.eval()
            eval_loss = 0
            eval_szloss = 0
            eval_degloss = 0
            with torch.no_grad():
                for i in range(args.save_iter):
                    cur_loss, cur_szloss, cur_degloss, cur_svds = model.write_matrix(hgraph, args.batch_size, args.save_path + "sampled_" + str(i) +  ".txt", nosave_flag=True)
                    eval_loss += cur_loss
                    eval_szloss += cur_szloss
                    eval_degloss += cur_degloss
                            
            eval_loss /= args.save_iter
            eval_szloss /= args.save_iter
            eval_degloss /= args.save_iter
            with open(args.save_path + "eval_log.txt", '+a') as lossfile:   
                lossfile.write(f'epoch:{epoch}, svloss:{eval_loss}, szloss:{eval_szloss}, degloss:{eval_degloss}\n')  
            print(f"Eval svLoss: {eval_loss}, szloss:{eval_szloss}, degloss:{eval_degloss}")
            save_cnt += 1
            
    print("End of Training")
                    
    
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.set_num_threads(4)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train, eval')
    parser.add_argument("-de", "--device", action="store", nargs='+', type=int, help="gpu id")
    
    parser.add_argument("-id", "--inputdir", action="store", default="./input/", type=str, help="data directory")
    parser.add_argument("-d", "--dataset", type=str, help="name of dataset, e.g., email-Enron-full")
    parser.add_argument("-r", "--init_row", action="store", default=0, type=int, help="number of row of the initiator matrix")
    parser.add_argument("-c", "--init_col", action="store", default=0, type=int, help="number of column of the initiator matrix")
    parser.add_argument("-param", "--numparam", action="store", default=100, type=int, help="limit of the size of the initiator matrix")
    
    parser.add_argument("-b", "--batch_size", action="store", default=10**6, type=int, help="number of elements generated at once")
    parser.add_argument("-e", "--max_epochs", action="store", default=100000, type=int, help="maximum number of training iteration")
    parser.add_argument("-se", "--save_epoch", action="store", default=100000, type=int, help="period of sampling from HyRec during training") 
    parser.add_argument("-si", "--save_iter", action="store", default=1, type=int, help="iteration of sampling from HyRec")
    parser.add_argument("-lr", "--lr", action="store", default=1e-2, type=float, help="learning rate")
    parser.add_argument("-sld", "--sizelambda", action="store", default=0.0, type=float, help="weight for the size loss")
    parser.add_argument("-dld", "--deglambda", action="store", default=0.0, type=float, help="weight for the degree loss")
    parser.add_argument("-nt", "--num_unit", action="store", default=1, type=int, help="")
    
    parser.add_argument("--recalculateflag", action="store_true", help="retrain the model eventhough it is already trained")
    parser.add_argument("--extflag", action="store_true", help="used for extrapolation")
    parser.add_argument("--noeval", action="store_true", help="used for not wanting to sample from the model")
    
    parser.add_argument("-lp", "--load_path", action="store", default="", type=str, help="path for the trained model to load")
    parser.add_argument("-sp", "--save_path", action="store", default="", type=str, help="path for the trained model to save")
    
    parser.add_argument("-tp", "--temp", action="store", default=1.0, type=float, help="initial temperature in gumbel softmax")
    parser.add_argument("-pat", "--patience", action="store", default=1000, type=int, help="patience for early-stopping")
    parser.add_argument("-ar", "--annealrate", action="store", default=0.00003, type=float, help="anneal rate of reducing temperature")
    args = parser.parse_args()
    
    # initialize_device
    device = torch.device("cuda:" + str(args.device[0]) if torch.cuda.is_available() else "cpu")
            
    # Load graph    
    data_file = args.inputdir + args.dataset + ".txt"    
    hgraph = hyperGraph(data_file, args.inputdir)
    print("Calculate SVs ...")
    hgraph.calculate_sv()
    print(f'rows: {hgraph.num_row}, columns:{hgraph.num_col}, nnz:{hgraph.real_num_nonzero}')
    
    # Initialize model
    if len(args.load_path) > 0:
        tmp = args.load_path.split("/")[-2].split("_")
        args.init_row = int(tmp[0])
        args.init_col = int(tmp[1])
        args.num_unit = int(tmp[4])
        prev_k = int(tmp[2])
        args.k = max(math.ceil(math.log(hgraph.num_row) / math.log(args.init_row)), math.ceil(math.log(hgraph.num_col) / math.log(args.init_col)))
        if args.extflag and prev_k == args.k:
            sys.exit("No Size Change")
        print(prev_k, args.k)
        assert args.init_row ** args.k >= hgraph.num_row
        assert args.init_col ** args.k >= hgraph.num_col
    elif args.init_row + args.init_col == 0:
        diff = 1e+12
        for _k in range(2, 50):
            init_row = math.ceil(hgraph.num_row ** (1./_k))
            init_col = math.ceil(hgraph.num_col ** (1./_k))
            if init_row <= 1.0 or init_col <= 1.0:
                break
            cur_diff = ((init_row ** _k) * (init_col ** _k)) - (hgraph.num_row * hgraph.num_col)
            if (cur_diff < diff) and (init_row * init_col < args.numparam):
                diff = cur_diff
                args.k = _k
        args.init_row = int(math.ceil(hgraph.num_row ** (1./args.k)))
        args.init_col = int(math.ceil(hgraph.num_col ** (1./args.k)))
    else:
        args.k = max(math.ceil(math.log(hgraph.num_row) / math.log(args.init_row)), math.ceil(math.log(hgraph.num_col) / math.log(args.init_col)))
        assert args.init_row ** args.k >= hgraph.num_row
        assert args.init_col ** args.k >= hgraph.num_col
    if args.k < args.num_unit:
        sys.exit("Not Valid K and num_unit")
        
    print(f'k: {args.k}, init row: {args.init_row}, init col: {args.init_col}')
    args.init_row_k = args.init_row ** args.k
    args.init_col_k = args.init_col ** args.k
    
    if len(args.save_path) == 0 and len(args.load_path) == 0:
        args.save_path = f"./result/{args.dataset}/"
        args.save_path += f"{args.init_row}_{args.init_col}_{args.k}_" + "%.3f_%d_%.5f_sl%.4f_dl%.4f/" % (args.lr, args.num_unit, args.annealrate, args.sizelambda, args.deglambda)
        args.load_path = args.save_path
        if os.path.isdir(args.save_path) is False:
            if args.action == "train":
                os.makedirs(args.save_path)
            else:
                print(args.save_path)
                sys.exit("No Exist")
    elif len(args.load_path) > 0:
        if len(args.save_path) == 0:
            args.save_path = args.load_path
        else:
            args.save_path += f"{args.dataset}/"
            args.save_path += f"{args.init_row}_{args.init_col}_{args.k}_" + "%.3f_%d_%.5f_sl%.4f_dl%.4f/" % (args.lr, args.num_unit, args.annealrate, args.sizelambda, args.deglambda)
        if args.extflag:
            args.save_path += "full/"
        if os.path.isdir(args.save_path) is False:
            os.makedirs(args.save_path)
            
    # ---------------------------------------------------------------------------------
    model = SingFit(args.init_row, args.init_col, args.k, device, hgraph.sq_sum, args.num_unit, args.save_path)
    if len(args.device) > 1:
        model = nn.DataParallel(model, device_ids = args.device)
    model = model.to(device)
    
    print("k List", model.intermediate_klist)
                  
    answer_svds = hgraph.svds.to(device)
    answer_sizes = hgraph.size_dist
    answer_sizes = torch.concat([answer_sizes, torch.zeros((args.init_col ** args.k) - hgraph.num_col)])
    answer_sizes = answer_sizes.to(device)
    
    answer_degrees = hgraph.degree_dist
    answer_degrees = torch.concat([answer_degrees, torch.zeros((args.init_row ** args.k) - hgraph.num_row)])
    answer_degrees = answer_degrees.to(device)
    print()
        
    # --------------------------------------------------
    if args.action == "train":
        if args.recalculateflag is False and check_end(args):
            print(args.save_path)
            if args.noeval is False:
                evaluate_model(model, args, hgraph, answer_svds)
            sys.exit("Already exist")

        out_fname_list = ["log.txt", "temp_log.txt", "eval_log.txt",
                         "final_eval_log.txt", "final_eval_log_dev.txt",
                         "train_time.txt", "eval_time.txt"]          
        for out_fname in out_fname_list:
            if os.path.isfile(args.save_path + out_fname):
                  os.remove(args.save_path + out_fname)
                  
        train_model(model, args, hgraph, answer_svds, answer_sizes, answer_degrees, device)
        
    if args.noeval is False:
        evaluate_model(model, args, hgraph, answer_svds)
                  
    os._exit(1)
