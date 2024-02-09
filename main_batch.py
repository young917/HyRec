import sys
import torch
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

from graph import hyperGraph
from model import *
from dataset import *

def eval_loss(k_model, args, inc_dataset, adj_dataset, line_dataset, samplenum=3):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")  
    
    if len(args.load_path) == 0:
        checkpoint = torch.load(args.save_path + "min.pt", map_location=device)
    else:
        checkpoint = torch.load(args.load_path + "min.pt", map_location=device)
        
    k_model.model.load_state_dict(checkpoint['model_state_dict'])
    k_model.node_perm = checkpoint['node_perm']
    k_model.edge_perm = checkpoint['edge_perm']
    
    inc_dataset.update_perm(k_model.node_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
    adj_dataset.update_perm(k_model.node_perm.detach().cpu(), k_model.node_perm.detach().cpu())
    line_dataset.update_perm(k_model.edge_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
    inc_loader = torch.utils.data.DataLoader(dataset=inc_dataset, batch_size=1, shuffle=True, num_workers=1)
    adj_loader = torch.utils.data.DataLoader(dataset=adj_dataset, batch_size=1, shuffle=True, num_workers=1)
    line_loader = torch.utils.data.DataLoader(dataset=line_dataset, batch_size=1, shuffle=True, num_workers=1)
    
    k_model.model.eval()
    with torch.no_grad():            
        model_lossI, model_lossA, model_lossL = 0., 0., 0.
        model_lossI += k_model.model.get_sq_sum("I").detach().cpu().item()
        for data in inc_loader:
            inputs, vals = data
            inputs, vals = inputs.squeeze(0), vals.squeeze(0)
            inputs, vals = inputs.to(device), vals.to(device)
            outputs = k_model.model(inputs, "I")
            model_lossI += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
        model_lossA += k_model.model.get_sq_sum("A").detach().cpu().item()
        for data in adj_loader:
            inputs, vals = data
            inputs, vals = inputs.squeeze(0), vals.squeeze(0)
            inputs, vals = inputs.to(device), vals.to(device)
            outputs = k_model.model(inputs, "A")
            model_lossA += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
        model_lossL += k_model.model.get_sq_sum("L").detach().cpu().item()
        for data in line_loader:
            inputs, vals = data
            inputs, vals = inputs.squeeze(0), vals.squeeze(0)
            inputs, vals = inputs.to(device), vals.to(device)
            outputs = k_model.model(inputs, "L")
            model_lossL += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
        if args.norm_flag:
            model_loss = args.lambdaI * (model_lossI / (args.init_row_k * args.init_col_k)) + args.lambdaA * (model_lossA / (args.init_row_k * args.init_row_k)) + args.lambdaL * (model_lossL / (args.init_col_k * args.init_col_k))
        else:
            model_loss = args.lambdaI * model_lossI + args.lambdaA * model_lossA + args.lambdaL * model_lossL
        print(f'Approximation Error I:{model_lossI}')
        print(f'Approximation Error A:{model_lossA}')
        print(f'Approximation Error L:{model_lossL}')
        
        eval_loss = 0
        for i in range(samplenum):
            eval_loss += k_model.write_matrix(args.batch_size, args.save_path + "sampled_" + str(i) +  ".txt", sv_print_flag=args.print_svflag)
            
        eval_loss /= 3
        if args.print_svflag:
            with open(args.save_path + "log_sv.txt", "+a") as lossSVfile:
                lossSVfile.write(f'epoch:{args.max_epochs}, MSE loss SV:{eval_loss}\n')

def train_model(k_model, args, hgraph, inc_dataset, adj_dataset, line_dataset, testflag):
    print(f'learning rate: {args.lr}')
    curr_lr = args.lr
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")  
    print(f'learning rate: {args.lr}')    
    optimizer = torch.optim.Adam(k_model.model.parameters(), lr=args.lr)
    # optimizerI = torch.optim.Adam(k_model.model.parameters(), lr=args.lr)
    # optimizerA = torch.optim.Adam(k_model.model.parameters(), lr=args.lr)
    # optimizerL = torch.optim.Adam(k_model.model.parameters(), lr=args.lr)
    if args.retrain:
        # Load paramter and optimizer
        checkpoint = torch.load(args.load_path + "_min.pt", map_location=device)
        k_model.model.load_state_dict(checkpoint['model_state_dict'])
        optimizerI.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizerI.load_state_dict(checkpoint['optimizerI_state_dict'])
        # optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
        # optimizerL.load_state_dict(checkpoint['optimizerL_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['loss']
        k_model.node_perm = checkpoint['node_perm']
        k_model.edge_perm = checkpoint['edge_perm']
    else:
        print("RETRAIN=False")
        start_epoch = 0
        min_loss = sys.float_info.max  
        
    save_cnt = 1
    epoch = 0
    for epoch in range(start_epoch, args.max_epochs):
        epoch_start_time = time.time()
        
        inc_dataset.update_perm(k_model.node_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
        adj_dataset.update_perm(k_model.node_perm.detach().cpu(), k_model.node_perm.detach().cpu())
        line_dataset.update_perm(k_model.edge_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
        inc_loader = torch.utils.data.DataLoader(dataset=inc_dataset, batch_size=1, shuffle=True, num_workers=1)
        adj_loader = torch.utils.data.DataLoader(dataset=adj_dataset, batch_size=1, shuffle=True, num_workers=1)
        line_loader = torch.utils.data.DataLoader(dataset=line_dataset, batch_size=1, shuffle=True, num_workers=1)
        
        k_model.model.eval()
        with torch.no_grad(): 
            # Calculate Loss before Perm.
            time_before_perm = 0.
            start_time = time.time()
            
            # For Debugging
            LossI, LossA, LossL = 0., 0., 0.
            LossI_sq = k_model.model.get_sq_sum("I").detach().cpu().item()
            LossA_sq = k_model.model.get_sq_sum("A").detach().cpu().item()
            LossL_sq = k_model.model.get_sq_sum("L").detach().cpu().item()
                        
            # Before Updating Loss
            for data in inc_loader:
                inputs, vals = data
                inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                inputs, vals = inputs.to(device), vals.to(device)
                outputs = k_model.model(inputs, "I")
                tmp = torch.square(outputs).sum().detach().cpu().item()
                tmp2 = torch.square(outputs - vals).sum().detach().cpu().item()
                LossI += (tmp2 - tmp)
                if testflag:
                    assert LossI_sq > tmp
            LossI += LossI_sq
            if args.norm_flag:
                LossI /= (args.init_row_k * args.init_col_k)
            
            for data in adj_loader:
                inputs, vals = data
                inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                inputs, vals = inputs.to(device), vals.to(device)
                outputs = k_model.model(inputs, "A")
                tmp = torch.square(outputs).sum().detach().cpu().item()
                tmp2 = torch.square(outputs - vals).sum().detach().cpu().item()
                LossA += (tmp2 - tmp)
                if testflag:
                    assert LossA_sq > tmp
            LossA += LossA_sq
            if args.norm_flag:
                LossA /= (args.init_row_k * args.init_row_k)
            
            for data in line_loader:
                inputs, vals = data
                inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                inputs, vals = inputs.to(device), vals.to(device)
                outputs = k_model.model(inputs, "L")
                tmp = torch.square(outputs).sum().detach().cpu().item()
                tmp2 = torch.square(outputs - vals).sum().detach().cpu().item()
                LossL += (tmp2 - tmp)
                # (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
                if testflag:
                    assert LossL_sq > tmp
            LossL += LossL_sq
            if args.norm_flag:
                LossL /= (args.init_col_k * args.init_col_k)
            
            perm_loss = args.lambdaI * LossI + args.lambdaA * LossA + args.lambdaL * LossL
            with open(args.save_path + "log.txt", '+a') as lossfile:   
                lossfile.write(f'epoch:{epoch}, loss before perm:{perm_loss}, {LossI}, {LossA}, {LossL}\n')    
            print(f'epoch:{epoch}, loss before perm:{perm_loss}, {LossI}, {LossA}, {LossL}\n')
            
            end_time = time.time()
            time_before_perm = (end_time - start_time) / 60
            with open(args.save_path + "timelog.txt", '+a') as timefile:   
                timefile.write(f'epoch:{epoch}, time before perm (min.):{time_before_perm}\n')    
            print(f'epoch:{epoch}, time before perm (min.):{time_before_perm}\n')
            
            # Sample Permutation
            time_perm = 0.
            start_time = time.time()
            for _ in range(args.perm_per_update):
                k_model.sample_node_batches(args.batch_size)
                k_model.sample_edge_batches(args.batch_size)
            end_time = time.time()
            time_perm = (end_time - start_time) / 60
            with open(args.save_path + "timelog.txt", '+a') as timefile:   
                timefile.write(f'epoch:{epoch}, time for perm (min.):{time_perm}\n')    
            print(f'epoch:{epoch}, time for perm (min.):{time_perm}\n')
        
        torch.cuda.empty_cache()
        # Update Permutation
        inc_dataset.update_perm(k_model.node_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
        adj_dataset.update_perm(k_model.node_perm.detach().cpu(), k_model.node_perm.detach().cpu())
        line_dataset.update_perm(k_model.edge_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
        inc_loader = torch.utils.data.DataLoader(dataset=inc_dataset, batch_size=1, shuffle=True, num_workers=1)
        adj_loader = torch.utils.data.DataLoader(dataset=adj_dataset, batch_size=1, shuffle=True, num_workers=1)
        line_loader = torch.utils.data.DataLoader(dataset=line_dataset, batch_size=1, shuffle=True, num_workers=1)
        
        # Save before Permutation
        prev_opt_params = copy.deepcopy(optimizer.state_dict()) 
        # prev_opt_paramsI = copy.deepcopy(optimizerI.state_dict()) 
        # prev_opt_paramsA = copy.deepcopy(optimizerA.state_dict()) 
        # prev_opt_paramsL = copy.deepcopy(optimizerL.state_dict())       
        # prev_params = copy.deepcopy(k_model.model.state_dict())
        
        # Calculate Gradient
        time_grad = 0.
        start_time = time.time()
        
        k_model.model.train()
        LossI_agg, LossA_agg, LossL_agg = 0., 0., 0.
        
        optimizer.zero_grad()
        # optimizerI.zero_grad() # .zero_grad(set_to_none=True)
        LossI = (k_model.model.get_sq_sum("I") * args.lambdaI)
        if args.norm_flag:
            LossI /= (args.init_row_k * args.init_col_k)
        LossI.backward()
        LossI_agg += LossI.detach().cpu().item()
        for data in inc_loader:
            inputs, vals = data
            inputs, vals = inputs.squeeze(0), vals.squeeze(0)
            inputs, vals = inputs.to(device), vals.to(device)
            outputs = k_model.model(inputs, "I")
            LossI = (torch.square(outputs - vals) - torch.square(outputs)).sum() 
            if args.norm_flag:
                LossI /= (args.init_row_k * args.init_col_k)
            LossI *= args.lambdaI
            LossI.backward()
            LossI_agg += LossI.detach().cpu().item()
        # optimizerI.step() 
        
        # optimizerA.zero_grad()
        LossA = k_model.model.get_sq_sum("A") * args.lambdaA
        if args.norm_flag:
            LossA /= (args.init_row_k * args.init_row_k)
        LossA.backward()
        LossA_agg += LossA.detach().cpu().item()
        for data in adj_loader:
            inputs, vals = data
            inputs, vals = inputs.squeeze(0), vals.squeeze(0)
            inputs, vals = inputs.to(device), vals.to(device)
            outputs = k_model.model(inputs, "A")
            LossA = (torch.square(outputs - vals) - torch.square(outputs)).sum() 
            if args.norm_flag:
                LossA /= (args.init_row_k * args.init_row_k)
            LossA *= args.lambdaA
            LossA.backward()
            LossA_agg += LossA.detach().cpu().item()
        # optimizerA.step() 
        
        # optimizerL.zero_grad()
        LossL = k_model.model.get_sq_sum("L") * args.lambdaL
        if args.norm_flag:
            LossL /= (args.init_col_k * args.init_col_k)  
        LossL.backward()
        LossL_agg += LossL.detach().cpu().item()
        for data in line_loader:
            inputs, vals = data
            inputs, vals = inputs.squeeze(0), vals.squeeze(0)
            inputs, vals = inputs.to(device), vals.to(device)
            outputs = k_model.model(inputs, "L")
            LossL = (torch.square(outputs - vals) - torch.square(outputs)).sum()
            if args.norm_flag:
                LossL /= (args.init_col_k * args.init_col_k)  
            LossL *= args.lambdaL
            LossL.backward()
            LossL_agg += LossL.detach().cpu().item()
        # optimizerL.step()
        optimizer.step()
        
        end_time = time.time()
        time_grad = (end_time - start_time) / 60
        with open(args.save_path + "timelog.txt", '+a') as timefile:   
            timefile.write(f'epoch:{epoch}, time for gradient (min.):{time_grad}\n')    
        print(f'epoch:{epoch}, time for gradient (min.):{time_grad}\n')
        
        with open(args.save_path + "log.txt", '+a') as lossfile:   
            lossfile.write(f'epoch:{epoch}, loss after perm:{perm_loss}, {LossI_agg}, {LossA_agg}, {LossL_agg}\n')    
        print(f'epoch:{epoch}, loss after perm:{perm_loss}, {LossI_agg}, {LossA_agg}, {LossL_agg}\n')
        
        # After directly update model
        time_after_update = 0.
        start_time = time.time()
        
        k_model.model.eval()
        with torch.no_grad():            
            model_lossI, model_lossA, model_lossL = 0., 0., 0.
            model_lossI += k_model.model.get_sq_sum("I").detach().cpu().item()
            for data in inc_loader:
                inputs, vals = data
                inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                inputs, vals = inputs.to(device), vals.to(device)
                outputs = k_model.model(inputs, "I")
                model_lossI += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
            if args.norm_flag:
                model_lossI /= (args.init_row_k * args.init_col_k)
            model_lossA += k_model.model.get_sq_sum("A").detach().cpu().item()
            for data in adj_loader:
                inputs, vals = data
                inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                inputs, vals = inputs.to(device), vals.to(device)
                outputs = k_model.model(inputs, "A")
                model_lossA += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
            if args.norm_flag:
                model_lossA /= (args.init_row_k * args.init_row_k)
            model_lossL += k_model.model.get_sq_sum("L").detach().cpu().item()
            for data in line_loader:
                inputs, vals = data
                inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                inputs, vals = inputs.to(device), vals.to(device)
                outputs = k_model.model(inputs, "L")
                model_lossL += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
            if args.norm_flag:
                model_lossL /= (args.init_col_k * args.init_col_k)
            model_loss = args.lambdaI * model_lossI + args.lambdaA * model_lossA + args.lambdaL * model_lossL
            
        end_time = time.time()
        time_after_update = (end_time - start_time) / 60
        with open(args.save_path + "timelog.txt", '+a') as timefile:   
            timefile.write(f'epoch:{epoch}, time after update (min.):{time_after_update}\n')    
        print(f'epoch:{epoch}, time after update (min.):{time_after_update}\n')
            
        # Loss increased a lot     
        time_for_update = 0.
        start_time = time.time()
        
        _cnt = 0
        while epoch > 0 and model_loss > 1.1*perm_loss:
            _cnt += 1
            # Restore parameters                        
            k_model.model.load_state_dict(prev_params)
            optimizer.load_state_dict(prev_opt_params)
            # optimizerI.load_state_dict(prev_opt_paramsI)
            # optimizerA.load_state_dict(prev_opt_paramsA)
            # optimizerL.load_state_dict(prev_opt_paramsL)
            
            # Update parameter                            
            for i, param_group in enumerate(optimizer.param_groups):                
                param_group['lr'] = 0.1 * float(param_group['lr'])
#             for i, param_group in enumerate(optimizerI.param_groups):                
#                 param_group['lr'] = 0.1 * float(param_group['lr'])
#             for i, param_group in enumerate(optimizerA.param_groups):                
#                 param_group['lr'] = 0.1 * float(param_group['lr'])
#             for i, param_group in enumerate(optimizerL.param_groups):                
#                 param_group['lr'] = 0.1 * float(param_group['lr'])
            optimizer.step()   
            # optimizerI.step()        
            # optimizerA.step()        
            # optimizerL.step()                
            
            # re-Evaluate
            k_model.model.eval()
            with torch.no_grad():            
                model_lossI, model_lossA, model_lossL = 0., 0., 0.
                model_lossI += k_model.model.get_sq_sum("I").detach().cpu().item()
                for data in inc_loader:
                    inputs, vals = data
                    inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                    inputs, vals = inputs.to(device), vals.to(device)
                    outputs = k_model.model(inputs, "I")
                    model_lossI += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
                if args.norm_flag:
                    model_lossI /= (args.init_row_k * args.init_col_k)
                model_lossA += k_model.model.get_sq_sum("A").detach().cpu().item()
                for data in adj_loader:
                    inputs, vals = data
                    inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                    inputs, vals = inputs.to(device), vals.to(device)
                    outputs = k_model.model(inputs, "A")
                    model_lossA += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
                if args.norm_flag:
                    model_lossA /= (args.init_row_k * args.init_row_k)
                model_lossL += k_model.model.get_sq_sum("L").detach().cpu().item()
                for data in line_loader:
                    inputs, vals = data
                    print(inputs.shape)
                    inputs, vals = inputs.squeeze(0), vals.squeeze(0)
                    inputs, vals = inputs.to(device), vals.to(device)
                    outputs = k_model.model(inputs, "L")
                    model_lossL += (torch.square(outputs - vals) - torch.square(outputs)).sum().detach().cpu().item()
                if args.norm_flag:
                    model_lossL /= (args.init_col_k * args.init_col_k)
                model_loss = args.lambdaI * model_lossI + args.lambdaA * model_lossA + args.lambdaL * model_lossL            
        
            if model_loss < 1.1*perm_loss or _cnt == 10:
                for i, param_group in enumerate(optimizer.param_groups):                
                    param_group['lr'] = args.lr
#                 for i, param_group in enumerate(optimizerI.param_groups):                
#                     param_group['lr'] = args.lr
#                 for i, param_group in enumerate(optimizerA.param_groups):                
#                     param_group['lr'] = args.lr
#                 for i, param_group in enumerate(optimizerL.param_groups):                
#                     param_group['lr'] = args.lr
                break
                
        end_time = time.time()
        time_for_update = (end_time - start_time) / 60
        with open(args.save_path + "timelog.txt", '+a') as timefile:   
            timefile.write(f'epoch:{epoch}, time for update (min.):{time_for_update}\n')    
        print(f'epoch:{epoch}, time for update (min.):{time_for_update}\n')
                                        
        time_per_epoch = time.time() - epoch_start_time
        with open(args.save_path + "log.txt", '+a') as lossfile:                
            lossfile.write(f'epoch:{epoch}, loss after model update:{model_loss}, {model_lossI}, {model_lossA}, {model_lossL}, time per epoch: {time_per_epoch}\n')
        print(f'epoch:{epoch}, loss after model update:{model_loss}, {model_lossI}, {model_lossA}, {model_lossL}, time after model: {time_per_epoch}\n')

        if min_loss > model_loss:
            min_loss = model_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': k_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
#                 'optimizerI_state_dict': optimizerI.state_dict(),
#                 'optimizerA_state_dict': optimizerA.state_dict(),
#                 'optimizerL_state_dict': optimizerL.state_dict(),
                'lossI': model_lossI,
                'lossA': model_lossA,
                'lossL': model_lossL,
                'loss': model_loss,
                'node_perm': k_model.node_perm,
                'edge_perm': k_model.edge_perm
            }, args.save_path + "min.pt") 
            
            if (epoch > (args.save_epoch * save_cnt)) or (epoch == (args.max_epochs - 1)):
                start_time = time.time()
                
                k_model.model.eval()
                with torch.no_grad():
                    for i in range(1):
                        eval_loss = k_model.write_matrix(args.batch_size, args.save_path + "sampled_" + str(i) +  ".txt", sv_print_flag=args.print_svflag)
                
                end_time = time.time()
                time_write = (end_time - start_time) / 60
                print(f'epoch:{epoch}, time for writing (min.):{time_write}\n')
                save_cnt += 1
                
                if args.print_svflag:
                    with open(args.save_path + "log_sv.txt", "+a") as lossSVfile:
                        lossSVfile.write(f'epoch:{epoch}, MSE loss SV:{eval_loss}\n')

def load_model(k_model, args, device):
    # Load model
    use_cuda = torch.cuda.is_available()
    i_device = torch.device("cuda:" + str(device[0]) if use_cuda else "cpu")  
    checkpoint = torch.load(args.load_path + ".pt", map_location=i_device) # , map_location=args.device
    k_model.model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    k_model.node_perm = checkpoint['node_perm']
    k_model.edge_perm = checkpoint['edge_perm']
    k_model.model.eval()
    
if __name__ == '__main__':
    torch.set_num_threads(4)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train, eval')
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-td", "--test_data", action="store", default="none", type=str)
    parser.add_argument("-p", "--perm_file", action="store", default=False, type=bool)
    parser.add_argument("-ip", "--init_file", action="store", default="", type=str)
    parser.add_argument("-r", "--init_row", action="store", default=0, type=int)
    parser.add_argument("-c", "--init_col", action="store", default=0, type=int)
    parser.add_argument("-de", "--device", action="store", nargs='+', type=int)
    # parser.add_argument("-de", "--device", action="store", type=int)
    
    parser.add_argument("-b", "--batch_size", action="store", default=2**25, type=int)
    parser.add_argument("-e", "--max_epochs", action="store", default=5000, type=int) # 10**5, 5000
    parser.add_argument("-se", "--save_epoch", action="store", default=500, type=int) # 10**5, 5000
    parser.add_argument("-lr", "--lr", action="store", default=1e-1, type=float)
    parser.add_argument("-ppu", "--perm_per_update", action="store", default=2, type=int)
    parser.add_argument("-sw", "--sample_weight", action="store", default=10, type=float)
    parser.add_argument("-ldI", "--lambdaI", action="store", default=1.0, type=float)
    parser.add_argument("-ldA", "--lambdaA", action="store", default=1.0, type=float)
    parser.add_argument("-ldL", "--lambdaL", action="store", default=1.0, type=float)
    parser.add_argument("-np", "--num_param", action="store", default=100, type=int)
    
    parser.add_argument("-dt", "--data_type", action="store", default="double", type=str)
    parser.add_argument("--saveflag", action="store_true")
    parser.add_argument("--norm_flag", action="store_true")
    parser.add_argument("--print_svflag", action="store_true")
    
    parser.add_argument("-lp", "--load_path", action="store", default="", type=str)
    parser.add_argument("-sp", "--save_path", action="store", default="", type=str)
    parser.add_argument("-rt", "--retrain", action="store", default=False, type=bool)
    parser.add_argument("-perm", "--load_perm", action="store", default="False", type=str)
    parser.add_argument("-tst", "--test", action="store_true")
    args = parser.parse_args()
    
    lambdasum = args.lambdaI + args.lambdaA + args.lambdaL
    args.lambdaI /= lambdasum
    args.lambdaA /= lambdasum
    args.lambdaL /= lambdasum
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device[0]) if use_cuda else "cpu")  
        
          
    # Load graph    
    data_file = "../kronfit_neukron/input/" + args.dataset + ".txt"    
    hgraph = hyperGraph(data_file, args.batch_size, svflag=False, saveflag=args.saveflag)
    if args.print_svflag:
        hgraph.calculate_sv()
    print(f'rows: {hgraph.num_row}, columns:{hgraph.num_col}, nnz:{hgraph.real_num_nonzero}')

    if args.init_row + args.init_col == 0:
        diff = 1e+12
        for _k in range(3, 10):
            init_row = math.ceil(hgraph.num_row ** (1./_k))
            init_col = math.ceil(hgraph.num_col ** (1./_k))
            diff_row = (init_row ** _k) - hgraph.num_row
            diff_col = (init_col ** _k) - hgraph.num_col
            assert diff_row >= 0.0
            assert diff_col >= 0.0
            if (diff_row + diff_col < diff) and (init_row * init_col < args.num_param):
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
        if len(args.init_file) > 0:
            args.save_path = f"./result_batch_init/{args.dataset}/"
        else:
            args.save_path = f"./result_batch/{args.dataset}/"
        args.save_path += "%.2f_%.2f_%.2f" % (args.lambdaI, args.lambdaA, args.lambdaL) + "/"
        args.save_path += f"{args.init_row}_{args.init_col}_{args.k}_" + "%.3f_" % (args.lr)
        if args.norm_flag:
            args.save_path += "1/"
        else:
            args.save_path += "0/"
        args.load_path = args.save_path
        
        if os.path.isdir(args.save_path) is False:
            os.makedirs(args.save_path)
        elif os.path.isfile(args.save_path + "log.txt"):
            os.remove(args.save_path + "log.txt")
        elif os.path.isfile(args.save_path + "timelog.txt"):
            os.remove(args.save_path + "timelog.txt")
        if args.print_svflag and os.path.isfile(args.save_path + "log_sv.txt"):
            os.remove(args.save_path + "log_sv.txt")       
    
    # add agraph, lgraph
    k_model = KroneckerFitting(hgraph, args.init_row, args.init_col, args.k, args.device, args.sample_weight, args.saveflag, args.test) 
        
    # input_size, input_size_sec = args.init_row * args.init_col, args.init_col
    # k_model.init_model(args.hidden_size, args.model, args.data_type)
    k_model.init_model()
    
    if args.load_perm == "True":
        row_perm_file, col_perm_file = "../data/" + args.dataset + "_row.txt", "../data/" + args.dataset + "_col.txt" 
        k_model.set_permutation(row_perm_file, col_perm_file)
    else:
        k_model.init_permutation()
        
    # Load Initialize Models
    if len(args.init_file) > 0:
        if os.path.isfile(args.init_file) is False:
            sys.exit("No exist file")
        checkpoint = torch.load(args.init_file, map_location=device)
        # k_model.model.load_state_dict(checkpoint['model_state_dict'])
        k_model.model.sos.data = checkpoint["model_state_dict"]["sos"]
        # k_model.node_perm = checkpoint['node_perm']
        # k_model.edge_perm = checkpoint['edge_perm']
    
    # load data
    incdata = GraphDataset(hgraph.row_idx, hgraph.col_idx, hgraph.val, args.batch_size, False, k_model.node_bases.detach().cpu(), k_model.edge_bases.detach().cpu(), k_model.init_row, k_model.init_col, k_model.node_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
    if args.saveflag:
        adjdata = GraphDataset(hgraph.adj_row_idx, hgraph.adj_col_idx, hgraph.savename + "adj", args.batch_size, True, k_model.node_bases.detach().cpu(), k_model.node_bases.detach().cpu(), k_model.init_row, k_model.init_row, k_model.node_perm.detach().cpu(), k_model.node_perm.detach().cpu())
        linedata = GraphDataset(hgraph.line_row_idx, hgraph.line_col_idx, hgraph.savename + "line", args.batch_size, True, k_model.edge_bases.detach().cpu(), k_model.edge_bases.detach().cpu(), k_model.init_col, k_model.init_col, k_model.edge_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
    else:
        adjdata = GraphDataset(hgraph.adj_row_idx, hgraph.adj_col_idx, hgraph.adj_vals, args.batch_size, False, k_model.node_bases.detach().cpu(), k_model.node_bases.detach().cpu(), k_model.init_row, k_model.init_row, k_model.node_perm.detach().cpu(), k_model.node_perm.detach().cpu())
        linedata = GraphDataset(hgraph.line_row_idx, hgraph.line_col_idx, hgraph.line_vals, args.batch_size, False, k_model.edge_bases.detach().cpu(), k_model.edge_bases.detach().cpu(), k_model.init_col, k_model.init_col, k_model.edge_perm.detach().cpu(), k_model.edge_perm.detach().cpu())
    
        # test loss
    if args.action == 'train':
        train_model(k_model, args, hgraph, incdata, adjdata, linedata, args.test) 
        eval_loss(k_model, args, incdata, adjdata, linedata) 
    elif args.action == 'eval':
        eval_loss(k_model, args, incdata, adjdata, linedata) 
    else:
        assert(False)