# Copyright (c) Zhisheng Zheng, The University of Texas at Austin.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Audio-MAE: https://github.com/facebookresearch/AudioMAE
# --------------------------------------------------------
import os
import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.stat import calculate_stats, concat_all_gather

from seld.cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from seld.SELD_evaluation_metrics import distance_between_cartesian_coordinates

def train_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None, log_writer=None, args=None
    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    nb_train_batches = 0
    loss_value = 0

    for data_iter_step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)

        waveforms, reverbs = batch[0], batch[1]
        targets, spaital_targets, adpit_labels = batch[2], batch[3], batch[-1]

        targets = targets.to(device, non_blocking=True)
        distance = spaital_targets['distance'].long().to(device, non_blocking=True)
        azimuth = spaital_targets['azimuth'].long().to(device, non_blocking=True)
        elevation = spaital_targets['elevation'].long().to(device, non_blocking=True)
        
        adpit_labels = adpit_labels.float().to(device, non_blocking=True)
        # with torch.cuda.amp.autocast():
        outputs = model(waveforms, reverbs, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)

        outputs_0 = outputs[0].unsqueeze(-1)#.view(-1, 15, 1)
        outputs_1 = outputs[1].unsqueeze(-1)#.view(-1, 21, 1)
        outputs_2 = outputs[2].unsqueeze(-1)#.view(-1, 360, 1)
        outputs_3 = outputs[3].unsqueeze(-1)#.view(-1, 180, 1)

        adpit_preds = outputs[-1]
        
        targets_rs = targets#.view(-1, 15, 1)

        #loss_t1 = criterion(outputs_0[..., 0], targets_rs[..., 0])
        #loss_t2 = criterion(outputs_0[..., 1], targets_rs[..., 1])
        
        #loss_d1 = F.cross_entropy(outputs_1[..., 0], distance[:, 0])
        #loss_a1 = F.cross_entropy(outputs_2[..., 0], azimuth[:, 0])
        #loss_e1 = F.cross_entropy(outputs_3[..., 0], elevation[:, 0])
        
        #loss_d2 = F.cross_entropy(outputs_1[..., 1], distance[:, 1])
        #loss_a2 = F.cross_entropy(outputs_2[..., 1], azimuth[:, 1])
        #loss_e2 = F.cross_entropy(outputs_3[..., 1], elevation[:, 1])


        # loss = loss1
        #print("shapes prior to adpit loss", adpit_preds.unsqueeze(1).shape, adpit_labels[:, :1, :, :].shape)
        
        loss = criterion(adpit_preds.unsqueeze(1), adpit_labels[:, :1, :, :]) # TODO: for now using :1 as we have frames per 100ms
        loss.backward()
        optimizer.step()
        #1250 * (loss_t1 + 1 + loss_t2 + 1) * (loss_d1 + loss_d2) + 2 * ((loss_a1 + loss_a2) + (loss_e1 + loss_e2))
            
        loss_value += loss.item()
        nb_train_batches += 1

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        #loss /= accum_iter
        #loss_scaler(loss, optimizer, clip_grad=max_norm,
        #            parameters=model.parameters(), create_graph=False,
        #            update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    loss_value /= nb_train_batches
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, dist_eval=False):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs = []
    targets = []
    vids = []

    # Source 1 metrics
    all_distance_preds_1 = []
    all_distances_1 = []
    doa_dists_1 = []

    # Source 2 metrics
    all_distance_preds_2 = []
    all_distances_2 = []
    doa_dists_2 = []

    for batch in metric_logger.log_every(data_loader, 300, header):

        waveforms, reverbs = batch[0], batch[1]
        target, spaital_targets, adpit_labels = batch[2], batch[3], batch[-1]
        
        adpit_labels = adpit_labels[:, :1, :, :] # using :1 as we have 100ms audio frames

        batch_file_list = batch[4]

        target = target.to(device, non_blocking=True)
        # compute output

        output = model(waveforms, reverbs)
        # remark: 
        # 1. use concat_all_gather and --dist_eval for faster eval by distributed load over gpus
        # 2. otherwise comment concat_all_gather and remove --dist_eval one every gpu
        if dist_eval:
            cls_output = concat_all_gather(output[0].detach())
            target = concat_all_gather(target)
        outputs.append(cls_output)
        targets.append(target)

        #print(output[1].shape, output[2].shape, output[3].shape)
        outputs_1 = output[1].unsqueeze(-1)#.view(-1, 21, 1)
        outputs_2 = output[2].unsqueeze(-1)#.view(-1, 360, 1)
        outputs_3 = output[3].unsqueeze(-1)#.view(-1, 180, 1)

        adpit_preds = output[-1]

        all_distances_1.append(spaital_targets['distance'][:, 0].numpy())
        all_distance_preds_1.append(torch.argmax(outputs_1[..., 0], dim=1).detach().cpu().numpy())
        
        az_pred = torch.argmax(outputs_2[..., 0], dim=1).detach().cpu().numpy()
        ele_pred = torch.argmax(outputs_3[..., 0], dim=1).detach().cpu().numpy()
        az_gt = spaital_targets['azimuth'][:, 0].long().numpy()
        ele_gt = spaital_targets['elevation'][:, 0].long().numpy()
        doa_dist = distance_between_spherical_coordinates_rad(az_gt, ele_gt, az_pred, ele_pred)
        doa_dists_1.append(doa_dist)
        
        ### Compute SELD metrics ###
        sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(adpit_preds.unsqueeze(1).detach().cpu().numpy(), 15)
        sed_pred0 = reshape_3Dto2D(sed_pred0)
        doa_pred0 = reshape_3Dto2D(doa_pred0)
        sed_pred1 = reshape_3Dto2D(sed_pred1)
        doa_pred1 = reshape_3Dto2D(doa_pred1)
        sed_pred2 = reshape_3Dto2D(sed_pred2)
        doa_pred2 = reshape_3Dto2D(doa_pred2)
        
        seld_preds = (sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2)
        write_SELD_results(seld_preds, batch_file_list)

    outputs = torch.cat(outputs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    vids = [j for sub in vids for j in sub]
    # np.save('inf_output.npy', {'vids':vids, 'embs_527':outputs, 'targets':targets})

    stats = calculate_stats(outputs, targets)

    # AP = [stat['AP'] for stat in stats]
    mAP = np.mean([stat['AP'] for stat in stats])
    print("mAP: {:.6f}".format(mAP))

    all_distance_preds_1 = np.concatenate(all_distance_preds_1)
    all_distances_1 = np.concatenate(all_distances_1)
    doa_dists_1 = np.concatenate(doa_dists_1)

    total_samples = len(all_distances_1)
    spatial_outputs = []

    distance_correct = np.sum([1 for truth, pred in zip(all_distances_1, all_distance_preds_1) if abs(truth - pred) <= 1])
    spatial_outputs.append(distance_correct)

    threshold = 20
    doa_angular_error = np.sum(doa_dists_1)
    doa_error = np.sum(doa_dists_1 > threshold) # 
    spatial_outputs.append(doa_error)
    spatial_outputs.append(doa_angular_error)

    if dist_eval:        
        spatial_outputs = torch.tensor(spatial_outputs).to(device)
        torch.distributed.all_reduce(spatial_outputs, op=torch.distributed.ReduceOp.SUM)
        
        total_samples = torch.tensor(total_samples).to(device)
        torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
        
        spatial_outputs = spatial_outputs.cpu().numpy()
        total_samples = total_samples.cpu().numpy()

    return {
        "mAP": mAP,
        "distance_accuracy": spatial_outputs[0] / total_samples,
        "doa_error": spatial_outputs[1] / total_samples,
        "doa_angular_error": spatial_outputs[2] / total_samples
    }


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5
      
    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 6*nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def write_SELD_results(seld_preds, batch_file_list):
    sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = seld_preds
    # dump SELD results to the correspondin file
    # Hardcoding the following params as we dont load the parameters.py yet
    params = {}
    params['thresh_unify'] = 20
    params['unique_classes'] = 15
    for frame_cnt, filepath in enumerate(batch_file_list): # for each file within the batch
        filename = os.path.basename(filepath)
        filename = filename.replace(".wav", ".csv")
        output_file = os.path.join("./test_seld_output", filename)
        output_dict = {}
        for class_cnt in range(sed_pred0.shape[1]):
            # determine whether track0 is similar to track1
            flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
            flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
            flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
            # unify or not unify according to flag
            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                #print("values being recorded here", sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt])
                if sed_pred0[frame_cnt][class_cnt]>0.5:
                    if frame_cnt not in output_dict:
                        output_dict[0] = []
                    output_dict[0].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]])
                if sed_pred1[frame_cnt][class_cnt]>0.5:
                    if frame_cnt not in output_dict:
                        output_dict[0] = []
                    output_dict[0].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]])
                if sed_pred2[frame_cnt][class_cnt]>0.5:
                    if frame_cnt not in output_dict:
                        output_dict[0] = []
                    output_dict[0].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]])
            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                if frame_cnt not in output_dict:
                    output_dict[0] = []
                if flag_0sim1:
                    if sed_pred2[frame_cnt][class_cnt]>0.5:
                        output_dict[0].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]])
                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                    output_dict[0].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                elif flag_1sim2:
                    if sed_pred0[frame_cnt][class_cnt]>0.5:
                        output_dict[0].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]])
                    doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                    output_dict[0].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                elif flag_2sim0:
                    if sed_pred1[frame_cnt][class_cnt]>0.5:
                        output_dict[0].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]])
                    doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                    output_dict[0].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                if frame_cnt not in output_dict:
                    output_dict[0] = []
                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                output_dict[0].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
        
        write_output_format_file(output_file, output_dict)


def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
            _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), 0))
    _fid.close()

def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    #NOTE: [0, 180] --> [0, +180]; [+180, +360] --> [-180, 0]
    az1[az1 > 180] -= 360
    az2[az2 > 180] -= 360
    az1 = az1 * np.pi / 180.
    az2 = az2 * np.pi / 180.
    ele1 = (ele1 - 90) * np.pi / 180.
    ele2 = (ele2 - 90) * np.pi / 180.

    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist
