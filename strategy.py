# -*- coding: utf-8 -*-
import numpy as np
import math
import os
import sys
import time
import torch
import cv2
import matplotlib.pyplot as plt

from models.mt import MTmodel
from utils.loss import ComputeLoss


def rename_attribute(a, b, obj):
    setattr(obj, a, getattr(obj, b))
    setattr(obj, b, '')


def change_head(task: str, open_head: bool, model, params):
    a = task + '_decoder_tmp'
    b = task + '_decoder'
    if open_head:
        a, b = b, a
    c, d = a.replace('decoder', 'head'), b.replace('decoder', 'head')
    if task == 'object_detection':
        c, d = c.replace('object_detection', 'obj'), d.replace('object_detection', 'obj')
    if hasattr(model, b):
        model._modules[a] = model._modules.pop(b)
        rename_attribute(c, d, model)
        rename_attribute(c, d, params)


def Strategy1(params, epoch, epochs, model):
    part = int(epochs / 10)
    origin_type = type(model)
    model = model.module if type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel) else model
    if part >= epoch > params.warmup:
        # Only Train Depth
        if hasattr(model, 'semantic_decoder'):
            rename_attribute('semantic_head_tmp', 'semantic_head', model)
            rename_attribute('semantic_head_tmp', 'semantic_head', params)
            model._modules['semantic_decoder_tmp'] = model._modules.pop('semantic_decoder')
        if hasattr(model, 'object_detection_decoder'):
            rename_attribute('obj_head_tmp', 'obj_head', model)
            rename_attribute('obj_head_tmp', 'obj_head', params)
            model._modules['object_detection_decoder_tmp'] = model._modules.pop('object_detection_decoder')
    elif part * 2 >= epoch > part:
        # Only Train Smnt
        if hasattr(model, 'depth_decoder'):
            rename_attribute('depth_head_tmp', 'depth_head', model)
            rename_attribute('depth_head_tmp', 'depth_head', params)
            model._modules['depth_decoder_tmp'] = model._modules.pop('depth_decoder')
        if hasattr(model, 'semantic_decoder_tmp'):
            rename_attribute('semantic_head', 'semantic_head_tmp', model)
            rename_attribute('semantic_head', 'semantic_head_tmp', params)
            model._modules['semantic_decoder'] = model._modules.pop('semantic_decoder_tmp')
    elif part * 3 >= epoch > part * 2:
        # Only Train Obj
        if hasattr(model, 'object_detection_decoder_tmp'):
            rename_attribute('obj_head', 'obj_head_tmp', model)
            rename_attribute('obj_head', 'obj_head_tmp', params)
            model._modules['object_detection_decoder'] = model._modules.pop('object_detection_decoder_tmp')
        if hasattr(model, 'semantic_decoder'):
            rename_attribute('semantic_head_tmp', 'semantic_head', model)
            rename_attribute('semantic_head_tmp', 'semantic_head', params)
            model._modules['semantic_decoder_tmp'] = model._modules.pop('semantic_decoder')
    elif epoch > part * 3:
        if hasattr(model, 'depth_decoder_tmp'):
            rename_attribute('depth_head', 'depth_head_tmp', model)
            rename_attribute('depth_head', 'depth_head_tmp', params)
            model._modules['depth_decoder'] = model._modules.pop('depth_decoder_tmp')
        if hasattr(model, 'semantic_decoder_tmp'):
            rename_attribute('semantic_head', 'semantic_head_tmp', model)
            rename_attribute('semantic_head', 'semantic_head_tmp', params)
            model._modules['semantic_decoder'] = model._modules.pop('semantic_decoder_tmp')
        if hasattr(model, 'object_detection_decoder_tmp'):
            rename_attribute('obj_head', 'obj_head_tmp', model)
            rename_attribute('obj_head', 'obj_head_tmp', params)
            model._modules['object_detection_decoder'] = model._modules.pop('object_detection_decoder_tmp')
    compute_loss = ComputeLoss(model)
    if origin_type == torch.nn.parallel.DataParallel:
        model = torch.nn.DataParallel(model)
    elif origin_type == torch.nn.parallel.DistributedDataParallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    return model, compute_loss


def Strategy2(params, epoch, epochs, model):
    part = int(epochs / 10)
    origin_type = type(model)
    model = model.module if type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel) else model
    if part >= epoch > params.warmup:
        # Only Train Depth
        if hasattr(model, 'semantic_decoder'):
            rename_attribute('semantic_head_tmp', 'semantic_head', model)
            rename_attribute('semantic_head_tmp', 'semantic_head', params)
            model._modules['semantic_decoder_tmp'] = model._modules.pop('semantic_decoder')
        if hasattr(model, 'object_detection_decoder'):
            rename_attribute('obj_head_tmp', 'obj_head', model)
            rename_attribute('obj_head_tmp', 'obj_head', params)
            model._modules['object_detection_decoder_tmp'] = model._modules.pop('object_detection_decoder')
    elif part * 2 >= epoch > part:
        # Train Depth and Smnt
        if hasattr(model, 'semantic_decoder_tmp'):
            rename_attribute('semantic_head', 'semantic_head_tmp', model)
            rename_attribute('semantic_head', 'semantic_head_tmp', params)
            model._modules['semantic_decoder'] = model._modules.pop('semantic_decoder_tmp')
    elif part * 3 >= epoch > part * 2:
        # Train Depth, Smnt and Obj
        if hasattr(model, 'object_detection_decoder_tmp'):
            rename_attribute('obj_head', 'obj_head_tmp', model)
            rename_attribute('obj_head', 'obj_head_tmp', params)
            model._modules['object_detection_decoder'] = model._modules.pop('object_detection_decoder_tmp')
    elif epoch > part * 3:
        if hasattr(model, 'depth_decoder_tmp'):
            rename_attribute('depth_head', 'depth_head_tmp', model)
            rename_attribute('depth_head', 'depth_head_tmp', params)
            model._modules['depth_decoder'] = model._modules.pop('depth_decoder_tmp')
        if hasattr(model, 'semantic_decoder_tmp'):
            rename_attribute('semantic_head', 'semantic_head_tmp', model)
            rename_attribute('semantic_head', 'semantic_head_tmp', params)
            model._modules['semantic_decoder'] = model._modules.pop('semantic_decoder_tmp')
        if hasattr(model, 'object_detection_decoder_tmp'):
            rename_attribute('obj_head', 'obj_head_tmp', model)
            rename_attribute('obj_head', 'obj_head_tmp', params)
            model._modules['object_detection_decoder'] = model._modules.pop('object_detection_decoder_tmp')
    compute_loss = ComputeLoss(model)
    if origin_type == torch.nn.parallel.DataParallel:
        model = torch.nn.DataParallel(model)
    elif origin_type == torch.nn.parallel.DistributedDataParallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    return model, compute_loss


def Strategy3(params, epoch, epochs, model):
    origin_type = type(model)
    model = model.module if type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel) else model
    if int(epochs * 0.7) >= epoch > int(epochs * 0.5):
        # Only Train Smnt
        change_head('depth', False, model, params)
        change_head('object_detection', False, model, params)
        change_head('semantic', True, model, params)        
    elif int(epochs * 0.9) >= epoch > int(epochs * 0.7):
        # Only Train Obj
        change_head('depth', False, model, params)
        change_head('object_detection', True, model, params)
        change_head('semantic', False, model, params)   
    elif epoch > int(epochs * 0.9):
        # Train All
        change_head('depth', True, model, params)
        change_head('object_detection', True, model, params)
        change_head('semantic', True, model, params)   
    compute_loss = ComputeLoss(model)
    if origin_type == torch.nn.parallel.DataParallel:
        model = torch.nn.DataParallel(model)
    elif origin_type == torch.nn.parallel.DistributedDataParallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    return model, compute_loss