import torch
import numpy as np


# Analyze the cls embeddings in a trasformer model.
def analyze_cls(model, device, test_loader, contrastive = False):
    cls_list = []
    targets_list = []
    prediction_list = []
    with torch.no_grad():
        model.eval()
        if not contrastive:
            for images, targets in test_loader:
                outputs, feat = model(images.to(device), return_features=True)
                pred = torch.cat(outputs, dim=1).argmax(1)
                #pred = pred.cpu().numpy()
                cls = feat.cpu().numpy()
                for i in range(len(feat)):
                    cls_list.append(cls[i].reshape((-1)))
                    targets_list.append(targets[i])
                    #prediction_list.append(pred[i])
        else:
            for x1, x2, targets in test_loader:
                out1, feat = model(x1.to(device), return_features=True)

                #pred = torch.cat(outputs, dim=1).argmax(1)
                #pred = pred.cpu().numpy()
                cls = feat.cpu().numpy()
                targets = targets.cpu().numpy()
                for i in range(len(feat)):
                    cls_list.append(cls[i].reshape((-1)))
                    targets_list.append(targets[i])
                    #prediction_list.append(pred[i])

    cls_list = np.asarray(cls_list)
    targets_list = np.asarray(targets_list)
    #prediction_list = np.asarray(prediction_list)
    return cls_list, targets_list#, prediction_list

def analyze_prototypes(appr, n_classes):
    prototype_list = []
    prototype = appr.prototypes[:n_classes].clone().detach().cpu().numpy()
    for item in prototype:
        prototype_list.append(item.reshape((-1)))
    
    prototype_list = np.asarray(prototype_list)

    return prototype_list
    
def analyze_heads(model):
    weights_list = []
    heads = model.heads
    for item in heads:
        weights_list.append(item.weight.reshape((-1)))
    
    weights_list = np.asarray(weights_list)

    return weights_list
