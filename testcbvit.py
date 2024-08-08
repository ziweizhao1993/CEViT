import torch
def test_cbvit(cbvit, device, test_cls_loader, K, N):
    cbvit.eval()

    correct = 0
    count = 0
    
    #randomly select 
    references = [[] for _ in range(10)]
    for img_batch, label_batch in test_cls_loader:
        for i in range(img_batch.size(0)):
            references[label_batch[i]].append(img_batch[i:i+1])
        if min(len(d) for d in references) >= N:
            break;

    for img, targets in test_cls_loader:
        img, targets = img.to(device), targets.to(device)

        outputs = []
        labels = []
        for i in range(10): #classes
            for j in range(N): #25*10 samples
                inputs = torch.cat([img, torch.stack([references[i][j][0].to(device)]*1000)],axis=1)
                inputs = inputs.to(device)
                with torch.no_grad():
                    outputs.append(cbvit(inputs)[0].squeeze())
                labels.append(i)
        outputs = torch.stack(outputs)
        labels = torch.tensor(labels).to(device)
        indices = torch.topk(outputs,K,dim=0).indices #K=10
        labels_topk = labels[indices]
        predictions = torch.mode(labels_topk,dim=0).values
        correct += torch.sum(predictions==targets)
        count += targets.size(0)

    return (correct/count)