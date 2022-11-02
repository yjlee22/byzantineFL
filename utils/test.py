import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

def test_img(net_g, dataset, args):
    # testing
    correct = 0
    data_loader = DataLoader(dataset, batch_size=128)
    test_loss = 0
    with torch.no_grad():
        net_g.eval()
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target.squeeze(dim=-1), reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        test_acc = 100.00 * correct / len(data_loader.dataset)

        return test_acc, test_loss

