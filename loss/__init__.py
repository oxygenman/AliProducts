from .label_smooth import LabelSmoothing
from .arc import ArcLoss
from torch import nn
from options import opt

criterionCE = nn.CrossEntropyLoss()
arcLoss = ArcLoss()
if opt.smooth != 0:
    label_smooth_loss = LabelSmoothing(smoothing=opt.smooth)
else:
    label_smooth_loss = 0.

def get_loss(predicted, label, avg_meters, *args):
    ce_loss = criterionCE(predicted, label)

    assert opt.smooth < 1, 'smooth should be smaller than 1.0'

    if opt.smooth == 0:  # 不用label smooth
        avg_meters.update({'CE loss': ce_loss.item()})
        return ce_loss
    else:
        smooth_loss = label_smooth_loss(predicted, label) 
        avg_meters.update({f'CE loss(smooth={opt.smooth})': smooth_loss.item()})
        return smooth_loss
def get_arc_loss(predicted, label, avg_meters):
    arc_loss,acc= arcLoss(predicted, label)
    avg_meters.update({'arc loss':arc_loss.item()})
    return arc_loss,acc
