import satmm_cuda
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from functools import reduce

__all__ = ['birealnet18', 'birealnet34']

class satmm_psum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X, t):
        ctx.t = t
        out = satmm_cuda.forward_psum(A, X, t)
        ctx.save_for_backward(A, X)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.sum(axis=-1) / grad_output.shape[-1]
        A, X = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, X.T)
        grad_weight = torch.matmul(A.transpose(1,2), grad_output)
        return grad_input, grad_weight, None


def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y


def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y


def quantizeLSQ_psum(v, s, p):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    #gradScaleFactor = 1.0 / math.sqrt(v.numel()*Qp)
    #s = round_pass(grad_scale(s, gradScaleFactor))

    vbar = round_pass((v/s).clamp(Qn, Qp))
    #vhat = vbar * s

    return vbar, s


def satmm_cuda_temp(A, X, T=64, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    width=2**b # 256
    max = (width >> signed) - 1 #127 or 255
    min = max - width + 1

    satmm_cuda_psum = satmm_psum.apply
    psum = satmm_cuda_psum(A.contiguous(),X.contiguous(), T)

    out = reduce(lambda x,y: (x+y).clip(min, max), psum.transpose(0,3)).squeeze().transpose(0,-1)
    return out, psum


def satconv2D(image, kernel, padding=0, stride=1, T=64, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    B,Cin,H,W=image.shape
    Cout,_,CH,CW = kernel.shape
    OH = (H - CH + 2 * padding) // stride + 1
    OW = (W - CW + 2 * padding) // stride + 1
    inp_unf = torch.nn.functional.unfold(image, (CH, CW),padding=padding,stride=stride)
    out, psum = satmm_cuda_temp(inp_unf.transpose(1, 2),kernel.view(Cout, -1).t(), T=T, b=b, signed=signed,
                                nbits_psum=nbits_psum, step_size_psum=step_size_psum)
    return out.reshape(B,Cout,OH,OW), psum


def OA(x, b=4):
    mask = (1 << b) - 1
    mask2 = 2**(b-1)

    Qn = -2**(b-1)
    Qp = 2**(b-1)-1

    upper = (x > Qp).float()
    lower = (x < Qn).float()
    middle = 1.0 - upper - lower

    out = x*middle

    out2 = (x*(upper+lower)).int()&mask

    upper2 = (out2 > Qp).float()
    lower2 = (out2 < Qn).float()
    middle2 = 1.0 - upper2 - lower2

    out3 = out2*middle2 + (out2-2*mask2)*upper2 + (out2+2*mask2)*lower2

    return out+out3


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def total_std(stdv1, m1, n1, stdv2, m2, n2):
    left_part = ((n1-1)*(stdv1**2) + (n2-1)*(stdv2**2))/(n1+n2-1)
    right_part = n1*n2*((m1-m2)**2)/((n1+n2)*(n1+n2-1))

    return (left_part+right_part)**0.5

def update_psum_array(psums, psum, my_index):
    psum = psum.flatten()
    n_curr, m_curr, std_curr = psums[my_index]

    n_total = n_curr + psum.numel()
    m_total = (m_curr*n_curr + psum.mean()*psum.numel())/n_total
    std_total = total_std(std_curr, m_curr, n_curr, psum.std(), psum.mean(), psum.numel())

    psums[my_index] = [n_total, m_total.detach().item(), std_total.detach().item()]

    return psums

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, **kwargs):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

        self.nbits_acc = kwargs['nbits_acc']

        self.step_size_psum = kwargs['s']

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        #y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        y, psum = satconv2D(x, binary_weights, self.padding, self.stride,
                      T=64, b=self.nbits_acc, signed=True, nbits_psum=self.nbits_acc,
                      step_size_psum=self.step_size_psum)

        #y = OA(y.int(), b=self.nbits_acc).float() + y - y.int()

        return y*scaling_factor.reshape(1, -1, 1, 1), psum


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits_acc=32, s=8, pn=0):
        super(BasicBlock, self).__init__()

        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride, nbits_acc=nbits_acc, s=s)
        self.bn1 = nn.BatchNorm2d(planes)

        self.pn = pn

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, psums):
        residual = x

        out = self.binary_activation(x)
        out, psum = self.binary_conv(out)

        update_psum_array(psums, psum, self.pn)

        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out, psums

class DownsampleBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(DownsampleBlock, self).__init__()

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=stride)
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.avgpool1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        return out


class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, nbits_acc=32, s=8):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_0 = self._make_layer(block, 64, layers[0], nbits_acc=nbits_acc, s=s, pn=0)
        self.layer1_1 = self._make_layer_2(block, 64, layers[0], nbits_acc=nbits_acc, s=s, pn=1)
        self.layer1_2 = self._make_layer_2(block, 64, layers[0], nbits_acc=nbits_acc, s=s, pn=2)
        self.layer1_3 = self._make_layer_2(block, 64, layers[0], nbits_acc=nbits_acc, s=s, pn=3)
        self.layer2_0 = self._make_layer(block, 128, layers[1], stride=2, nbits_acc=nbits_acc, s=s, pn=4)
        self.layer2_1 = self._make_layer_2(block, 128, layers[1], stride=2, nbits_acc=nbits_acc, s=s, pn=5)
        self.layer2_2 = self._make_layer_2(block, 128, layers[1], stride=2, nbits_acc=nbits_acc, s=s, pn=6)
        self.layer2_3 = self._make_layer_2(block, 128, layers[1], stride=2, nbits_acc=nbits_acc, s=s, pn=7)
        self.layer3_0 = self._make_layer(block, 256, layers[2], stride=2, nbits_acc=nbits_acc, s=s, pn=8)
        self.layer3_1 = self._make_layer_2(block, 256, layers[2], stride=2, nbits_acc=nbits_acc, s=s, pn=9)
        self.layer3_2 = self._make_layer_2(block, 256, layers[2], stride=2, nbits_acc=nbits_acc, s=s, pn=10)
        self.layer3_3 = self._make_layer_2(block, 256, layers[2], stride=2, nbits_acc=nbits_acc, s=s, pn=11)
        self.layer4_0 = self._make_layer(block, 512, layers[3], stride=2, nbits_acc=nbits_acc, s=s, pn=12)
        self.layer4_1 = self._make_layer_2(block, 512, layers[3], stride=2, nbits_acc=nbits_acc, s=s, pn=13)
        self.layer4_2 = self._make_layer_2(block, 512, layers[3], stride=2, nbits_acc=nbits_acc, s=s, pn=14)
        self.layer4_3 = self._make_layer_2(block, 512, layers[3], stride=2, nbits_acc=nbits_acc, s=s, pn=15)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, nbits_acc=32, s=8, pn=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleBlock(self.inplanes, planes * block.expansion, stride)
            '''
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )
            '''

        return block(self.inplanes, planes, stride, downsample, nbits_acc=nbits_acc, s=s, pn=pn)

    def _make_layer_2(self, block, planes, blocks, stride=1, nbits_acc=32, s=8, pn=0):
        self.inplanes = planes * block.expansion

        return block(self.inplanes, planes, nbits_acc=nbits_acc, s=s, pn=pn)

    def forward(self, x, psums):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x, psums = self.layer1_0(x, psums)
        x, psums = self.layer1_1(x, psums)
        x, psums = self.layer1_2(x, psums)
        x, psums = self.layer1_3(x, psums)
        x, psums = self.layer2_0(x, psums)
        x, psums = self.layer2_1(x, psums)
        x, psums = self.layer2_2(x, psums)
        x, psums = self.layer2_3(x, psums)
        x, psums = self.layer3_0(x, psums)
        x, psums = self.layer3_1(x, psums)
        x, psums = self.layer3_2(x, psums)
        x, psums = self.layer3_3(x, psums)
        x, psums = self.layer4_0(x, psums)
        x, psums = self.layer4_1(x, psums)
        x, psums = self.layer4_2(x, psums)
        x, psums = self.layer4_3(x, psums)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, _


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model
