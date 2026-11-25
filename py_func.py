import numpy as np
import random
import time
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F


torch.set_printoptions(threshold=2000)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.exclusive = kwargs.pop("exclusive")
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer("mask", torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + (not self.exclusive) :] = 0
        self.mask[kh // 2 + 1 :] = 0
        self.weight.data *= self.mask
        self.weight.data = self.weight.data.to(torch.float32)
        self.bias.data = self.bias.data.to(torch.float32)

        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.conv2d(
            x,
            self.mask * self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MadeMask(nn.Module):
    def __init__(self, in_c, out_c, *args, **kwargs) -> None:
        super(MadeMask, self).__init__(*args, **kwargs)
        self.in_c = in_c
        self.out_c = out_c
        self.kernal_size = 5
        self.padding = (self.kernal_size - 1) // 2
        start = time.time()
        self.mask_convi = MaskedConv2d(
            self.in_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=True,
        )
        self.mask_conv1 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv2 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )

        self.mask_convf = MaskedConv2d(
            self.out_c,
            self.in_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.prelu1 = nn.PReLU(dtype=torch.float32)
        self.prelu2 = nn.PReLU(dtype=torch.float32)
        self.prelu3 = nn.PReLU(dtype=torch.float32)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        self.hidden1 = self.prelu1(self.mask_convi(x))
        self.hidden2 = self.prelu2(self.mask_conv1(self.hidden1))
        self.hidden3 = self.prelu3(self.mask_conv2(self.hidden2))

        self.final = self.sig(self.mask_convf(self.hidden3))
        return self.final


class MadeMaskRes(nn.Module):
    def __init__(self, in_c, out_c, *args, **kwargs) -> None:
        super(MadeMaskRes, self).__init__(*args, **kwargs)
        self.in_c = in_c
        self.out_c = out_c
        self.kernal_size = 5
        self.padding = (self.kernal_size - 1) // 2
        start = time.time()
        self.mask_conv1 = MaskedConv2d(
            self.in_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=True,
        )
        self.mask_conv2 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv3 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv4 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv5 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv6 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv7 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv8 = MaskedConv2d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv9 = MaskedConv2d(
            self.out_c,
            self.in_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )

        self.conv2d_1 = nn.Conv2d(self.out_c, self.out_c, 1)
        self.conv2d_2 = nn.Conv2d(self.out_c, self.out_c, 1)
        self.conv2d_3 = nn.Conv2d(self.out_c, self.out_c, 1)
        self.conv2d_4 = nn.Conv2d(self.out_c, self.out_c, 1)
        self.conv2d_5 = nn.Conv2d(self.out_c, self.out_c, 1)
        self.conv2d_6 = nn.Conv2d(self.out_c, self.out_c, 1)
        self.conv2d_7 = nn.Conv2d(self.out_c, self.out_c, 1)
        self.conv2d_8 = nn.Conv2d(self.out_c, self.out_c, 1)

        self.prelu1 = nn.PReLU(dtype=torch.float32)
        self.prelu2 = nn.PReLU(dtype=torch.float32)
        self.prelu3 = nn.PReLU(dtype=torch.float32)
        self.prelu4 = nn.PReLU(dtype=torch.float32)
        self.prelu5 = nn.PReLU(dtype=torch.float32)
        self.prelu6 = nn.PReLU(dtype=torch.float32)
        self.prelu7 = nn.PReLU(dtype=torch.float32)
        self.prelu8 = nn.PReLU(dtype=torch.float32)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        self.hidden1 = self.conv2d_1(self.mask_conv1(x))
        self.hidden1 = self.prelu1(self.hidden1)

        self.hidden2 = self.conv2d_2(self.mask_conv2(self.hidden1))
        self.hidden2 = self.prelu2(self.hidden2)

        self.hidden3 = self.conv2d_3(self.mask_conv3(self.hidden2))
        self.hidden3 = self.prelu3(self.hidden3)

        self.hidden4 = self.conv2d_4(self.mask_conv4(self.hidden3))
        self.hidden4 = self.prelu4(self.hidden4)

        self.hidden5 = self.conv2d_5(self.mask_conv5(self.hidden4))
        self.hidden5 = self.prelu5(self.hidden5)

        self.hidden6 = self.conv2d_6(self.mask_conv6(self.hidden5))
        self.hidden6 = self.prelu6(self.hidden6)

        self.hidden7 = self.conv2d_7(self.mask_conv7(self.hidden6))
        self.hidden7 = self.prelu7(self.hidden7)

        self.hidden8 = self.conv2d_8(self.mask_conv8(self.hidden7))
        self.hidden8 = self.prelu8(self.hidden8)
        self.final = self.sig(self.mask_conv9(self.hidden8))

        return self.final


class MaskedConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        self.exclusive = kwargs.pop("exclusive")
        super(MaskedConv3d, self).__init__(*args, **kwargs)

        _, _, kh, kw, kd = self.weight.shape
        self.register_buffer("mask", torch.ones([kh, kw, kd]))
        self.mask[kh // 2, kw // 2, kd // 2 + (not self.exclusive) :] = 0
        self.mask[kh // 2, kw // 2 + 1 :, :] = 0
        self.mask[kh // 2 + 1 :, :, :] = 0
        self.weight.data *= self.mask
        self.weight.data = self.weight.data.to(torch.float32)
        self.bias.data = self.bias.data.to(torch.float32)

        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.conv3d(
            x,
            self.mask * self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MadeMask3d(nn.Module):
    def __init__(self, in_c, out_c, *args, **kwargs) -> None:
        super(MadeMask3d, self).__init__(*args, **kwargs)
        self.in_c = in_c
        self.out_c = out_c
        self.kernal_size = 3
        self.padding = (self.kernal_size - 1) // 2
        self.mask_convi = MaskedConv3d(
            self.in_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=True,
        )
        self.mask_conv1 = MaskedConv3d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv2 = MaskedConv3d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv3 = MaskedConv3d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_conv4 = MaskedConv3d(
            self.out_c,
            self.out_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.mask_convf = MaskedConv3d(
            self.out_c,
            self.in_c,
            self.kernal_size,
            padding=self.padding,
            exclusive=False,
        )
        self.prelu1 = nn.PReLU(init=0.5, dtype=torch.float32)
        self.prelu2 = nn.PReLU(init=0.5, dtype=torch.float32)
        self.prelu3 = nn.PReLU(init=0.5, dtype=torch.float32)
        self.prelu4 = nn.PReLU(init=0.5, dtype=torch.float32)
        self.prelu5 = nn.PReLU(init=0.5, dtype=torch.float32)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        self.hidden1 = self.prelu1(self.mask_convi(x))
        self.hidden2 = self.prelu2(self.mask_conv1(self.hidden1))
        self.hidden3 = self.prelu3(self.mask_conv2(self.hidden2))
        self.hidden4 = self.prelu4(self.mask_conv3(self.hidden3))
        self.hidden5 = self.prelu5(self.mask_conv4(self.hidden4))
        self.final = self.sig(self.mask_convf(self.hidden4))
        return self.final


class MaskedConv(nn.Module):
    def __init__(self, in_channel, out_channel, mask, **kwargs):
        super().__init__()
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation * (kernel_size[i] - 1) // 2 for i in range(2)])
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=padding, **kwargs
        )
        self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        self.conv.weight.data = self.conv.weight.data.to(torch.float32)
        self.conv.bias.data = self.conv.bias.data.to(torch.float32)
        torch.nn.init.xavier_uniform(self.conv.weight)
        self.conv.weight.data *= self.mask
        self.conv.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())
        return self.conv(x)


class VerticalStackConv(MaskedConv):
    def __init__(
        self, in_channel, out_channel, kernel_size=5, mask_center=False, **kwargs
    ):
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0

        if mask_center:
            mask[kernel_size // 2, :] = 0
        super().__init__(in_channel, out_channel, mask, **kwargs)


class HorizontalStackConv(MaskedConv):
    def __init__(
        self, in_channel, out_channel, kernel_size=5, mask_center=False, **kwargs
    ):
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0
        if mask_center:
            mask[0, kernel_size // 2] = 0
        super().__init__(in_channel, out_channel, mask, **kwargs)


class GatedMaskedConv(nn.Module):
    def __init__(self, in_channel, **kwargs):
        super().__init__()
        self.vert_cv = VerticalStackConv(
            in_channel, out_channel=2 * in_channel, **kwargs
        )
        self.horz_cv = HorizontalStackConv(
            in_channel, out_channel=2 * in_channel, **kwargs
        )

        self.vert_horz_cv = nn.Conv2d(
            2 * in_channel, 2 * in_channel, kernel_size=1, padding=0
        )
        torch.nn.init.xavier_uniform(self.vert_horz_cv.weight)

        self.vert_horz_cv.weight.data = self.vert_horz_cv.weight.data.to(torch.float32)
        self.vert_horz_cv.bias.data = self.vert_horz_cv.bias.data.to(torch.float32)

        self.horz_1_by_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0)
        torch.nn.init.xavier_uniform(self.horz_1_by_1.weight)
        self.horz_1_by_1.weight.data = self.horz_1_by_1.weight.data.to(torch.float32)
        self.horz_1_by_1.bias.data = self.horz_1_by_1.bias.data.to(torch.float32)

        self.prelu2 = nn.PReLU(dtype=torch.float32)
        self.prelu3 = nn.PReLU(dtype=torch.float32)

    def forward(self, v_stack, h_stack):
        v_stack_feat = self.vert_cv(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)

        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        h_stack_feat = self.horz_cv(h_stack)
        h_stack_feat = h_stack_feat + self.vert_horz_cv(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)

        h_stack_out = self.horz_1_by_1(h_stack_feat)
        h_stack_out = h_stack_out

        return v_stack_out, h_stack_out


def linear_index(m, n):
    Mu = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):
                    for p in range(n):
                        x = m * m * m * n * i + m * m * n * j + m * n * k + n * l + p
                        Mu.append(x)
    I_Val_1 = []
    for i in Mu:
        a = i // (m * m * m * n)
        b = (i % (m * m * m * n)) // (m * m * n)
        c = ((i % (m * m * m * n)) % (m * m * n)) // (m * n)
        d = (((i % (m * m * m * n)) % (m * m * n)) % (m * n)) // n
        e = (((i % (m * m * m * n)) % (m * m * n)) % (m * n)) % n
        I_Val_1.append([a, b, c, d, e])
    return torch.as_tensor(I_Val_1)


def ising_linear(m):
    Mu = []
    for i in range(m):
        for j in range(m):
            x = m * i + j
            Mu.append(x)
    I_Val_1 = []
    for i in Mu:
        a = i // (m)
        b = i % (m)
        I_Val_1.append([a, b])
    return torch.as_tensor(I_Val_1)


def linear_index_2(m):
    Mu = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):
                    x = m * m * m * i + m * m * j + m * k + l
                    Mu.append(x)
    Fu = []
    for i in Mu:
        a = i // (m * m * m)
        b = (i % (m * m * m)) // (m * m)
        c = ((i % (m * m * m)) % (m * m)) // m
        d = ((i % (m * m * m)) % (m * m)) % m
        Fu.append([a, b, c, d])
    return torch.as_tensor(Fu)


def matrix_ising(bs, P_list, I_Val, Mu):
    for j, k in zip(range(len(P_list[0, :])), I_Val):
        Mu[:, k[0], k[1]] = P_list[:, j]
    return Mu.type(torch.int64)


def matrix(P_list, I_Val, Mu):
    for i, j in zip(range(len(P_list[0, :])), I_Val):
        Mu[:, j[0], j[1], j[2], j[3]] = P_list[:, i]
    return Mu.type(torch.int64)


def matrix2(j, P_list, I_Val, Mu):
    for j, k in zip(I_Val, range(len(P_list[0, :]))):
        Mu[:, j[0], j[1], j[2]] = P_list[:, k]
    return Mu.type(torch.int64)


def matrix3(j, P_list, I_Val, Mu):
    for j, k in zip(I_Val, range(len(P_list[0, :]))):
        Mu[:, j[0], j[1]] = P_list[:, k]
    return Mu.type(torch.int64)


def matrix3d(j, P_list, I_Val, Mu):
    for j, k in zip(I_Val, range(len(P_list[0, :]))):
        Mu[:, j[0], j[1], j[2], j[3]] = P_list[:, k]
    return Mu.type(torch.int64)


def matrix4d(j, P_list, I_Val, Mu):
    for j, k in zip(I_Val, range(len(P_list[0, :]))):
        Mu[:, j[0], j[1], j[2], j[3], j[4]] = P_list[:, k]
    return Mu.type(torch.int64)


def mat_to_vec(input_tensor):
    reshaped_tensor = input_tensor.view(input_tensor.size(0), -1)
    return reshaped_tensor


def channel_to_mac(bs, m, n, Input_ten, I_Val):
    Input_ten = Input_ten.view(Input_ten.size(0), Input_ten.size(1), -1)
    Mu = torch.zeros((bs, n, m, m))
    for i, j in zip(I_Val, range(len(Input_ten[0, 0, :]))):
        Mu[:, 0, i[0], i[1]] = Input_ten[:, 0, j]
        Mu[:, 1, i[0], i[1]] = Input_ten[:, 1, j]
    return Mu.type(torch.int32)


def add_gauge_fix(Conf, m):
    Conf[:, 0 : m - 1, :, :, :, 0] = 1
    Conf[:, 0, 0 : m - 1, :, :, 1] = 1
    Conf[:, 0, 0, 0 : m - 1, :, 2] = 1
    Conf[:, 0, 0, 0, 0 : m - 1, 3] = 1
    return Conf


def add_polyakov_line(Conf, m):
    Conf[:, m - 1, 0, 0] = 1
    Conf[:, 0, m - 1, 1] = 1
    return Conf


def add_polyakov_line2(Conf, m):
    Conf[:, 0, m - 1, 0] = 1
    Conf[:, 1, 0, m - 1] = 1
    return Conf


def add_polyakov_line3d(Conf, m):
    Conf[:, m - 1, 0, 0, 0] = 1
    Conf[:, 0, m - 1, 0, 1] = 1
    Conf[:, 0, 0, m - 1, 2] = 1
    return Conf


def add_polyakov_line3d2(Conf, m):
    Conf[:, 0, m - 1, 0, 0] = 1
    Conf[:, 1, 0, m - 1, 0] = 1
    Conf[:, 2, 0, 0, m - 1] = 1
    return Conf


def add_polyakov_line4d(Conf, m):
    Conf[:, m - 1, 0, 0, 0, 0] = 1
    Conf[:, 0, m - 1, 0, 0, 1] = 1
    Conf[:, 0, 0, m - 1, 0, 2] = 1
    Conf[:, 0, 0, 0, m - 1, 3] = 1
    return Conf


def power1(x):
    return torch.where(x % 2 == 0, torch.tensor(1), torch.tensor(-1))


def energy3d(j, m, IM, M, a):
    Nu1 = torch.zeros((j, m, m, m, 3))
    Plaq = torch.zeros((j, 3))
    for i in M:
        Nu1[:, i[0], i[1], i[2], 0] = (
            IM[:, 0, i[0], i[1], i[2]]
            * IM[:, 1, (i[0] + 1) % m, i[1], i[2]]
            * IM[:, 0, i[0], (i[1] + 1) % m, i[2]]
            * IM[:, 1, i[0], i[1], i[2]]
        )
        Plaq[:, 0] += 1 - Nu1[:, i[0], i[1], i[2], 0]
        Nu1[:, i[0], i[1], i[2], 1] = (
            IM[:, 0, i[0], i[1], i[2]]
            * IM[:, 2, (i[0] + 1) % m, i[1], i[2]]
            * IM[:, 0, i[0], i[1], (i[2] + 1) % m]
            * IM[:, 2, i[0], i[1], i[2]]
        )
        Plaq[:, 1] += 1 - Nu1[:, i[0], i[1], i[2], 1]
        Nu1[:, i[0], i[1], i[2], 2] = (
            IM[:, 1, i[0], i[1], i[2]]
            * IM[:, 2, i[0], (i[1] + 1) % m, i[2]]
            * IM[:, 1, i[0], i[1], (i[2] + 1) % m]
            * IM[:, 2, i[0], i[1], i[2]]
        )
        Plaq[:, 2] += 1 - Nu1[:, i[0], i[1], i[2], 2]
    Plaq = a * torch.sum(Plaq, dim=1)
    return Plaq


def energy4d(j, m, IM, M, a):
    Nu1 = torch.zeros((j, m, m, m, m, 6))
    Plaq = torch.zeros((j, 6))
    for i in M:
        Nu1[:, i[0], i[1], i[2], i[3], 0] = (
            IM[:, i[0], i[1], i[2], i[3], 0]
            * IM[:, (i[0] + 1) % m, i[1], i[2], i[3], 1]
            * IM[:, i[0], (i[1] + 1) % m, i[2], i[3], 0]
            * IM[:, i[0], i[1], i[2], i[3], 1]
        )
        Plaq[:, 0] += 1 - Nu1[:, i[0], i[1], i[2], i[3], 0]
        Nu1[:, i[0], i[1], i[2], i[3], 1] = (
            IM[:, i[0], i[1], i[2], i[3], 0]
            * IM[:, (i[0] + 1) % m, i[1], i[2], i[3], 2]
            * IM[:, i[0], i[1], (i[2] + 1) % m, i[3], 0]
            * IM[:, i[0], i[1], i[2], i[3], 2]
        )
        Plaq[:, 1] += 1 - Nu1[:, i[0], i[1], i[2], i[3], 1]
        Nu1[:, i[0], i[1], i[2], i[3], 2] = (
            IM[:, i[0], i[1], i[2], i[3], 0]
            * IM[:, (i[0] + 1) % m, i[1], i[2], i[3], 3]
            * IM[:, i[0], i[1], i[2], (i[3] + 1) % m, 0]
            * IM[:, i[0], i[1], i[2], i[3], 3]
        )
        Plaq[:, 2] += 1 - Nu1[:, i[0], i[1], i[2], i[3], 2]
        Nu1[:, i[0], i[1], i[2], i[3], 3] = (
            IM[:, i[0], i[1], i[2], i[3], 1]
            * IM[:, i[0], (i[1] + 1) % m, i[2], i[3], 2]
            * IM[:, i[0], i[1], (i[2] + 1) % m, i[3], 1]
            * IM[:, i[0], i[1], i[2], i[3], 2]
        )
        Plaq[:, 3] += 1 - Nu1[:, i[0], i[1], i[2], i[3], 3]
        Nu1[:, i[0], i[1], i[2], i[3], 4] = (
            IM[:, i[0], i[1], i[2], i[3], 1]
            * IM[:, i[0], (i[1] + 1) % m, i[2], i[3], 3]
            * IM[:, i[0], i[1], i[2], (i[3] + 1) % m, 1]
            * IM[:, i[0], i[1], i[2], i[3], 3]
        )
        Plaq[:, 4] += 1 - Nu1[:, i[0], i[1], i[2], i[3], 4]
        Nu1[:, i[0], i[1], i[2], i[3], 5] = (
            IM[:, i[0], i[1], i[2], i[3], 2]
            * IM[:, i[0], i[1], (i[2] + 1) % m, i[3], 3]
            * IM[:, i[0], i[1], i[2], (i[3] + 1) % m, 2]
            * IM[:, i[0], i[1], i[2], i[3], 3]
        )
        Plaq[:, 5] += 1 - Nu1[:, i[0], i[1], i[2], i[3], 5]
    Plaq = a * torch.sum(Plaq, dim=1)
    return Plaq


def energy2(j, m, IM, M, a, b, ep):
    Nu1 = torch.zeros((j, m, m, 1))
    Plaq = torch.zeros((j, 1))
    t_plaq = torch.zeros((j, len(M)))
    for i in M:
        Nu1[:, i[0], i[1], 0] = (
            IM[:, i[0], i[1], 0]
            * IM[:, (i[0] + 1) % m, i[1], 1]
            * IM[:, i[0], (i[1] + 1) % m, 0]
            * IM[:, i[0], i[1], 1]
        )
        Plaq[:, 0] += 1 - Nu1[:, i[0], i[1], 0]
        t_plaq[:, i] = Nu1[:, i[0], i[1], 0]
        if b == ep - 1:
            print(t_plaq[j - 5 : j, i])
    Plaq = a * torch.sum(Plaq, dim=1)
    return Plaq


def energy22(j, m, IM, M, a, b, ep):
    Nu1 = torch.zeros((j, m, m, 1))
    Plaq = torch.zeros((j, 1))
    t_plaq = torch.zeros((j, len(M)))
    for i, j in zip(M, range(len(M))):
        Nu1[:, i[0], i[1], 0] = (
            IM[:, 0, i[0], i[1]]
            * IM[:, 1, (i[0] + 1) % m, i[1]]
            * IM[:, 0, i[0], (i[1] + 1) % m]
            * IM[:, 1, i[0], i[1]]
        )
        Plaq[:, 0] += 1 - Nu1[:, i[0], i[1], 0]
        t_plaq[:, j] = Nu1[:, i[0], i[1], 0]
    if b == ep - 1:
        print(t_plaq[j - 5 : j, :])
    Plaq = a * torch.sum(Plaq, dim=1)
    return Plaq


def energy_ising2d(j, m, IM, M, a):
    Nu1 = torch.zeros((j, m, m), dtype=torch.float32)
    Plaq = torch.zeros((j, 1), dtype=torch.float32)
    for i in M:
        Nu1[:, i[0], i[1]] = IM[:, i[0], i[1]] * (
            IM[:, (i[0] + 1) % m, i[1]]
            + IM[:, i[0], (i[1] + 1) % m]
            + IM[:, (i[0] - 1) % m, i[1]]
            + IM[:, i[0], (i[1] - 1) % m]
        )
        Plaq[:, 0] -= Nu1[:, i[0], i[1]]

    Plaq = 0.5 * a * Plaq
    return Plaq.type(torch.float32).squeeze()


def energy_ising3d(j: int, m: int, IM: torch.Tensor, M: list, a: float) -> torch.Tensor:
    """Calculate the periodic nearest-neighbour Ising energy for a spin batch."""
    Nu1 = torch.zeros((j, m, m, m), dtype=torch.float32)
    IM = IM.squeeze(1)
    Plaq = torch.zeros((j, 1), dtype=torch.float32)
    for i in M:
        Nu1[:, i[0], i[1], i[2]] = IM[:, i[0], i[1], i[2]] * (
            IM[:, (i[0] + 1) % m, i[1], i[2]]
            + IM[:, i[0], (i[1] + 1) % m, i[2]]
            + IM[:, i[0], i[1], (i[2] + 1) % m]
            + IM[:, (i[0] - 1) % m, i[1], i[2]]
            + IM[:, i[0], (i[1] - 1) % m, i[2]]
            + IM[:, i[0], i[1], (i[2] - 1) % m]
        )
        Plaq[:, 0] -= Nu1[:, i[0], i[1], i[2]]
    Plaq = 0.5 * a * Plaq
    return Plaq.type(torch.float32).squeeze()


def energy_ising4d(j, m, IM, M, a):
    Nu1 = torch.zeros((j, m, m, m, m), dtype=torch.float32)
    Plaq = torch.zeros((j, 1), dtype=torch.float32)
    for i in M:
        Nu1[:, i[0], i[1], i[2], i[3]] = IM[:, i[0], i[1], i[2], i[3]] * (
            IM[:, (i[0] + 1) % m, i[1], i[2], i[3]]
            + IM[:, i[0], (i[1] + 1) % m, i[2], i[3]]
            + IM[:, i[0], i[1], (i[2] + 1) % m, i[3]]
            + IM[:, i[0], i[1], i[2], (i[3] + 1) % m]
        )
        Plaq[:, 0] += Nu1[:, i[0], i[1], i[2], i[3]]
    Plaq = a * Plaq
    return Plaq.type(torch.float32).squeeze()


if __name__ == "__main__":
    main()
