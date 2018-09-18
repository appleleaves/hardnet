#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/26 20:04
# @Author  : yjm
# @Site    : 
# @File    : get_network_model.py
# @Software: PyCharm
import networks
import networks_iSORT
import networks_stn
import networks_Compress_BilinearCNN
import networks_Compress_BilinearCNN_pixel
import networks_pixel_relation
import networks_pixel_relationV3
import networks_CDbin_structure_v2 as networks_CDbin_structure_v1
import vgg_feature
import resnet_feature
# import networks_CDbin_structure_v1

import sys
def get_network_model(name,percent=1,overall=1,mode='topk',outnum=128,outpixel=64,x=8,connectmode="concat"):
    print("################# network choosing ################################")
    if(name=='L2NET'):
        model = networks.HardNet()
    elif(name=='L2Net_channelwise_max'):
        model = networks.L2Net_channelwise_max()
    elif(name=='L2Net_channelwise_max_iSORT'):
        model = networks_iSORT.L2Net_channelwise_max_iSORT()
    elif (name=='L2Net_channelwise_max_iSORT_nonlocal3'):
        model = networks_iSORT.L2Net_channelwise_max_iSORT_nonlocal3()
    elif (name=='L2Net_channelwise_max_iSORT_nonlocal4'):
        model = networks_iSORT.L2Net_channelwise_max_iSORT_nonlocal4()
    elif (name=='L2Net_channelwise_max_iSORT_nonlocal5'):
        model = networks_iSORT.L2Net_channelwise_max_iSORT_nonlocal5()
    elif (name == 'L2Net_channelwise_max_iSORT_nonlocal6'):
        model = networks_iSORT.L2Net_channelwise_max_iSORT_nonlocal6()
    elif(name=='L2Net_channelwise_max_nonloacl_1'):
        model = networks.L2Net_channelwise_max_nonloacl_1(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_channelwise_max_nonloacl_2'):
        model = networks.L2Net_channelwise_max_nonloacl_2(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_channelwise_max_nonloacl_3'):
        model = networks.L2Net_channelwise_max_nonloacl_3(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_channelwise_max_nonloacl_4'):
        model = networks.L2Net_channelwise_max_nonloacl_4(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_channelwise_max_nonloacl_5'):
        model = networks.L2Net_channelwise_max_nonloacl_5(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_channelwise_max_nonloacl_6'):
        model = networks.L2Net_channelwise_max_nonloacl_6(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_1'):
        model = networks.L2Net_nonloacl_1(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_2'):
        model = networks.L2Net_nonloacl_2(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_3'):
        model = networks.L2Net_nonloacl_3(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_4'):
        model = networks.L2Net_nonloacl_4(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_5'):
        model = networks.L2Net_nonloacl_5(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_6'):
        model = networks.L2Net_nonloacl_6(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_34'):
        model = networks.L2Net_nonloacl_34(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_35'):
        model = networks.L2Net_nonloacl_35(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_36'):
        model = networks.L2Net_nonloacl_36(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_45'):
        model = networks.L2Net_nonloacl_45(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_46'):
        model = networks.L2Net_nonloacl_46(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_nonloacl_56'):
        model = networks.L2Net_nonloacl_56(mypercent=percent, myoverall=overall)
    elif(name=='L2Net_STN_1'):
        model = networks_stn.L2Net_STN_1()
    elif(name=='L2Net_STN_2'):
        model = networks_stn.L2Net_STN_2()
    elif(name=='L2Net_STN_3'):
        model = networks.L2Net_STN_3()
    elif(name=='L2Net_STN_4'):
        model = networks_stn.L2Net_STN_4()
    elif(name=='L2Net_STN_5'):
        model = networks_stn.L2Net_STN_5()
    elif(name=='L2Net_STN_6'):
        model = networks_stn.L2Net_STN_6()
    elif(name=='L2Net_STN_CM_1'):
        model = networks_stn.L2Net_STN_CM_1()
    elif(name=='L2Net_STN_CM_2'):
        model = networks_stn.L2Net_STN_CM_2()
    elif(name=='L2Net_STN_CM_3'):
        model = networks_stn.L2Net_STN_CM_3()
    elif(name=='L2Net_STN_CM_4'):
        model = networks_stn.L2Net_STN_CM_4()
    elif(name=='L2Net_STN_CM_5'):
        model = networks_stn.L2Net_STN_CM_5()
    elif(name=='L2Net_STN_CM_6'):
        model = networks_stn.L2Net_STN_CM_6()
    elif(name=='L2Net_nonloacl_1_vis'):
        model = networks.L2Net_nonloacl_1_vis()
    elif(name=='L2Net_nonloacl_2_vis'):
        model = networks.L2Net_nonloacl_2_vis()
    elif(name=='L2Net_nonloacl_3_vis'):
        model = networks.L2Net_nonloacl_3_vis()
    elif(name=='L2Net_nonloacl_4_vis'):
        model = networks.L2Net_nonloacl_4_vis()
    elif(name=='L2Net_nonloacl_5_vis'):
        model = networks.L2Net_nonloacl_5_vis()
    elif(name=='L2Net_nonloacl_6_vis'):
        model = networks.L2Net_nonloacl_6_vis()
    elif(name=='L2Net_channelwise_max_nonloacl_1_vis'):
        model = networks.L2Net_channelwise_max_nonloacl_1_vis()
    elif(name=='L2Net_channelwise_max_nonloacl_2_vis'):
        model = networks.L2Net_channelwise_max_nonloacl_2_vis()
    elif(name=='L2Net_channelwise_max_nonloacl_3_vis'):
        model = networks.L2Net_channelwise_max_nonloacl_3_vis()
    elif(name=='L2Net_channelwise_max_nonloacl_4_vis'):
        model = networks.L2Net_channelwise_max_nonloacl_4_vis()
    elif(name=='L2Net_channelwise_max_nonloacl_5_vis'):
        model = networks.L2Net_channelwise_max_nonloacl_5_vis()
    elif(name=='L2Net_channelwise_max_nonloacl_6_vis'):
        model = networks.L2Net_channelwise_max_nonloacl_6_vis()
    elif (name == 'CDbin_NET'):
        model = networks.CDbin_NET()
    elif(name == 'L2Net_Compress_BCNN_pixel'):
        model = networks_Compress_BilinearCNN_pixel.L2Net_Compress_BCNN_pixel()
    elif (name == 'L2Net_Compress_BCNN'):
        model = networks_Compress_BilinearCNN.L2Net_Compress_BCNN(outnum)
    elif (name == 'CDbin_NET_deep5_2'):
        print("model is CDbin_NET_deep5_2_out"+str(outnum))
        model = networks_CDbin_structure_v1.CDbin_NET_deep5_2(outnum)
    elif (name == 'CDbin_NET_deep4_2'):
        print("model is CDbin_NET_deep4_2_out"+str(outnum))
        model = networks_CDbin_structure_v1.CDbin_NET_deep4_2(outnum)
    elif (name == 'CDbin_NET_deep5_1'):
        print("model is CDbin_NET_deep5_1_out"+str(outnum))
        model = networks_CDbin_structure_v1.CDbin_NET_deep5_1(outnum)
    elif (name == 'CDbin_NET_deep4_1'):
        print("model is CDbin_NET_deep4_1_out"+str(outnum))
        model = networks_CDbin_structure_v1.CDbin_NET_deep4_1(outnum)
    elif (name == 'CDbin_NET_deep3'):
        print("model is CDbin_NET_deep3_out"+str(outnum))
        model = networks_CDbin_structure_v1.CDbin_NET_deep3(outnum)
    elif (name == 'CDbin_NET_deep3_2'):
        print("model is CDbin_NET_deep3_2_out"+str(outnum))
        model = networks_CDbin_structure_v1.CDbin_NET_deep3_2(outnum)
    elif (name == 'CDbin_NET_deep2'):
        print("model is CDbin_NET_deep2_out"+str(outnum))
        model = networks_CDbin_structure_v1.CDbin_NET_deep2(outnum)
    elif (name == 'resnet50'):
        print("model is " + name)
        model = resnet_feature.resnet50()
    elif (name == 'resnet34'):
        print("model is " + name)
        model = resnet_feature.resnet34()
    elif (name == 'vgg11'):
        print("model is " + name)
        model = vgg_feature.vgg11()
    elif (name == 'vgg11_bn'):
        print("model is " + name)
        model = vgg_feature.vgg11_bn()
    elif (name == 'vgg13'):
        print("model is " + name)
        model = vgg_feature.vgg13()
    elif (name == 'vgg13_bn'):
        print("model is " + name)
        model = vgg_feature.vgg13_bn()
    elif (name == 'vgg16'):
        print("model is " + name)
        model = vgg_feature.vgg16()
    elif (name == 'vgg16_bn'):
        print("model is " + name)
        model = vgg_feature.vgg16_bn()
    elif (name == 'vgg19'):
        print("model is " + name)
        model = vgg_feature.vgg19()
    elif (name == 'vgg19_bn'):
        print("model is " + name)
        model = vgg_feature.vgg19_bn()
    elif(name=='L2Net_pixel_relation' or name=='pixel_relation'):
        print("model is L2Net_pixel_relation " +
            "\nmode:" + mode +
            "\noutpixel:" + str(outpixel) +
            "\nconnectmode:" + connectmode)
        model = networks_pixel_relation.L2Net_pixel_relation(mode,x = x, outpixel = outpixel, connectmode=connectmode)
    elif (name == 'L2Net_pixel_relation.V3' or name == 'pixel_relation.V3'):
        print("model is L2Net_pixel_relation.V3 " +
              "\nmode:" + mode +
              "\noutpixel:" + str(outpixel) +
              "\nconnectmode:" + connectmode)
        model = networks_pixel_relationV3.L2Net_pixel_relation(mode, x=x, outpixel=outpixel, connectmode=connectmode)
    else:
        # print ('Unknown batch reduce mode. Try L2NET, L2Net_channelwise_max_iSORT, L2Net_channelwise_max, A_channelwise_max, Ax_channelwise_max, G, H, I, L2Net_channelwise_max_nonloacl_6, L2Net_nonloacl_6')
        print("this network is not defined:",name)
        sys.exit(1)
    print("\n################# network choosing ################################")
    return model