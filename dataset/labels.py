#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Zpark labels"""

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class
    'clsId'       ,

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ])


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# labels = [
#     #     name           clsId    id   trainId   category  catId  hasInstanceignoreInEval   color
#     Label('其他'        ,    0 ,    0,    0   , '其他'    ,   0  ,False , True  , 0x000000 ),
#     Label('采集车'      , 0x01 ,    1,    0   , '其他'    ,   0  ,False , True  , 0X000000 ),
#     Label('天空'        , 0x11 ,   17,    1    , '天空'    ,   1  ,False , False , 0x4682B4 ),
#     Label('小汽车'      , 0x21 ,   33,    2    , '移动物体',   2  ,False , False , 0x00008E ),
#     Label('小汽车群'    , 0xA1 ,  161,    2    , '移动物体',   2  ,False , False , 0x00008E ),
#     Label('摩托车'      , 0x22 ,   34,    32    , '移动物体',   2  ,False , False , 0x0000E6 ),
#     Label('摩托车群'    , 0xA2 ,  162,    32    , '移动物体',   2  ,False , False , 0x0000E6 ),
#     Label('自行车'      , 0x23 ,   35,    3    , '移动物体',   2  ,False , False , 0x770B20 ),
#     Label('自行车群'    , 0xA3 ,  163,    3    , '移动物体',   2  ,False , False , 0x770B20 ),
#     Label('行人'        , 0x24 ,   36,    4    , '移动物体',   2  ,False , False , 0x0080c0 ),
#     Label('行人群'      , 0xA4 ,  164,    4    , '移动物体',   2  ,False , False , 0x0080c0 ),
#     Label('骑行'        , 0x25 ,   37,    5    , '移动物体',   2  ,False , False , 0x804080 ),
#     Label('骑行群'      , 0xA5 ,  165,    5    , '移动物体',   2  ,False , False , 0x804080 ),
#     Label('卡车'        , 0x26 ,   38,    33    , '移动物体',   2  ,False , False , 0x8000c0 ),
#     Label('卡车群'      , 0xA6 ,  166,    33    , '移动物体',   2  ,False , False , 0x8000c0 ),
#     Label('公交车'      , 0x27 ,   39,    34    , '移动物体',   2  ,False , False , 0xc00040 ),
#     Label('公交车群'    , 0xA7 ,  167,    34    , '移动物体',   2  ,False , False , 0xc00040 ),
#     Label('三轮车'      , 0x28 ,   40,    35    , '移动物体',   2  ,False , False , 0x8080c0 ),
#     Label('三轮车群'    , 0xA8 ,  168,    35    , '移动物体',   2  ,False , False , 0x8080c0 ),
#     Label('公路'        , 0x31 ,   49,    7    , '平面'    ,   3  ,False , False , 0xc080c0 ),
#     Label('人行道'      , 0x32 ,   50,    8    , '平面'    ,   3  ,False , False , 0xc08040 ),
#     Label('交通锥'      , 0x41 ,   65,    14   , '路间障碍',   4  ,False , False , 0x000040 ),
#     Label('路桩'        , 0x42 ,   66,    15   , '路间障碍',   4  ,False , False , 0x0000c0 ),
#     Label('栅栏'        , 0x43 ,   67,    16   , '路间障碍',   4  ,False , False , 0x404080 ),
#     Label('交通灯'      , 0x51 ,   81,    19   , '路边物体',   5  ,False , False , 0xc04080 ),
#     Label('杆'          , 0x52 ,   82,    20   , '路边物体',   5  ,False , False , 0xc08080 ),
#     Label('交通标识'    , 0x53 ,   83,    21   , '路边物体',   5  ,False , False , 0x004040 ),
#     Label('墙'          , 0x54 ,   84,    17   , '路边物体',   5  ,False , False , 0xc0c080 ),
#     Label('垃圾箱'      , 0x55 ,   85,    18   , '路边物体',   5  ,False , False , 0x4000c0 ),
#     Label('广告牌'      , 0x56 ,   86,    22   , '路边物体',   5  ,False , False , 0xc000c0 ),
#     Label('建筑'        , 0x61 ,   97,    25   , '建筑'    ,   6  ,False , False , 0xc00080 ),
#     Label('桥'          , 0x62 ,   98,    0    , '建筑'    ,   6  ,False , True  , 0x808000 ),
#     Label('隧道'        , 0x63 ,   99,    0    , '建筑'    ,   6  ,False , True  , 0x800000 ),
#     Label('天桥'        , 0x64 ,  100,    0    , '建筑'    ,   6  ,False , True  , 0x408040 ),
#     Label('植被'        , 0x71 ,  113,    31   , '自然'    ,   7  ,False , False , 0x808040 ),
#     Label('未标注'      , 0xFF ,  255,    0    , '未标注'  ,   8  ,False , True  , 0xFFFFFF ),
# ]

labels = [
    #     name           clsId    id   trainId   category  catId  hasInstanceignoreInEval   color
    Label('其他'        ,    0 ,    0,    0   , '其他'    ,   0  ,False , True  , 0x87CEEB ),
    Label('采集车'      , 0x01 ,    1,    0   , '其他'    ,   0  ,False , True  , 0x87CEEB ),
    Label('天空'        , 0x11 ,   17,    1    , '天空'    ,   1  ,False , False , 0x87CEEB ),
    Label('小汽车'      , 0x21 ,   33,    2    , '移动物体',   2  ,False , False , 0x00008E ),
    Label('小汽车群'    , 0xA1 ,  161,    2    , '移动物体',   2  ,False , False , 0x00008E ),
    Label('摩托车'      , 0x22 ,   34,    32    , '移动物体',   2  ,False , False , 0x0000E6 ),
    Label('摩托车群'    , 0xA2 ,  162,    32    , '移动物体',   2  ,False , False , 0x0000E6 ),
    Label('自行车'      , 0x23 ,   35,    3    , '移动物体',   2  ,False , False , 0x4080c0 ),
    Label('自行车群'    , 0xA3 ,  163,    3    , '移动物体',   2  ,False , False , 0x770B20 ),
    Label('行人'        , 0x24 ,   36,    4    , '移动物体',   2  ,False , False , 0xDC143C  ),
    Label('行人群'      , 0xA4 ,  164,    4    , '移动物体',   2  ,False , False , 0xDC143C ),
    Label('骑行'        , 0x25 ,   37,    5    , '移动物体',   2  ,False , False , 0x804080 ),
    Label('骑行群'      , 0xA5 ,  165,    5    , '移动物体',   2  ,False , False , 0x804080 ),
    Label('卡车'        , 0x26 ,   38,    33    , '移动物体',   2  ,False , False , 0xc080c0 ),
    Label('卡车群'      , 0xA6 ,  166,    33    , '移动物体',   2  ,False , False , 0xc080c0 ),
    Label('公交车'      , 0x27 ,   39,    34    , '移动物体',   2  ,False , False , 0xc08040 ),
    Label('公交车群'    , 0xA7 ,  167,    34    , '移动物体',   2  ,False , False , 0xc08040 ),
    Label('三轮车'      , 0x28 ,   40,    35    , '移动物体',   2  ,False , False , 0x8080c0 ),
    Label('三轮车群'    , 0xA8 ,  168,    35    , '移动物体',   2  ,False , False , 0x8080c0 ),
    Label('公路'        , 0x31 ,   49,    7    , '平面'    ,   3  ,False , False , 0x8000c0 ),
    Label('人行道'      , 0x32 ,   50,    8    , '平面'    ,   3  ,False , False , 0xc00040 ),
    Label('交通锥'      , 0x41 ,   65,    14   , '路间障碍',   4  ,False , False , 0x000040 ),
    Label('路桩'        , 0x42 ,   66,    15   , '路间障碍',   4  ,False , False , 0x0000c0 ),
    Label('栅栏'        , 0x43 ,   67,    16   , '路间障碍',   4  ,False , False , 0x404080 ),
    Label('交通灯'      , 0x51 ,   81,    19   , '路边物体',   5  ,False , False , 0xc04080 ),
    Label('杆'          , 0x52 ,   82,    20   , '路边物体',   5  ,False , False , 0xc08080 ),
    Label('交通标识'    , 0x53 ,   83,    21   , '路边物体',   5  ,False , False , 0x004040 ),
    Label('墙'          , 0x54 ,   84,    17   , '路边物体',   5  ,False , False , 0xc0c080 ),
    Label('垃圾箱'      , 0x55 ,   85,    18   , '路边物体',   5  ,False , False , 0x4000c0 ),
    Label('广告牌'      , 0x56 ,   86,    22   , '路边物体',   5  ,False , False , 0xc000c0 ),
    Label('建筑'        , 0x61 ,   97,    25   , '建筑'    ,   6  ,False , False , 0x800000 ),
    Label('桥'          , 0x62 ,   98,    0    , '建筑'    ,   6  ,False , True  , 0x808000 ),
    Label('隧道'        , 0x63 ,   99,    0    , '建筑'    ,   6  ,False , True  , 0x800000 ),
    Label('天桥'        , 0x64 ,  100,    0    , '建筑'    ,   6  ,False , True  , 0x408040 ),
    Label('植被'        , 0x71 ,  113,    31   , '自然'    ,   7  ,False , False , 0x40c000 ),
    Label('未标注'      , 0xFF ,  255,    1    , '未标注'  ,   8  ,False , True  , 0x87CEEB ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!
id2trainId = {label.id: label.trainId for label in labels}

# name to label object
name2label      = {label.name: label for label in labels}
# id to label object
id2label        = {label.id: label for label in labels}
# trainId to label object
trainId2label   = {label.trainId: label for label in reversed(labels)}
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

color2label = {}
label2color = {}

max_label = 0
for label in labels:
    #color = (int(label.color[2:4],16),int(label.color[4:6],16),int(label.color[6:8],16))
    color = label.color
    r =  color // (256*256)
    g = (color-256*256*r) // 256
    b = (color-256*256*r-256*g)
    color2label[(r, g, b)] = [label]
    label2color[label.trainId] = [r, g, b]
    if label.trainId > max_label:
        max_label = label.trainId

color_list = []
for i in range(max_label + 1):
    if i in label2color:
        color_list.append(label2color[i])
    else:
        color_list.append([0, 0, 0])


#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

""" returns the label name that describes a single instance (if possible)
 e.g.     input     |   output
        ----------------------
          car       |   car
          cargroup  |   car
          foo       |   None
          foogroup  |   None
          skygroup  |   None
"""
def assureSingleInstanceName(name):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} |".format('name', 'id', 'trainId', 'category')\
    +  "{:>10} | {:>12} | {:>12}".format('categoryId', 'hasInstances', 'ignoreInEval'))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} |".format(label.name, label.id, label.trainId)\
        + "  {:>14} |{:>10} ".format(label.category, label.categoryId)\
        + "| {:>12} | {:>12}".format(label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = '机动车'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format(name=name, id=id))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format(id=id, category=category))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format(id=trainId, name=name))
