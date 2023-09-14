from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

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
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  0 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  1 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'construction'         , 2 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                , 3 ,        3 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 4 ,        4 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        , 5 ,        5 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 6 ,        6 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'nature'              , 7 ,        7 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 8 ,       8 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 9 ,       9 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 10 ,       10 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 11 ,       11 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'background'            ,  12 ,      255 , 'void'            , 0       , False        , False         , (  0,  0,  0) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]
