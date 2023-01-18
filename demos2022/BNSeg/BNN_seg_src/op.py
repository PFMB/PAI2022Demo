import numpy as np

def get_camvid_labels():
    """Load the mapping that associates CamVid classes with label colors
    Returns:
        np.ndarray with dimensions (11, 3)
    """

    return np.asarray([[0 ,0, 0], # void 0
                    [128, 128, 128], # sky 1
                    [128, 0, 0], # building 2 
                    [192, 192, 128], # TrafficCone 3
                    [128, 64, 128], # road 4
                    [0, 0, 192], # sidewalk 5
                    [128, 128, 0], # tree 6
                    [192, 128, 128], # sign 7
                    [64, 64, 128], # fence 8
                    [64, 0, 128], # car 9
                     [64, 64, 0],  # Pedestrian 10
                    [0, 128, 192] # Bicycle 11
                    ])
def vis_segments_CamVid(labels):
    ""
    colors = get_camvid_labels()
    # print("labels",np.shape(labels)) # (128, 96, 64)
    # print("max label", np.max(labels))
    # print("min label", np.min(labels))
    batch, height, width = np.shape(labels)

    imgs = np.zeros((batch,height, width, 3), dtype=np.uint8)
    xv, yv = np.meshgrid(np.arange(0, width), np.arange(0, height))
    imgs[:,yv, xv] = colors[labels[:]]
    return imgs