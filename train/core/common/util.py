import os

def print_network(net, verbose=False, name=""):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    num_params = num_params / 1e6
    if verbose:
        print(net)
    if hasattr(net, 'flops'):
        flops = net.flops()
        print(f"number of GFLOPs: {flops / 1e9}")
    print('network:{} Total number of parameters: {:.2f}M'.format(name, num_params))


def get_iou(boxA, boxB):
    """
    boxA, boxB: (x0, y0, x1, y1)
    Computes Intersection over Union.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea <= 0 or boxBArea <= 0:
        return 0.0

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def get_file_path_list(path, ext=None):
    assert os.path.exists(path), 'path not exist {}'.format(path)
    assert ext is not None, 'ext is None'
    if os.path.isfile(path):
        return [path]
    file_path_list = []
    for root, _, files in os.walk(path):
        for file in files:
            try:
                if file.rsplit('.')[-1].lower() in ext:
                    file_path_list.append(os.path.join(root, file))
            except Exception as e:
                pass
    return file_path_list


# Function to calculate IoU (Jaccard Index)
def calculate_iou(boxA, boxB):
    # Get coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate the area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou