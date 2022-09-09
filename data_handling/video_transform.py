import torch
import torch.nn.functional as F

import decord
decord.bridge.set_bridge("torch")

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    
    clip = clip.float().permute(3, 0, 1, 2) / 255.0
    clip = clip * 2 - 1
    return clip

def to_int_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    
    clip = clip.float().permute(3, 0, 1, 2) / 255.0
    clip = clip * 2 - 1
    return clip


class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self):
        return self.__class__.__name__

def pad(tensor, length, dim=0):
    no_dims = len(tensor.size())
    if dim == -1:
        dim = no_dims - 1

    padding = [0] * 2 * no_dims
    padding[2 * (no_dims - dim - 1) + 1] = max(length - tensor.size(dim), 0)
    return F.pad(tensor, padding, "constant", 0)

def extact_int_frames(video_path):
    vr = decord.VideoReader(video_path, height=224, width=224)
    video_tensor = vr.get_batch(range(0, len(vr)))
    video_tensor = video_tensor[:248, :, :, :]
    if video_tensor.size()[0] > 250:
        video_tensor = video_tensor[:250, :, :, :]
    elif video_tensor.size()[0] < 250:
        video_tensor = pad(video_tensor, 250, 0)
    video_tensor = video_tensor.permute(3, 0, 1, 2)
    return video_tensor.numpy()
if __name__ == '__main__':
    input = torch.randn(3,7,2,2)
    output = pad(input, 8, 1)
    pass