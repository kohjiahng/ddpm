import torch
import matplotlib.pyplot as plt
import logging
# ------------ infer_type convenience function for config logging ------------ #
def infer_type(val):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val

def channel_first(x):
    return torch.permute(x, (0,3,1,2))
def channel_last(x):
    return torch.permute(x, (0,2,3,1))

def plot_images(history, breakpoints = None):
    if breakpoints is None:
        breakpoints = [None] * len(history)
    if len(history) != len(breakpoints):
        logging.warn('plot_images: history and breakpoints are of different lengths')
    n = len(history)
    m = len(history[0])
    fig, ax = plt.subplots(n,m,figsize=(m*8,n*8))

    for t_idx, (t, images) in enumerate(zip(breakpoints, history)):
        images = torch.clamp(images, -1, 1)
        if t:
            ax[t_idx,0].set_ylabel(f"T={int(t)}", visible=True)
        images = channel_last(images)
        for idx, img in enumerate(images):
            ax[t_idx,idx].imshow((img+1)/2)
            ax[t_idx,idx].axis('off')
        fig.subplots_adjust(wspace=0,hspace=0.1)
    return fig
